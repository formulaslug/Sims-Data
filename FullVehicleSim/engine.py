import numpy as np

from .Electrical.powertrain import (
    calcCurrent,
    calcMaxMotorTorque,
    calcMaxPower,
    calcMaxWheelTorque,
    calcMotorForce,
    calcVoltage,
)
from .Mech.aero import calcDownForce, calcDrag
from .Mech.braking import calcBrakeCooling, calcBrakeForce, calcBrakeHeating
from .Mech.general import calcResistiveForces
from .Mech.steering import (
    calcBodySlip,
    calcBodySlipVelocity,
    calcFrontLateralSlipVelocity,
    calcLateralDynamicsRates,
    calcLongitudinalVelocityRate,
    calcRearLateralSlipVelocity,
    getDefaultTirePressures,
    stepTransientSlipAngle,
)
from .Mech.tireLoad import calcLoadTransfer
from .paramLoader import *

# REFERENCE FRAMES
# ----------------
# World (inertial) frame: X, Y ground plane, Z up. Used for: varPos*, varVel*, varHeading*, varYawAngle.
# Vehicle (body) frame:   x forward, y LEFT, z up (ISO 8855 / Pacejka). Used for: varLongitudinalVelocity,
#                         varLateralVelocity, varAccelerationX/Y (what an IMU reads), slip angles, tyre forces.
# Yaw angle psi is measured CCW from world +X. Yaw rate, steer angle and body slip are + CCW / + left.
#
# The dynamics are integrated in the body frame, then rotated into the world frame for position:
#   V_world = R(psi) @ [Vx, Vy],  R(psi) = [[cos, -sin], [sin, cos]]

def bodyToWorld(vector:np.ndarray, yawAngle:float) -> np.ndarray:
    c, s = np.cos(yawAngle), np.sin(yawAngle)
    return np.array([c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]])

def worldToBody(vector:np.ndarray, yawAngle:float) -> np.ndarray:
    return bodyToWorld(vector, -yawAngle)

def stepState(worldArray:np.ndarray, step:int) -> np.ndarray:
    """
    The order by which things get updated in this function is incredibly important.
    If you calculate velocity before you calculate acceleration,
    you would just wind up using the 0 that is present in the array at that index.
    Update acceleration before you update velocity. This will also fail somewhat
    silently so be cautious.

    The worldArray will also fail silently if it doen't contain a row before step.
    The 0-1 evaluates to -1 so it just grabs the last row in the array instead of the
    previous one.

    :param worldArray: The main world array. To work properly, the worldArray needs to contain a row (step) and a previous row (step-1) with the same format as the output of this function. The function will read from the previous row to calculate the new values for the current step.
    :type worldArray: np.ndarray
    :param step: The current step in the simulation
    :type step: int
    :return: The updated state array for the current step
    :rtype: ndarray[Any, Any]
    """

    # Empirically we see that throttle can only go from about 0-.75. Currently not accounted for.
    arr = np.copy(worldArray[step, :]) # This is the array that is updated and then returned.
    prev = worldArray[step-1, :]
    delta = 1/Parameters["stepsPerSecond"]

    # Previous dynamic state (body frame unless noted)
    velocityX = float(prev[varLongitudinalVelocity])
    lateralVelocity = float(prev[varLateralVelocity])
    yawRate = float(prev[varYawRate])
    yawAngle = float(prev[varYawAngle]) # world frame
    frontSlipAngle = float(prev[varFrontSlipAngle]) # transient (relaxation-length) slip angle state
    rearSlipAngle = float(prev[varRearSlipAngle])
    steerAngle = float(arr[varSteerAngle]) / Parameters["steeringRatio"] # ROAD WHEEL angle, rad, + left. varSteerAngle is the steering wheel angle

    arr[varMaxTraction] = 180.0 # Needs a more complex implementation before being used. Potentially something akin to the gaussian kernel of the voltage histeresis model but for acceleration? Or literally based on the suspension travel.
    arr[varVoltage] = calcVoltage(worldArray, step) # Volts, Voltage of the entire pack, not a single cell
    arr[varMaxPower] = calcMaxPower(arr[varVoltage]) # Watts

    arr[varResistiveForces] = calcResistiveForces(worldArray, step)
    arr[varFrontBrakeHeating], arr[varRearBrakeHeating] = calcBrakeHeating(worldArray, step)
    arr[varFrontBrakeCooling], arr[varRearBrakeCooling] = calcBrakeCooling(worldArray, step)
    arr[varFrontBrakeForce], arr[varRearBrakeForce] = calcBrakeForce(worldArray, step)
    arr[varFrontBrakeTemperature] = prev[varFrontBrakeTemperature] + arr[varFrontBrakeHeating] - arr[varFrontBrakeCooling]
    arr[varRearBrakeTemperature] = prev[varRearBrakeTemperature] + arr[varRearBrakeHeating] - arr[varRearBrakeCooling]

    arr[varMaxMotorTorque] = calcMaxMotorTorque(worldArray, step, arr[varResistiveForces], arr[varMaxPower], arr[varMaxTraction]) # Nm
    arr[varMotorTorque] = min(Parameters["maxTorque"]*(arr[varThrottle]-arr[varBrakePedalTravel]), arr[varMaxMotorTorque]) # Nm

    arr[varPower] = arr[varMotorTorque] * prev[varMotorRotationsHZ] # Watts
    arr[varMotorForce] = calcMotorForce(arr[varMotorTorque]) # Newtons, Longitudinal force at the wheel from the motor
    arr[varNetForce] = arr[varMotorForce] + arr[varResistiveForces] # Newtons, body x

    arr[varCurrent] = calcCurrent(arr[varPower], arr[varVoltage]) # Amps. clamps for current limit. pack current.

    arr[varCharge] = prev[varCharge] - worldArray[step, varCurrent] * delta / 3600.0 / Parameters["parallelCells"] / Parameters["cellCapacity_Ah"]

    arr[varFLWheelLoad:varBRWheelLoad+1] = calcLoadTransfer(prev[varAccelerationX], prev[varAccelerationY], yawRate) + calcDownForce(prev[varVelX], prev[varVelY], prev[varHeadingX], prev[varHeadingY])
    tireNormalForces = arr[varFLWheelLoad:varBRWheelLoad+1].astype(float)

    lat = calcLateralDynamicsRates(velocityX, lateralVelocity, yawRate, steerAngle, frontSlipAngle, rearSlipAngle,
                                   tireNormalForces, getDefaultTirePressures())
    arr[varFrontLateralForce] = lat.frontLateralForce
    arr[varRearLateralForce] = lat.rearLateralForce
    arr[varYawAcceleration] = lat.yawAcceleration

    # Body-frame accelerations (IMU readings). Steered front tyre force has a rearward component FyF*sin(delta)
    # (Pacejka Eq. 1.34a, p. 20: m(du/dt - r*v) = Fx1 - Fy1*delta + Fx2).
    arr[varAccelerationX] = (arr[varNetForce] - lat.frontLateralForce * np.sin(steerAngle)) / Parameters["Mass"] # m/s^2
    arr[varAccelerationY] = lat.lateralAcceleration # m/s^2
    arr[varAccelerationZ] = prev[varAccelerationZ] # TODO: When suspension modeling added do this maybe?
    longitudinalVelocityRate = calcLongitudinalVelocityRate(arr[varAccelerationX], lateralVelocity, yawRate) # dVx/dt = a_x + Vy*r

    # ---- Semi-implicit Euler: velocities first, then use the NEW velocities for the slip angle & position updates ----
    newYawRate = yawRate + lat.yawAcceleration * delta
    newLateralVelocity = lateralVelocity + lat.lateralVelocityRate * delta
    newVelocityX = velocityX + longitudinalVelocityRate * delta
    if velocityX >= 0.0 and newVelocityX < 0.0:
        # Brakes/regen/drag can stop the car but not drive it backwards (no reverse gear).
        newVelocityX = 0.0
    newYawAngle = yawAngle + newYawRate * delta

    arr[varYawRate] = newYawRate
    arr[varLateralVelocity] = newLateralVelocity
    arr[varLongitudinalVelocity] = newVelocityX
    arr[varYawAngle] = newYawAngle

    # Transient slip angles: implicit Euler on Pacejka Eq. 7.18 (p. 333, stable at any speed) + Eq. 7.25 limit, using the new velocities
    arr[varFrontSlipAngle] = stepTransientSlipAngle(frontSlipAngle,
        calcFrontLateralSlipVelocity(newLateralVelocity, newYawRate, steerAngle, newVelocityX),
        newVelocityX, lat.frontPeakSlipAngle, delta)
    arr[varRearSlipAngle] = stepTransientSlipAngle(rearSlipAngle,
        calcRearLateralSlipVelocity(newLateralVelocity, newYawRate),
        newVelocityX, lat.rearPeakSlipAngle, delta)

    # Body slip is an OUTPUT (never integrated, nothing downstream uses it). Meaningless near Vx = 0.
    arr[varBodySlipAngle] = calcBodySlip(newLateralVelocity, newVelocityX)
    arr[varBodySlipVelocity] = calcBodySlipVelocity(newLateralVelocity, lat.lateralVelocityRate, newVelocityX, longitudinalVelocityRate)

    # ---- Body -> world ----
    arr[varHeadingX:varHeadingZ+1] = np.array([np.cos(newYawAngle), np.sin(newYawAngle), 0.0])
    arr[varVelX:varVelY+1] = bodyToWorld(np.array([newVelocityX, newLateralVelocity]), newYawAngle)
    arr[varVelZ] = prev[varVelZ] + arr[varAccelerationZ] * delta
    arr[varSpeed] = np.linalg.norm(arr[varVelX:varVelZ+1], 2)
    arr[varPosX:varPosZ+1] = prev[varPosX:varPosZ+1] + arr[varVelX:varVelZ+1] * delta

    arr[varDrag] = calcDrag(worldArray, step)

    arr[varMaxWheelTorque] = calcMaxWheelTorque(arr[varMaxMotorTorque])
    arr[varWheelRotationsHZ] = newVelocityX / Parameters["wheelRadius"] # rad/s. Wheels roll with the body-x velocity (no slip ratio yet)
    arr[varMotorRotationsHZ] = arr[varWheelRotationsHZ] * Parameters["gearRatio"]
    arr[varWheelRPM] = arr[varWheelRotationsHZ] * 60.0 / (2 * np.pi)
    arr[varMotorRPM] = arr[varWheelRPM] * Parameters["gearRatio"]
    return arr

def dynamicStepState(step:np.ndarray) -> np.ndarray:
    """
    Interface for simulation step state that takes a single row with inputs and a previous step.
    Separates this into 2 rows with the first row being the previous step and the second row being the current inputs.
    Then calls stepState to get the new state for the current step.

    :param step: Step you wish to input (Array of all features)
    :type step: np.ndarray
    :return: Next step in the simulation given your contorl input and vehicle state.
    :rtype: ndarray[Any, Any]
    """
    arr = np.array([step, step])
    arr[1, varSteerAngle+1:] = 0 # Keep inputs (time..steerAngle), clear the state to be computed
    return stepState(arr, 1)
