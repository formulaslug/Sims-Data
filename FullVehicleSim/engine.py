from paramLoader import *
import numpy as np
from numpy.typing import NDArray
from Mech.braking import calcBrakeCooling, calcBrakeHeating, calcBrakeForce
from Mech.aero import calcDrag, calcDownForce
from Mech.steering import calcSlipAngle
from Mech.general import calcResistiveForces
from Electrical.powertrain import calcCurrent, calcMaxMotorTorque, calcMaxWheelTorque, calcMotorForce, calcMaxPower, calcVoltage
from yaw_rate_model.double_bicycle_model import calcYawRate
from scipy.integrate import RK45

# Vibe coded but it looks about right so idk.
# TODO: Verify that this is correct
def calculateHeading(worldArray:NDArray[np.float64], step:int) -> NDArray[np.float64]:
    time_increment = 1/Parameters["stepsPerSecond"]
    initial_heading = worldArray[step-1, varHeadingX:varHeadingZ] # Yes this removes Z, we just want X and Y for this simplification
    rotation_angle = worldArray[step-1, varYawRate] * time_increment
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)

    rotation_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta,  cos_theta]
    ])

    new_heading = rotation_matrix @ initial_heading
    new_heading = new_heading / np.linalg.norm(new_heading)

    return np.append(new_heading, 0)

def stepState(worldArray:NDArray[np.float64], step:int) -> NDArray[np.float64]:
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
    delta = 1/Parameters["stepsPerSecond"]

    arr[varMaxTraction] = 180.0 # Needs a more complex implementation before being used. Potentially something akin to the gaussian kernel of the voltage histeresis model but for acceleration? Or literally based on the suspension travel.
    arr[varVoltage] = calcVoltage(worldArray, step) # Volts, Voltage of the entire pack, not a single cell
    arr[varMaxPower] = calcMaxPower(arr[varVoltage]) # Watts
    
    arr[varResistiveForces] = calcResistiveForces(worldArray, step)
    arr[varFrontBrakeHeating], arr[varRearBrakeHeating] = calcBrakeHeating(worldArray, step)
    arr[varFrontBrakeCooling], arr[varRearBrakeCooling] = calcBrakeCooling(worldArray, step)
    arr[varFrontBrakeForce], arr[varRearBrakeForce] = calcBrakeForce(worldArray, step)
    arr[varFrontBrakeTemperature] = worldArray[step-1, varFrontBrakeTemperature] + arr[varFrontBrakeHeating] - arr[varFrontBrakeCooling]
    arr[varRearBrakeTemperature] = worldArray[step-1, varRearBrakeTemperature] + arr[varRearBrakeHeating] - arr[varRearBrakeCooling]
    
    arr[varMaxMotorTorque] = calcMaxMotorTorque(worldArray, step, arr[varResistiveForces], arr[varMaxPower], arr[varMaxTraction]) # Nm
    arr[varMotorTorque] = min(Parameters["maxTorque"]*(arr[varThrottle]-arr[varBrakePedalTravel]), arr[varMaxMotorTorque]) # Nm
    
    arr[varPower] = arr[varMotorTorque] * worldArray[step-1, varMotorRotationsHZ] # Watts
    arr[varMotorForce] = calcMotorForce(arr[varMotorTorque]) # Newtons, Longitudinal force at the wheel from the motor
    arr[varNetForce] = arr[varMotorForce] + arr[varResistiveForces] # Newtons
    
    arr[varAcceleration] = arr[varNetForce] / Parameters["Mass"] # m/s^2
    
    arr[varCurrent] = calcCurrent(arr[varPower], arr[varVoltage]) # Amps. clamps for current limit. pack current.

    arr[varCharge] = worldArray[step-1, varCharge] - worldArray[step, varCurrent] * delta / 3600.0 / Parameters["parallelCells"] / Parameters["cellCapacity_Ah"]
    arr[varPosX:varPosZ+1] = worldArray[step-1, varPosX:varPosZ+1] + worldArray[step-1, varVelX:varVelZ+1] * delta
    arr[varSpeed] = max(0, worldArray[step-1, varSpeed] + arr[varAcceleration] * delta) # Sometimes braking falls a tad below 0 so we just correct that because otherwise everything breaks
    arr[varLateralVelocty], arr[varYawRate] = calcYawRate(worldArray, step)
    arr[varVelX:varVelZ+1] = arr[varSpeed] * worldArray[step-1, varHeadingX:varHeadingZ+1]
    arr[varHeadingX:varHeadingZ+1] = calculateHeading(worldArray, step)

    arr[varDrag] = calcDrag(worldArray, step)
    
    arr[varFrontSlipAngle], arr[varRearSlipAngle] = calcSlipAngle(worldArray, step)
    arr[varMaxWheelTorque] = calcMaxWheelTorque(arr[varMaxMotorTorque])
    arr[varWheelRotationsHZ] = arr[varSpeed] / Parameters["wheelRadius"]
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
    arr[1, 5:] = np.zeros((step.shape[0]-5))
    return stepState(arr, 1)