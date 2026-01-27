from paramLoader import Parameters, Magic
import numpy as np
from state import SF, VehicleState
from Mech.braking import calcBrakeCooling, calcBrakeHeating
from scipy.integrate import RK45

# Vibe coded but it looks about right so idk.
# TODO: Verify that this is correct
def calculateHeading(heading, yaw_rate, time_increment):
    initial_heading = heading[:2]
    rotation_angle = yaw_rate * time_increment
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)

    rotation_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta,  cos_theta]
    ])

    new_heading = rotation_matrix @ initial_heading

    new_heading = new_heading / np.linalg.norm(new_heading)

    return np.append(new_heading, 0)

def stepState(worldPrev:VehicleState, inputs):

    # Empirically we see that throttle can only go from about 0-.75.
    # TODO: Update later
    # Made it so you can just comment this out when it's fixed.
    # Throttle, brakesFront, brakesRear, steering angle
    # 0-1, PSI, PSI, Radians
    delta = 1/Parameters["stepsPerSecond"]

    maxTraction = 180.0 # Needs a more complex implementation before being used. Potentially something akin to the gaussian kernel of the voltage histeresis model but for acceleration? Or literally based on the suspension travel.
    voltage = SF.voltage() # Not yet implemented. Returns 120 for now.
    maxPower = SF.maxPower(voltage) # Watts
    resistiveForces= SF.resistiveForces(worldPrev, inputs[1])
    frontBrakeHeating, rearBrakeHeating = calcBrakeHeating(worldPrev, inputs)
    frontBrakeCooling, rearBrakeCooling = calcBrakeCooling(worldPrev, inputs)
    maxMotorTorque = SF.maxMotorTorque(worldPrev, resistiveForces, maxPower, maxTraction)
    motorTorque = max(Parameters["maxTorque"]*inputs[0], maxMotorTorque) # Nm
    power = motorTorque * worldPrev.motorRotationsHZ # Watts
    motorForce = SF.motorForce(motorTorque) # Newtons
    netForce = motorForce + resistiveForces # Newtons
    acceleration = netForce / Parameters["Mass"] # m/s^2
    current = power/voltage # Amps

    charge = worldPrev.charge - current * delta / 3600.0
    position = worldPrev.position + worldPrev.velocity * delta
    speed = max(0, worldPrev.speed + acceleration * delta) # Sometimes braking falls a tad below 0 so we just correct that because otherwise everything breaks
    yawRate = worldPrev.yawRate
    if inputs[2] == 0:
        yawRate = 0
    heading = calculateHeading(worldPrev.heading, yawRate, delta)
    

    worldNext = VehicleState(
        position=position,
        speed=speed, 
        heading = heading,
        charge=charge,
        frontBrakeTemperature = frontBrakeTemperature,
        rearBrakeTemperature = rearBrakeTemperature,
        yawRate = worldPrev.yawRate
    )
    return worldNext
