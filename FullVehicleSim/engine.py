import numpy as np
from state import VehicleState
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

def rk4(t_i, y_i, dt, f):
    k1 = f(t_i, y_i) 
    k2 = f(t_i + dt/2, y_i + k1*dt/2)
    k3 = f(t_i + dt/2, y_i + k2*dt/2)
    k4 = f(t_i + dt, y_i + k3*dt) 
    return y_i + dt/6*(k1 + 2*k2 + 2*k3 +  k4)

def rk4_implementation(worldPrev, delta):
    state = np.array([
        worldPrev.charge
        worldPrev.position
        worldPrev.speed
    ])

    derivatives = np.array([
        charge = worldPrev.charge - worldPrev.current * delta / 3600.0
        position = worldPrev.position + worldPrev.velocity * delta
        speed = worldPrev.speed + worldPrev.acceleration * delta
    ])

    return rk4(0, state, delta, derivatives)

def stepState(worldPrev, inputsRaw, delta, timeSinceLastSteer, initSpeed, integrationMethod):

    if integrationMethod == "RK45":
        
        inputs = [inputsRaw[0] * 0.75, inputsRaw[1], inputsRaw[2]]

        charge = worldPrev.charge - worldPrev.current * delta / 3600.0
        position = worldPrev.position + worldPrev.velocity * delta
        speed = worldPrev.speed + worldPrev.acceleration * delta
        yawRate = worldPrev.yawRate
        if inputs[2] == 0:
            yawRate = 0
        heading = calculateHeading(worldPrev.heading, yawRate, delta)
        acceleration = worldPrev.acceleration

        worldNext = VehicleState(
            stepSize = delta,
            position=position,
            speed=max(0,speed), # Sometimes braking falls a tad below 0 so we just correct that because otherwise everything breaks
            acceleration=acceleration,
            heading = heading,
            charge=charge,
            lastCurrent=worldPrev.current,
            throttle = inputs[0],
            brakes = inputs[1],
            yawRate = worldPrev.yawRate,
            steerAngle = inputs[2],
            brakeTemperature = worldPrev.cooledBrakeTemperature,
            timeSinceLastSteer = timeSinceLastSteer,
            initSpeed = initSpeed
        )
        return worldNext

    elif integrationMethod == "RS":
        # Empirically we see that throttle can only go from about 0-.75.
        # TODO: Update later
        inputs = [inputsRaw[0] * 0.75, inputsRaw[1], inputsRaw[2]]

        charge = worldPrev.charge - worldPrev.current * delta / 3600.0
        position = worldPrev.position + worldPrev.velocity * delta
        speed = worldPrev.speed + worldPrev.acceleration * delta
        yawRate = worldPrev.yawRate
        if inputs[2] == 0:
            yawRate = 0
        heading = calculateHeading(worldPrev.heading, yawRate, delta)
        acceleration = worldPrev.acceleration

        worldNext = VehicleState(
            stepSize = delta,
            position=position,
            speed=max(0,speed), # Sometimes braking falls a tad below 0 so we just correct that because otherwise everything breaks
            acceleration=acceleration,
            heading = heading,
            charge=charge,
            lastCurrent=worldPrev.current,
            throttle = inputs[0],
            brakes = inputs[1],
            yawRate = worldPrev.yawRate,
            steerAngle = inputs[2],
            brakeTemperature = worldPrev.cooledBrakeTemperature,
            timeSinceLastSteer = timeSinceLastSteer,
            initSpeed = initSpeed
        )
        return worldNext
