# Steering model
import numpy as np
import matplotlib.pyplot as plt

def calculateSlipAngle(yawRate, velocity, steerAngle, parameters):
    speed = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2 + velocity[2]**2)
    if yawRate == 0 or speed == 0: # WRONG. RELAXATION LENGTH. PROJECT
        return (0, 0)
    else:
        bodySlip = np.arctan(velocity[1]/velocity[0])

    frontSlipAngle = calculateVirtualSlipAngle(parameters) + bodySlip + (parameters["wheelBase"]*parameters["frontWeightDist"]/100 * yawRate)/speed - steerAngle
    rearSlipAngle = bodySlip - (parameters["wheelBase"]*(100-parameters["frontWeightDist"])/100 * yawRate)/speed

    return (frontSlipAngle, rearSlipAngle)

def calculateVirtualSlipAngle(parameters):
    return 0

def calculateYawRate(currYawRate, speed, stepSteerInput, timeSinceLastSteer, frontCorneringStiffnessDeg_, rearCorneringStiffnessDeg_, parameters):
    frontCorneringStiffnessDeg = -140
    rearCorneringStiffnessDeg = -140

    if speed == 0 or stepSteerInput == 0:
        return 0

    CF = frontCorneringStiffnessDeg * 180 / np.pi
    CR = rearCorneringStiffnessDeg * 180 / np.pi
    a = parameters['a']
    b = parameters["wheelBase"] - a
    m = parameters["Mass"]
    I = parameters["polarMoment"]
    Y_beta = CF + CR
    Y_delta = -CF
    N_beta = a * CF - b * CR
    N_delta = -1 * a * CF
    NR_v = a**2 * CF + b**2 * CR
    YR_v = a * CF - b * CR
    c = -(NR_v / speed + (I * Y_beta) / (m * speed))
    k = N_beta + (Y_beta * NR_v - N_beta * YR_v) / (m * speed**2)
    C2 = (Y_delta * N_beta - Y_beta * N_delta) / (m * speed)
    r_inf = (C2 * stepSteerInput) / k
    r_dot_0 = N_delta * stepSteerInput / I
    omega_n = np.sqrt(abs(k / I))
    Cc = 2 * I * omega_n
    zeta = c / Cc

    if zeta < 1: # Underdamped
        omega_d = np.sqrt(1 - zeta**2) * omega_n
        A = -r_inf
        B = (r_dot_0 - zeta * omega_n * r_inf) / omega_d
        exp_term = np.exp(-zeta * omega_n * timeSinceLastSteer)
        cos_term = A * np.cos(omega_d * timeSinceLastSteer)
        sin_term = B * np.sin(omega_d * timeSinceLastSteer)
        normalizedR = exp_term * (cos_term + sin_term) + r_inf
    elif zeta > 1: # Overdamped
        f = (-zeta - np.sqrt(zeta**2 - 1)) * omega_n
        g = (-zeta + np.sqrt(zeta**2 - 1)) * omega_n
        A = (r_dot_0 + r_inf * f) / (g - f)
        B = -(A + r_inf)
        r = A * np.exp(g * timeSinceLastSteer) + B * np.exp(f * timeSinceLastSteer) + r_inf
        normalizedR = r / r_inf
    else: # Critically
        term1 = (-1* (CF * stepSteerInput * a)/(I * r_inf) - omega_n)
        normalizedR = (-1 + term1 * timeSinceLastSteer) * np.e **(-1 * omega_n * timeSinceLastSteer) + 1

    return normalizedR * r_inf

def yawRateToEnergy(yaw_rate, parameters):
    """
    Convert yaw rate to rotational kinetic energy
    E = 0.5 * I * omega^2
    where I is polar moment of inertia and omega is yaw rate
    """
    I = parameters["polarMoment"]
    energy = 0.5 * I * (yaw_rate ** 2)
    return energy


parameters = {
    "ambientTemperature": 20,
    "wheelCircumferance": 1.35716802635079,
    "wheelRadius": 0.216,
    "gearRatio": 3.33333333,
    "maxTorque": 180,
    "friction-coeff-lat": 1.7333,
    "friction-coeff-long": 1.7333,
    "unloaded-radius": 1.7333,
    "p_0": 82000,
    "load_0": 300,
    "tractiveIMax": 300,
    "Mass": 300,
    "wheelBase": 1.65471,
    "a": 0.853506,
    "frontWeightDist": 46.46,
    "CoG-height": 0.999628,
    "CoG-distanceToRollAxis": 0.999628,
    "polarMoment": 658.088580080000,
    "rollSteerCoefficient": 0,
    "rollCamberSteerCoefficient": 0,
    "casterLength": 0,
    "frontToe": 0,
    "brakeSpecificHeatCapacity": 450,
    "brakeThermalConductivity": 50,
    "brakeSurfaceArea": 0.001180643,
    "brakepadThickness": 0.007874,
    "brakeMass": 0.408,
    "maxBrakeForce": 1500
}

# Generate data for plotting
steer_angles = np.linspace(0, 30, 50)  # Steering angles in degrees
velocities = np.linspace(5, 50, 50)    # Velocities in m/s
timeSinceLastSteer = 0.5  # Fixed time for steady-state analysis

# Create meshgrid
STEER, VEL = np.meshgrid(steer_angles, velocities)
YAW_RATE = np.zeros_like(STEER)
STEERING_ENERGY = np.zeros_like(STEER)

# Calculate yaw rate and energy for each combination
for i in range(len(velocities)):
    for j in range(len(steer_angles)):
        steer_rad = np.deg2rad(steer_angles[j])
        yaw_rate = calculateYawRate(
            0,
            velocities[i],
            steer_rad,
            timeSinceLastSteer,
            -1086.083,
            -890.0656,
            parameters
        )
        YAW_RATE[i, j] = yaw_rate
        STEERING_ENERGY[i, j] = yawRateToEnergy(yaw_rate, parameters)

# Create 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(STEER, VEL, STEERING_ENERGY, cmap='viridis',
                       edgecolor='none', alpha=0.8)

ax.set_xlabel('Steering Angle (degrees)', fontsize=12, labelpad=10)
ax.set_ylabel('Velocity (m/s)', fontsize=12, labelpad=10)
ax.set_zlabel('Rotational Energy (Joules)', fontsize=12, labelpad=10)
ax.set_title('Steering-Induced Rotational Energy vs Steering Angle and Velocity', fontsize=14, pad=20)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Energy (J)')

ax.view_init(elev=25, azim=45)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
