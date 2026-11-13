import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_simpson

from Data.FSLib.AnalysisFunctions import read, simpleTimeCol
from FullVehicleSim.paramLoader import Magic, Parameters


def normalizePressureLong(tirePressure):
    # Only long because lat doesn't use it
    lambda_pressurescalarlong = Magic["lambda_pressurescalarlong"]

    return (tirePressure - lambda_pressurescalarlong * tirePressure) / (lambda_pressurescalarlong * tirePressure)

def normalizeLoadLong(normalForce):
    lambda_loadscalarlong = Magic["lambda_loadscalarlong"]

    return (normalForce - lambda_loadscalarlong * normalForce) / (lambda_loadscalarlong * normalForce)

def calculateLongCompositeLongFrictionScalingFactor(velocityX):
    unloaded_radius = Parameters["unloaded-radius"]
    friction_coeff_long = Parameters["friction-coeff-long"]

    sqrt_term = np.sqrt(9.81 * unloaded_radius)
    velocity_ratio = abs(velocityX) / sqrt_term
    denominator = 1.0 + friction_coeff_long * velocity_ratio
    # Add epsilon to prevent division by zero
    eps = 1e-8
    return friction_coeff_long / (denominator + eps)

def getLongFrictionCoefficient(normalForce, velocityX, tirePressure):
    p_dx1 = Magic["p_dx1"]
    p_dx2 = Magic["p_dx2"]
    p_dx3 = Magic["p_dx3"]
    p_px3 = Magic["p_px3"]
    p_px4 = Magic["p_px4"]
    camber = 0

    term1 = (p_dx1 + p_dx2 * normalizeLoadLong(normalForce))
    term2 = (1.0 + p_px3 * normalizePressureLong(tirePressure) + p_px4 * normalizePressureLong(tirePressure) ** 2)
    scaling_factor = calculateLongCompositeLongFrictionScalingFactor(velocityX)
    camber_term = np.abs(camber) ** scaling_factor
    term3 = (1.0 - (p_dx3 * camber_term))
    print(f"{term1 * term2 * term3=}")
    return term1 * term2 * term3

def getMaxLongFriction(normalForce, velocityX, tirePressure):
    zeta_1 = Magic["zeta_1"]

    return getLongFrictionCoefficient(normalForce, velocityX, tirePressure) * normalForce * zeta_1

def getStiffnessFactorLong(normalForce, velocityX, tirePressure):
    p_px1 = Magic["p_px1"]
    p_px2 = Magic["p_px2"]
    p_kx1 = Magic["p_kx1"]
    p_kx2 = Magic["p_kx2"]
    p_kx3 = Magic["p_kx3"]
    shape_factor = Magic["shape-factor"]
    
    Cx = shape_factor
    Dx = getMaxLongFriction(normalForce, velocityX, tirePressure)    
    exp_term = p_kx3 * normalizeLoadLong(normalForce)
    Kx = normalForce * (p_kx1 + p_kx2 * normalizeLoadLong(normalForce)) * np.exp(exp_term) * (1.0 + p_px1 * normalizePressureLong(tirePressure) + p_px2 * normalizePressureLong(tirePressure) ** 2)
    denominator = Cx * Dx
    # Add epsilon to prevent division by zero
    print(f"{Cx=}")
    print(f"{Dx=}")
    print(f"{denominator=}")
    print(f"{Kx=}")
    print(f"{normalForce=}")
    eps = 1e-8
    return Kx / (denominator + eps)

def calcYawAcceleration(yawRate, bodySlip, speed, steeringAngle, velocityX, tireNormalForces = None, tirePressures = None):
    tireNormalForces = np.array([55*9.81] * 4) if type(tireNormalForces) is None else tireNormalForces 
    tirePressures = np.array([12] * 4) if type(tirePressures) is None else tirePressures 
    frontCorneringStiffness = getStiffnessFactorLong(np.mean(tireNormalForces[:2], axis=0), velocityX, np.mean(tirePressures[:2], axis=0))
    rearCorneringStiffness = getStiffnessFactorLong(np.mean(tireNormalForces[2:], axis=0), velocityX, np.mean(tirePressures[2:], axis=0))
    frontAxleDistanceToCOM = Parameters["a"]
    rearAxleDistanceToCOM = Parameters["wheelBase"] - Parameters["a"]
    MOI = Parameters["polarMoment"]

    component1 = (frontCorneringStiffness * frontAxleDistanceToCOM - rearCorneringStiffness * rearAxleDistanceToCOM) * bodySlip
    component2 = yawRate / speed * (frontCorneringStiffness * frontAxleDistanceToCOM**2 + rearCorneringStiffness * rearAxleDistanceToCOM**2)
    component3 = -frontCorneringStiffness * frontAxleDistanceToCOM * steeringAngle
    return (component1 + component2 + component3) / MOI

def calcBodySlipVelocity(yawRate, bodySlip, speed, steeringAngle, velocityX, tireNormalForces = None, tirePressures = None):
    tireNormalForces = np.array([55*9.81] * 4) if type(tireNormalForces) is None else tireNormalForces 
    tirePressures = np.array([12] * 4) if type(tirePressures) is None else tirePressures 
    frontCorneringStiffness = getStiffnessFactorLong(np.mean(tireNormalForces[:2], axis=0), velocityX, np.mean(tirePressures[:2], axis=0))
    rearCorneringStiffness = getStiffnessFactorLong(np.mean(tireNormalForces[2:], axis=0), velocityX, np.mean(tirePressures[2:], axis=0))
    frontAxleDistanceToCOM = Parameters["a"]
    rearAxleDistanceToCOM = Parameters["wheelBase"] - Parameters["a"]
    mass = Parameters["Mass"]

    component1 = (frontCorneringStiffness + rearCorneringStiffness) * bodySlip
    component2 = yawRate / speed * frontCorneringStiffness * frontAxleDistanceToCOM - rearCorneringStiffness * rearAxleDistanceToCOM
    component3 = -frontCorneringStiffness * steeringAngle
    component4 = mass * speed
    bodySlipVelocity = ((component1 + component2 + component3) / component4) - yawRate
    return bodySlipVelocity

df = read("FS-3/08172025/08172025_20_Endurance1P1.parquet", fs2or3=True)
df = df.insert_column(0, simpleTimeCol(df))
t = df["Time"]

[x for x in df.columns if "VDM" in x]

yawRate = (df["VDM_Z_AXIS_YAW_RATE"] - 2.3) / 360 * 2 * np.pi
speed = df["VDM_GPS_SPEED"]
yaw_integrated = cumulative_simpson((df['VDM_Z_AXIS_YAW_RATE'] - 2.87), x=t)

a = 50
n = a**2

bodySlip = np.ones(n) * 5/360*2*np.pi
speed = np.ones(n) * 1
yawRateA = np.linspace(-60/360*2*np.pi, 60/360*2*np.pi, a) 
steeringAngleB = np.linspace(-40/360*2*np.pi, 40/360*2*np.pi, a)
tireNormalForces = np.ones((4, n)) * 280 * 9.81 / 4
tirePressures = np.ones((4, n)) * 12

yawRate, steeringAngle = np.meshgrid(yawRateA, steeringAngleB)
yawRate = yawRate.reshape(n)
steeringAngle = steeringAngle.reshape(n)

yaw_acceleration = calcYawAcceleration(yawRate, bodySlip, speed, steeringAngle, speed, tireNormalForces, tirePressures)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(yawRate, steeringAngle, yaw_acceleration, s=0.2)
ax.set_xlabel("yawRate")
ax.set_ylabel("Steering Angle")
ax.set_zlabel("Yaw acceleration")
fig.show()

bodySlip = np.ones(n) * 5/360*2*np.pi
steeringAngle = np.zeros(n)
yawRateA = np.linspace(-60/360*2*np.pi, 60/360*2*np.pi, a) 
speedA = np.linspace(-15, 15, a)
tireNormalForces = np.ones((4, n)) * 280 * 9.81 / 4
tirePressures = np.ones((4, n)) * 12

yawRate, speed = np.meshgrid(yawRateA, speedA)
yawRate = yawRate.reshape(n)
speed = speed.reshape(n)

yaw_acceleration = calcYawAcceleration(yawRate, bodySlip, speed, steeringAngle, speed, tireNormalForces, tirePressures)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(yawRate, speed, yaw_acceleration, s=0.2)
ax.set_xlabel("yawRate")
ax.set_ylabel("Speed")
ax.set_zlabel("Yaw acceleration")
fig.show()