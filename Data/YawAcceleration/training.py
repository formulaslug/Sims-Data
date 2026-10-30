import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_simpson

from Data.FSLib.AnalysisFunctions import read, simpleTimeCol


def normalizePressureLong(tirePressure):
    # Only long because lat doesn't use it
    lambda_pressurescalarlong = 0.501448392868042

    return (tirePressure - lambda_pressurescalarlong * tirePressure) / (lambda_pressurescalarlong * tirePressure)

def normalizeLoadLong(normalForce):
    lambda_loadscalarlong = 1.1716148853302002

    return (normalForce - lambda_loadscalarlong * normalForce) / (lambda_loadscalarlong * normalForce)

def calculateLongCompositeLongFrictionScalingFactor(velocityX):
    unloaded_radius = 1.7333 # was unloaded-radius
    friction_coeff_long = 1.7333 # was friction-coeff-long

    sqrt_term = np.sqrt(9.81 * unloaded_radius)
    velocity_ratio = abs(velocityX) / sqrt_term
    denominator = 1.0 + friction_coeff_long * velocity_ratio
    # Add epsilon to prevent division by zero
    eps = 1e-8
    return friction_coeff_long / (denominator + eps)

def getLongFrictionCoefficient(normalForce, velocityX, tirePressure):
    p_dx1 = 0.642947793006897
    p_dx2 = 0.9363532066345215
    p_dx3 = 0
    p_px3 = -0.9542407989501953
    p_px4 = -0.9245550036430359
    camber = 0

    term1 = (p_dx1 + p_dx2 * normalizeLoadLong(normalForce))
    term2 = (1.0 + p_px3 * normalizePressureLong(tirePressure) + p_px4 * normalizePressureLong(tirePressure) ** 2)
    scaling_factor = calculateLongCompositeLongFrictionScalingFactor(velocityX)
    camber_term = np.abs(camber) ** scaling_factor
    term3 = (1.0 - (p_dx3 * camber_term))
    print(f"{term1 * term2 * term3=}")
    return term1 * term2 * term3

def getMaxLongFriction(normalForce, velocityX, tirePressure):
    zeta_1 = 0.6045444011688232

    return getLongFrictionCoefficient(normalForce, velocityX, tirePressure) * normalForce * zeta_1

def getStiffnessFactorLong(normalForce, velocityX, tirePressure):
    p_px1 = 0.013196180574595928
    p_px2 = 0.8705743551254272
    p_kx1 = 21.55539894104004
    p_kx2 = 13.510042190551758
    p_kx3 = -0.5750095248222351
    shape_factor = 0.696268618106842 # var is "shape-factor"
    
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

def calcYawAcceleration(yawRate, bodySlip, speed, steeringAngle, tireNormalForces = np.array([100, 100, 100, 100]), tirePressures = np.array([12, 12, 12, 12])):
    # tire forces defined as FL, FR, BL, BR
    velocityX = np.cos(bodySlip) * speed
    frontCorneringStiffness = getStiffnessFactorLong(np.sum(tireNormalForces[:2], axis=0), velocityX, np.sum(tirePressures[:2], axis=0))
    rearCorneringStiffness = getStiffnessFactorLong(np.sum(tireNormalForces[2:], axis=0), velocityX, np.sum(tirePressures[2:], axis=0))
    wheelBase = 1.65471
    frontAxleDistanceToCOM = 0.853506
    rearAxleDistanceToCOM = wheelBase - frontAxleDistanceToCOM
    MOI = 658.088580080000

    component1 = (frontCorneringStiffness * frontAxleDistanceToCOM - rearCorneringStiffness * rearAxleDistanceToCOM) * bodySlip
    component2 = yawRate / speed * (frontCorneringStiffness * frontAxleDistanceToCOM**2 + rearCorneringStiffness * rearAxleDistanceToCOM**2)
    component3 = -frontCorneringStiffness * frontAxleDistanceToCOM * steeringAngle
    return (component1 + component2 + component3) / MOI

def calcBodySlipAcceleration()

df = read("FS-3/08172025/08172025_20_Endurance1P1.parquet", fs2or3=True)
df = df.insert_column(0, simpleTimeCol(df))
t = df["Time"]

[x for x in df.columns if "STEER" in x]

yawRate = (df["VDM_Z_AXIS_YAW_RATE"] - 2.3) / 360 * 2 * np.pi
speed = df["VDM_GPS_SPEED"]


yaw_integrated = cumulative_simpson((df['VDM_Z_AXIS_YAW_RATE'] - 2.87), t)
plt.plot(t, df["VDM_GPS_TRUE_COURSE"])
plt.plot(t[:-1], (yaw_integrated + 110)%360)
plt.show()

plt.plot(t, df["VDM_Z_AXIS_YAW_RATE"])
plt.show()

a = 50
n = a**2

yawRate = np.ones(n) * 5/360*2*np.pi
bodySlip = np.ones(n) * 5/360*2*np.pi
speedA = np.linspace(0, 30, a)
steeringAngleB = np.linspace(-40/360*2*np.pi, 40/360*2*np.pi, a)
tireNormalForces = np.ones((4, n)) * 280 * 9.81 / 4
tirePressures = np.ones((4, n)) * 12

speed, steeringAngle = np.meshgrid(speedA, steeringAngleB)
speed = speed.reshape(n)
steeringAngle = steeringAngle.reshape(n)

yaw_acceleration = calcYawAcceleration(yawRate, bodySlip, speed, steeringAngle, tireNormalForces, tirePressures)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(speed, steeringAngle, yaw_acceleration, s=0.2)
ax.set_xlabel("Speed")
ax.set_ylabel("Steering Angle")
ax.set_zlabel("Yaw acceleration")
fig.show()