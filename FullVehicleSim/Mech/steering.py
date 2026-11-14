from typing import NamedTuple

import numpy as np

from ..paramLoader import Magic, Parameters, TireFront, TireRear

# References (page numbers are the printed book pages):
#   [Pacejka] H. B. Pacejka, Tire and Vehicle Dynamics, 3rd ed., Butterworth-Heinemann (2012).
#   [Balkwill] J. Balkwill, Performance Vehicle Dynamics, Butterworth-Heinemann (2018).
#
# SIGN CONVENTION: Pacejka defines tan(a) = -Vsy/|Vx| and Fy = +C*a' (Eqs. 7.14, 7.18, p. 333).
# This file uses a' with the opposite sign (a'_here = -a'_Pacejka) so that Fy = -C*a' and the
# cornering stiffness C stays positive. Every equation below is the Pacejka one with that flip applied.

def normalizePressureLat(tirePressure):
    # [Pacejka] Eq. 4.E2b, p. 177: dpi = (p - p0) / p0. Pressures in Pa.
    nominalPressure = Parameters["p_0"]

    return (tirePressure - nominalPressure) / nominalPressure

def normalizeLoadLat(normalForce):
    # [Pacejka] Eq. 4.E1, 4.E2a, p. 177: Fz0' = lambda_Fz0 * Fz0,  dfz = (Fz - Fz0') / Fz0'
    nominalLoad = Magic["lambda_nominalload"] * Parameters["load_0"]

    return (normalForce - nominalLoad) / nominalLoad

def calcCompositeFrictionScalingFactor(slipSpeed, frictionScaling=1.0):
    """
    [Pacejka] Eq. 4.E7, p. 179: lambda*_mu = lambda_mu / (1 + lambda_muV * Vs / V0)

    Friction that decays with SLIP speed Vs (sliding speed of the contact patch, NOT vehicle speed).
    lambda_muV = 0 turns it off (Pacejka's default, p. 178); > 0 models e.g. a wet surface.
    V0 is the reference velocity = sqrt(g * R0), R0 = unloaded tyre radius ([Pacejka] p. 176).

    :param slipSpeed: magnitude of the contact patch slip velocity, m/s
    :param frictionScaling: lambda_mu (lambda_mux or lambda_muy), the peak friction scaling factor
    """
    referenceVelocity = np.sqrt(9.81 * Parameters["unloaded-radius"])

    return frictionScaling / (1.0 + Magic["lambda_muV"] * abs(slipSpeed) / referenceVelocity)

def getLongFrictionCoefficient(normalForce, slipSpeed, tirePressure, camber=0.0):
    # [Pacejka] Eq. 4.E13, p. 179:
    # mu_x = (p_Dx1 + p_Dx2*dfz) * (1 + p_px3*dpi + p_px4*dpi^2) * (1 - p_Dx3*gamma^2) * lambda*_mux
    p_dx1 = Magic["p_dx1"]
    p_dx2 = Magic["p_dx2"]
    p_dx3 = Magic["p_dx3"]
    p_px3 = Magic["p_px3"]
    p_px4 = Magic["p_px4"]
    dpi = normalizePressureLat(tirePressure)

    loadTerm = (p_dx1 + p_dx2 * normalizeLoadLat(normalForce))
    pressureTerm = (1.0 + p_px3 * dpi + p_px4 * dpi ** 2)
    camberTerm = (1.0 - p_dx3 * camber ** 2)
    frictionScaling = calcCompositeFrictionScalingFactor(slipSpeed) # Eq. 4.E7 with lambda_mux = 1
    return loadTerm * pressureTerm * camberTerm * frictionScaling

def getMaxLongFriction(normalForce, slipSpeed, tirePressure):
    # [Pacejka] Eq. 4.E12, p. 179: Dx = mu_x * Fz * zeta_1
    zeta_1 = Magic["zeta_1"]

    return getLongFrictionCoefficient(normalForce, slipSpeed, tirePressure) * normalForce * zeta_1

def calcDegressiveFrictionScalingFactor(compositeFrictionScaling):
    # [Pacejka] Eq. 4.E8, p. 179: lambda'_mu = A_mu * lambda*_mu / (1 + (A_mu - 1) * lambda*_mu), A_mu = 10 suggested.
    # Used on the vertical shift S_Vy so it doesn't vanish as fast as the peak force when friction drops.
    A_mu = 10.0
    return A_mu * compositeFrictionScaling / (1.0 + (A_mu - 1.0) * compositeFrictionScaling)

def getCorneringStiffness(normalForce, tirePressure, tireFit:dict, camber=0.0):
    """
    [Pacejka] Eq. 4.E25, p. 180: K_ya = dFy/da at a = 0 (= C_Fa when camber = 0). Positive, N/rad, ONE tyre.
    K_ya = p_Ky1*Fz0' * (1 + p_py1*dpi) * (1 - p_Ky3*|gamma*|)
           * sin(p_Ky4 * atan((Fz/Fz0') / ((p_Ky2 + p_Ky5*gamma*^2) * (1 + p_py2*dpi)))) * zeta_3 * lambda_Kya
    Not scaled by the friction factors: "the slip stiffnesses are not affected through these changes" (p. 179).

    :param tireFit: TireFront / TireRear from paramLoader (made by TireModel/fitPureLateral.py)
    """
    nominalLoad = tireFit["nominalLoad"]
    dpi = (tirePressure - tireFit["nominalPressure"]) / tireFit["nominalPressure"] # 4.E2b
    camberTerm = np.sin(camber) # 4.E4: gamma* = sin(gamma)
    denominator = (tireFit["p_ky2"] + tireFit.get("p_ky5", 0.0) * camberTerm ** 2) * (1.0 + tireFit["p_py2"] * dpi)

    stiffness = (tireFit["p_ky1"] * nominalLoad
                 * (1.0 + tireFit["p_py1"] * dpi)
                 * (1.0 - tireFit.get("p_ky3", 0.0) * np.abs(camberTerm))
                 * np.sin(tireFit["p_ky4"] * np.arctan(normalForce / nominalLoad / denominator)))
    return abs(float(stiffness))

def getPeakLateralForce(normalForce, tirePressure, tireFit:dict, slipSpeed=0.0):
    """
    Peak factor D_y of ONE tyre, [Pacejka] Eqs. 4.E22-4.E23, p. 179 (camber 0):
        D_y = mu_y * Fz,  mu_y = (p_Dy1 + p_Dy2*dfz) * (1 + p_py3*dpi + p_py4*dpi^2) * lambda*_muy
    lambda*_muy (Eq. 4.E7) = tireFrictionScaling (TTC belt -> track) with the slip-speed decay.
    """
    nominalLoad = tireFit["nominalLoad"]
    dfz = (normalForce - nominalLoad) / nominalLoad # 4.E2a
    dpi = (tirePressure - tireFit["nominalPressure"]) / tireFit["nominalPressure"] # 4.E2b
    frictionScaling = calcCompositeFrictionScalingFactor(slipSpeed, Parameters["tireFrictionScaling"])
    mu = (tireFit["p_dy1"] + tireFit["p_dy2"] * dfz) * (1.0 + tireFit["p_py3"] * dpi + tireFit["p_py4"] * dpi ** 2) * frictionScaling
    return max(mu * normalForce, 0.0)

def calcMagicFormulaLateralForce(slipAngle, normalForce, tirePressure, tireFit:dict, slipSpeed=0.0):
    """
    Pure-slip Magic Formula lateral force of ONE tyre, [Pacejka] Eqs. 4.E19-4.E29, pp. 179-180 (camber 0),
    in PACEJKA's sign (Fy = +K*alpha). Driven by the transient slip angle as in [Pacejka] Eq. 7.22, p. 335.

    Only the odd (antisymmetric) part is used: 0.5*(F(a) - F(-a)). That drops the conicity / ply-steer
    offsets S_Hy, S_Vy, which on a real car largely cancel left vs right; keeping them with a single fit
    for both sides would push the car sideways on a straight.
    """
    if normalForce <= 0.0:
        return 0.0
    nominalLoad = tireFit["nominalLoad"]
    dfz = (normalForce - nominalLoad) / nominalLoad
    Cy = tireFit["p_cy1"] # 4.E21
    Dy = getPeakLateralForce(normalForce, tirePressure, tireFit, slipSpeed) # 4.E22
    Ky = getCorneringStiffness(normalForce, tirePressure, tireFit) # 4.E25
    By = Ky / (Cy * Dy + 1e-6) # 4.E26 (+ epsilon as Pacejka suggests, p. 177)
    frictionScaling = calcCompositeFrictionScalingFactor(slipSpeed, Parameters["tireFrictionScaling"])
    SHy = tireFit["p_hy1"] + tireFit["p_hy2"] * dfz # 4.E27 (camber 0)
    SVy = normalForce * (tireFit["p_vy1"] + tireFit["p_vy2"] * dfz) * calcDegressiveFrictionScalingFactor(frictionScaling) # 4.E29 (camber 0)

    def magicFormula(alpha):
        alphaY = np.tan(alpha) + SHy # 4.E3 (alpha* = tan(alpha)), 4.E20
        Ey = min((tireFit["p_ey1"] + tireFit["p_ey2"] * dfz) * (1.0 - tireFit["p_ey3"] * np.sign(alphaY)), 1.0) # 4.E24 (<= 1)
        BA = By * alphaY
        return Dy * np.sin(Cy * np.arctan(BA - Ey * (BA - np.arctan(BA)))) + SVy # 4.E19

    return 0.5 * (magicFormula(slipAngle) - magicFormula(-slipAngle))

def getAxleCorneringStiffness(tireNormalForces, tirePressures):
    # Sum of the two tyres on each axle -> (front, rear). Single-track lumping, [Pacejka] Section 1.3, p. 22.
    front = getCorneringStiffness(tireNormalForces[0], tirePressures[0], TireFront) + getCorneringStiffness(tireNormalForces[1], tirePressures[1], TireFront)
    rear = getCorneringStiffness(tireNormalForces[2], tirePressures[2], TireRear) + getCorneringStiffness(tireNormalForces[3], tirePressures[3], TireRear)
    return front, rear

def getAxlePeakLateralForce(tireNormalForces, tirePressures, tireFit:dict, slipSpeed=0.0):
    # D_y of the two tyres on one axle. tireNormalForces/tirePressures: [left, right]
    return (getPeakLateralForce(tireNormalForces[0], tirePressures[0], tireFit, slipSpeed)
            + getPeakLateralForce(tireNormalForces[1], tirePressures[1], tireFit, slipSpeed))

def getPeakSlipAngle(axlePeakForce, axleCorneringStiffness):
    # [Pacejka] p. 336 (below Eq. 7.25): slip angle where the peak force occurs, a_sl ~ 3 * Dy / C_Fa
    return 3.0 * axlePeakForce / max(axleCorneringStiffness, 1.0)

def calcLowSpeedDampingCoefficient(velocityX):
    """
    [Pacejka] Eq. 8.128, p. 397: the artificial low-speed damping fades to zero at |Vx| = V_low.
        k_Vlow = 0.5 * k_Vlow0 * (1 + cos(pi * |Vx| / V_low))   if |Vx| <= V_low
        k_Vlow = 0                                               if |Vx| >  V_low
    Returns N*s/m (per axle).
    """
    lowSpeed = Parameters["lowSpeedThreshold"]
    if abs(velocityX) > lowSpeed:
        return 0.0
    return 0.5 * Parameters["lowSpeedDampingCoeff"] * (1.0 + np.cos(np.pi * abs(velocityX) / lowSpeed))

def calcDampedTransientSlipAngle(transientSlipAngle, lateralSlipVelocity, velocityX, axleCorneringStiffness):
    """
    Besselink low-speed damping, [Pacejka] Eq. 7.26, p. 336 (written there for k'; "a similar equation may
    be employed for the lateral transient slip"), applied as in Eq. 8.127, p. 397:
        a'_damped = a' - (k_Vlow / C_Fa) * Vsy          (Pacejka sign)
        a'_damped = a' + (k_Vlow / C_Fa) * Vsy          (this file's sign)
    In the linear range Fy = -C*a'_damped = -C*a' - k_Vlow*Vsy: a viscous force opposing contact-patch sliding.

    Without it the tyre is a pure spring at standstill and the car oscillates "practically undamped" (p. 336).
    Only used to compute the force; the a' state itself is still integrated with Eq. 7.18.
    """
    dampingCoeff = calcLowSpeedDampingCoefficient(velocityX)

    return transientSlipAngle + dampingCoeff / max(axleCorneringStiffness, 1.0) * lateralSlipVelocity

def calcAxleLateralForce(transientSlipAngle, tireNormalForces, tirePressures, tireFit:dict, slipSpeed=0.0):
    """
    Lateral force of one axle (two tyres, each on its own load, so lateral load transfer reduces axle grip
    through the tyres' load sensitivity). Magic Formula driven by the transient slip angle, [Pacejka] Eq. 7.22, p. 335.
    Sign: this file's a' is -a'_Pacejka, so Fy = F_MF(-a').

    :param tireNormalForces, tirePressures: [left, right] of the axle
    """
    return (calcMagicFormulaLateralForce(-transientSlipAngle, tireNormalForces[0], tirePressures[0], tireFit, slipSpeed)
            + calcMagicFormulaLateralForce(-transientSlipAngle, tireNormalForces[1], tirePressures[1], tireFit, slipSpeed))

def calcFrontLateralSlipVelocity(lateralVelocity, yawRate, steeringAngle, velocityX):
    # Lateral slip velocity of the front contact patch in the wheel frame: Vsy1 ~ Vy + a*r - Vx*delta
    # (small steer angle). [Pacejka] Eq. 1.36, p. 20: alpha1 = delta - (v + a*r)/u, so Vsy1 = -u*alpha1 (Pacejka sign).
    frontAxleDistanceToCOM = Parameters["a"]

    return lateralVelocity + yawRate * frontAxleDistanceToCOM - velocityX * steeringAngle

def calcRearLateralSlipVelocity(lateralVelocity, yawRate):
    # Lateral slip velocity of the rear contact patch: Vsy2 = Vy - b*r  ([Pacejka] Eq. 1.36, p. 20: alpha2 = -(v - b*r)/u)
    rearAxleDistanceToCOM = Parameters["wheelBase"] - Parameters["a"]

    return lateralVelocity - yawRate * rearAxleDistanceToCOM

def getRelaxationLength():
    # [Pacejka] p. 22: "At nominal vertical load, the relaxation length is of the order of magnitude of the wheel radius."
    # More precisely sigma_a = C_Fa / C_Fy ([Pacejka] Eq. 7.6, p. 331), needs the lateral carcass stiffness C_Fy.
    return Parameters["wheelRadius"]

def isSlipAngleLimited(transientSlipAngle, slipAngleRate, velocityX, peakSlipAngle):
    # [Pacejka] Eq. 7.25, p. 335: if |a'| > a_sl and |Vx| < V_low and the deflection is still growing,
    # hold it (dv/dt = 0). Otherwise Eq. 7.7 / 7.18 applies.
    return (abs(velocityX) < Parameters["lowSpeedThreshold"]
            and abs(transientSlipAngle) > peakSlipAngle
            and slipAngleRate * transientSlipAngle > 0)

def calcTransientSlipAngleRate(transientSlipAngle, lateralSlipVelocity, velocityX, peakSlipAngle):
    # [Pacejka] Eq. 7.18, p. 333: sigma * da'/dt + |Vx| * a' = Vsy   (this file's sign; Pacejka has -Vsy)
    # Vx never appears in a denominator, so it is valid through standstill (p. 334).
    slipAngleRate = (lateralSlipVelocity - abs(velocityX) * transientSlipAngle) / getRelaxationLength()

    if isSlipAngleLimited(transientSlipAngle, slipAngleRate, velocityX, peakSlipAngle):
        slipAngleRate = 0.0
    return slipAngleRate

def stepTransientSlipAngle(transientSlipAngle, lateralSlipVelocity, velocityX, peakSlipAngle, timeStep):
    """
    Advance the transient slip angle a' one step with IMPLICIT (backward) Euler on [Pacejka] Eq. 7.18, p. 333:
        sigma * da'/dt + |Vx| * a' = Vsy
    =>  a'_new = (a'_old + dt/sigma * Vsy) / (1 + |Vx| * dt/sigma)

    Unconditionally stable (explicit Euler is only stable for |Vx| < 2*sigma/dt ~ 43 m/s at 100 Hz)
    and still has no division by Vx. Then the [Pacejka] Eq. 7.25 (p. 335) low-speed limit holds a' if it is
    already past a_sl and would keep growing.
    """
    k = timeStep / getRelaxationLength()

    newSlipAngle = (transientSlipAngle + k * lateralSlipVelocity) / (1.0 + abs(velocityX) * k)

    if isSlipAngleLimited(transientSlipAngle, newSlipAngle - transientSlipAngle, velocityX, peakSlipAngle):
        newSlipAngle = transientSlipAngle
    return newSlipAngle

def getStaticNormalForces() -> np.ndarray:
    # [FL, FR, RL, RR] in Newtons from mass and CG position (b/L on the front axle, a/L on the rear)
    weight = Parameters["Mass"] * 9.81
    wheelBase = Parameters["wheelBase"]
    a = Parameters["a"]
    b = wheelBase - a
    front = weight * b / wheelBase / 2.0
    rear = weight * a / wheelBase / 2.0
    return np.array([front, front, rear, rear], dtype=float)

def getDefaultTirePressures() -> np.ndarray:
    # Pa. Must be in the same units as Parameters["p_0"] ([Pacejka] Eq. 4.E2b, p. 177).
    return np.full(4, float(Parameters["tirePressure"]))

def calcLateralAcceleration(frontLateralForce, rearLateralForce, steeringAngle):
    # Body-frame lateral acceleration (what a car-mounted IMU reads): a_y = (FyF*cos(delta) + FyR) / m
    # Right-hand side of [Pacejka] Eq. 1.42a, p. 22, with the front force rotated out of the steered wheel frame.
    return (frontLateralForce * np.cos(steeringAngle) + rearLateralForce) / Parameters["Mass"]

def calcLateralVelocityRate(lateralAcceleration, velocityX, yawRate):
    # [Pacejka] Eq. 1.42a, p. 22: m(dv/dt + u*r) = Fy1 + Fy2  ->  dVy/dt = a_y - Vx*r   [m/s^2, NOT angular]
    return lateralAcceleration - velocityX * yawRate

def calcLongitudinalVelocityRate(longitudinalAcceleration, lateralVelocity, yawRate):
    # [Pacejka] Eq. 1.34a, p. 20: m(du/dt - r*v) = Fx1 - Fy1*delta + Fx2   (a_x below already includes -Fy1*sin(delta)/m)
    # ->  dVx/dt = a_x + Vy*r
    return longitudinalAcceleration + lateralVelocity * yawRate

def calcYawAcceleration(frontLateralForce, rearLateralForce, steeringAngle):
    # [Pacejka] Eq. 1.42b, p. 22: I*dr/dt = a*Fy1 - b*Fy2   (front force rotated by cos(delta))
    a = Parameters["a"]
    b = Parameters["wheelBase"] - a
    return (a * frontLateralForce * np.cos(steeringAngle) - b * rearLateralForce) / Parameters["polarMoment"]

def calcBodySlip(lateralVelocity, velocityX):
    # beta = atan2(Vy, Vx). Output only -- never integrate it. Defined (=0) at standstill.
    # Large/meaningless near Vx = 0; nothing downstream consumes it.
    return np.arctan2(lateralVelocity, velocityX)

def calcBodySlipVelocity(lateralVelocity, lateralVelocityRate, velocityX, longitudinalVelocityRate=0.0):
    # d/dt atan2(Vy, Vx) = (Vx*dVy/dt - Vy*dVx/dt) / (Vx^2 + Vy^2)   [rad/s]. Logging / IMU comparison only.
    # Small-angle, constant-speed limit is [Balkwill] Eq. 5.6 (pp. 129-133), dbeta/dt = a_y/V - r.
    speedSquared = velocityX ** 2 + lateralVelocity ** 2
    if speedSquared < 1e-6:
        return 0.0
    return (velocityX * lateralVelocityRate - lateralVelocity * longitudinalVelocityRate) / speedSquared

class LateralDynamicsRates(NamedTuple):
    lateralVelocityRate: float   # dVy/dt, m/s^2
    yawAcceleration: float       # dr/dt, rad/s^2
    frontSlipAngleRate: float    # da'F/dt, rad/s (explicit rate, for logging)
    rearSlipAngleRate: float     # da'R/dt, rad/s
    frontLateralForce: float     # N, front axle, in the tyre frame
    rearLateralForce: float      # N, rear axle
    lateralAcceleration: float   # body-frame a_y, m/s^2
    frontPeakSlipAngle: float    # rad
    rearPeakSlipAngle: float     # rad

def calcLateralDynamicsRates(velocityX, lateralVelocity, yawRate, steeringAngle, frontSlipAngle, rearSlipAngle,
                             tireNormalForces: None | np.ndarray = None, tirePressures: None | np.ndarray = None) -> LateralDynamicsRates:
    """
    Single-track (bicycle) model, [Pacejka] Eq. 1.42a/b (p. 22), with relaxation-length tyres,
    [Pacejka] Eq. 7.18 (p. 333), the low-speed limit of Eq. 7.25 (p. 335) and the low-speed damping of
    Eq. 7.26 / 8.127-8.128 (pp. 336, 397). Replaces the beta-based form of [Balkwill] Eqs. 5.7-5.8, which divide by V.

    Everything is in the VEHICLE (body) frame: x forward, y left, z up, yaw rate + CCW, steer + left (ISO 8855).

    :param velocityX: longitudinal velocity Vx (body frame), m/s
    :param lateralVelocity: lateral velocity Vy (body frame), m/s
    :param steeringAngle: ROAD WHEEL angle, rad (steering wheel angle / steeringRatio)
    :param frontSlipAngle, rearSlipAngle: transient slip angle states a' (rad)
    :param tireNormalForces: [FL, FR, RL, RR] N
    :param tirePressures: [FL, FR, RL, RR] Pa
    """
    if tireNormalForces is None:
        tireNormalForces = getStaticNormalForces()
    if tirePressures is None:
        tirePressures = getDefaultTirePressures()

    frontStiffness, rearStiffness = getAxleCorneringStiffness(tireNormalForces, tirePressures)

    frontSlipVelocity = calcFrontLateralSlipVelocity(lateralVelocity, yawRate, steeringAngle, velocityX)
    rearSlipVelocity = calcRearLateralSlipVelocity(lateralVelocity, yawRate)
    # Slip speed for Eq. 4.E7. Only lateral slip is modelled so far (no slip ratio), so Vs = |Vsy|.
    frontSlipSpeed = abs(frontSlipVelocity)
    rearSlipSpeed = abs(rearSlipVelocity)

    frontPeak = getPeakSlipAngle(getAxlePeakLateralForce(tireNormalForces[0:2], tirePressures[0:2], TireFront, frontSlipSpeed), frontStiffness)
    rearPeak = getPeakSlipAngle(getAxlePeakLateralForce(tireNormalForces[2:4], tirePressures[2:4], TireRear, rearSlipSpeed), rearStiffness)

    frontDampedSlipAngle = calcDampedTransientSlipAngle(frontSlipAngle, frontSlipVelocity, velocityX, frontStiffness)
    rearDampedSlipAngle = calcDampedTransientSlipAngle(rearSlipAngle, rearSlipVelocity, velocityX, rearStiffness)
    frontForce = calcAxleLateralForce(frontDampedSlipAngle, tireNormalForces[0:2], tirePressures[0:2], TireFront, frontSlipSpeed)
    rearForce = calcAxleLateralForce(rearDampedSlipAngle, tireNormalForces[2:4], tirePressures[2:4], TireRear, rearSlipSpeed)

    lateralAcceleration = calcLateralAcceleration(frontForce, rearForce, steeringAngle)
    lateralVelocityRate = calcLateralVelocityRate(lateralAcceleration, velocityX, yawRate)
    yawAcceleration = calcYawAcceleration(frontForce, rearForce, steeringAngle)

    frontRate = calcTransientSlipAngleRate(frontSlipAngle, frontSlipVelocity, velocityX, frontPeak)
    rearRate = calcTransientSlipAngleRate(rearSlipAngle, rearSlipVelocity, velocityX, rearPeak)

    return LateralDynamicsRates(lateralVelocityRate, yawAcceleration, frontRate, rearRate,
                                float(frontForce), float(rearForce), lateralAcceleration, frontPeak, rearPeak)


def calcSlipAngle(steerAngle, speed, yawRate, bodySlip) -> tuple[float,float]:
    """
    DEPRECATED: steady-state slip angles (divides by speed). The engine now carries transient
    slip angles as states (see stepTransientSlipAngle). Kept only because traction.py imports it.

    Calculate Slip Angle Based on yawRate, Velocity, and Steering Angle.
    
    :param yawRate: Angular Rate of the car
    :param velocity: Ground Speed of the car (norm x,y)
    :param steerAngle: Steering Angle, radians, positive CCW
    :param bodySlip: Body slip angle of the car
    :return: (frontSlipAngle, rearSlipAngle)
    """

    frontSlipAngle = calcVirtualSlipAngle() + bodySlip + (Parameters["wheelBase"]*Parameters["frontWeightDist"]/100 * yawRate)/speed - steerAngle
    rearSlipAngle = bodySlip - (Parameters["wheelBase"]*(100-Parameters["frontWeightDist"])/100 * yawRate)/speed

    return (frontSlipAngle, rearSlipAngle)

def calcVirtualSlipAngle():
    # This model is based on Chapter 1 of Pacejka's 2012 book.
    # We treat all variables here as static to calculate virtual slip angle
    # This is entirely untrue. Every single variable is something to calculate every step.
    # But for now, we will guess
    # TODO: Improve every variable listed

    return 0# Parameters["frontToe"]

    # frontCorneringStiffnessDeg = -140 # Guess because this system isn't valid at high slip angle and when corrnering stiffness is dynamic
    # CF = frontCorneringStiffnessDeg * 180 / np.pi
    # Fy = 0

    # l = Parameters["wheelBase"]
    # m = Parameters["Mass"]
    # epsilon_i = Parameters["rollSteerCoefficient"]
    # tau_i = Parameters["rollCamberSteerCoefficient"]
    # hPrime = Parameters["CoG-distanceToRollAxis"]
    # e_i = Parameters["casterLength"]
    # t_i = = 0 # Pneumatic trail length. Hard Tire Modeling problem
    # c_phi1 = 0
    # c_phi2 = 0
    # c_psii = 0
    # c_sfi = 0
    # a_i = 0
    # sigma_i = 0 # Term 4 only
    # zeta_alphai = 0 # Term 4 only
    # psi_io = 0 # Term 4 only
    # zeta_gammai = 0 # Term 4 only
    # gamma_io = 0 # Term 4 only
    #
    # # CF here is used wrong. Hard tire modeling problem.
    # term1Num = l * (epsilon_i * CF + tau_i * CF) * hPrime
    # Term1Denom = (l - a_i) * (c_phi1 + cphi2 - m * 9.81 * hPrime)
    #
    # term2Num = CF * (e_i + t_i)
    # term2Denom = c_psii
    #
    # term3 = -1 * CF * c_sfi
    #
    # # WE NEGLECT TERM 4 BECAUSE WE ASSUME ZETA TO BE 0 WHICH IS WRONG. HARD TIRE MODELING PROBLEM
    #
    # return (Fy / CF) * (1 + term1Num/Term1Denom + term2Num/term2Denom + term3)
'''
def calcYawRate(currYawRate, speed, stepSteerInput, timeSinceLastSteer, frontCorneringStiffnessDeg_, rearCorneringStiffnessDeg_):
    # This model is based on Performance Vehicle Dynamics
    # It is a pretty meh model which uses euler's method to approximate transient behavior
    # Ideally we would use something a bit better like rk4 but i couldn't get that to work
    # This model is only valid for small slip angles, even though we use them for large slip angles.
    # This entire thing needs to get rewritten in the future
    # Potential future models include ones presented in the VD compendium book or Road Vehicle VD book
    # TODO: Implement new model

    frontCorneringStiffnessDeg = -140 # Guess because this system isn't valid at high slip angle and when corrnering stiffness is dynamic
    rearCorneringStiffnessDeg = -140 # Guess because this system isn't valid at high slip angle and when corrnering stiffness is dynamic
    #speed = 30 # Arbitrary because speed maybe doesn't work
    if speed == 0 or stepSteerInput == 0:
        return 0

    CF = frontCorneringStiffnessDeg * 180 / np.pi
    CR = rearCorneringStiffnessDeg * 180 / np.pi
    a = Parameters['a']
    b = Parameters["wheelBase"] - a
    m = Parameters["Mass"]
    I = Parameters["polarMoment"]
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

    #print(c, Cc)

    # if zeta < 1: # Underdamped
    #     omega_d = np.sqrt(1 - zeta**2) * omega_n
    #     A = -r_inf
    #     B = (r_dot_0 - zeta * omega_n * r_inf) / omega_d
    #     exp_term = np.exp(-zeta * omega_n * timeSinceLastSteer)
    #     cos_term = A * np.cos(omega_d * timeSinceLastSteer)
    #     sin_term = B * np.sin(omega_d * timeSinceLastSteer)
    #     normalizedR = exp_term * (cos_term + sin_term) + r_inf
    # elif zeta > 1: # Overdamped
    #     f = (-zeta - np.sqrt(zeta**2 - 1)) * omega_n
    #     g = (-zeta + np.sqrt(zeta**2 - 1)) * omega_n
    #     A = (r_dot_0 + r_inf * f) / (g - f)
    #     B = -(A + r_inf)
    #     r = A * np.exp(g * timeSinceLastSteer) + B * np.exp(f * timeSinceLastSteer) + r_inf
    #     normalizedR = r / r_inf
    # else: # Critically
    #     term1 = (-1* (CF * stepSteerInput * a)/(I * r_inf) - omega_n)
    #     normalizedR = (-1 + term1 * timeSinceLastSteer) * np.e **(-1 * omega_n * timeSinceLastSteer) + 1

    #print("STEERING INPUT", normalizedR, r_inf, zeta, Cc, omega_n, r_dot_0, r_inf, C2, k, c, YR_v, NR_v, N_delta, N_beta, Y_delta, Y_beta)
    # return normalizedR * r_inf
    return r_inf

# parameters = {"Mass": 300, "polarMoment": 658.088580080000, "a": 0.853506, "wheelBase": 1.65471}
# for i in np.arange(0, 1, 0.02):
#     res = calculateYawRate(0, 35.76, 0.4, i, -1086.083, -890.0656, parameters)
#     print(i, res)

'''