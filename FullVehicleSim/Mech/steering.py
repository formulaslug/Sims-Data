import numpy as np

from ..paramLoader import Magic, Parameters


def normalizePressureLong(tirePressure):
    # Only long because lat doesn't use it
    lambda_pressurescalarlong = Magic["lambda_pressurescalarlong"]

    return (tirePressure - lambda_pressurescalarlong * tirePressure) / (lambda_pressurescalarlong * tirePressure)

def normalizeLoadLong(normalForce):
    lambda_loadscalarlong = Magic["lambda_loadscalarlong"]

    return (normalForce - lambda_loadscalarlong * normalForce) / (lambda_loadscalarlong * normalForce)

def calcLongCompositeLongFrictionScalingFactor(velocityX):
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
    scaling_factor = calcLongCompositeLongFrictionScalingFactor(velocityX)
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

'''
def calcSlipAngle(worldArray:np.ndarray, step:int) -> tuple[float,float]:
    """
    Calculate Slip Angle Based on yawRate, Velocity, and Steering Angle.
    
    Currently returns 0,0 if yawRate or speed is 0 to avoid division by zero.
    This is incorrect and should be replaced with a better model in the future.
    
    :param yawRate: Description
    :param velocity: Description
    :param steerAngle: Description
    :return: (frontSlipAngle, rearSlipAngle)
    """
    steerAngle = worldArray[step, varSteerAngle]
    speed = worldArray[step-1, varSpeed]
    yawRate = worldArray[step-1, varYawRate]
    if yawRate == 0 or speed == 0: # WRONG. RELAXATION LENGTH. PROJECT
        return (0, 0)
    else:
        bodySlip = np.arctan(worldArray[step-1, varVelY]/worldArray[step-1, varVelX])

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

    frontCorneringStiffnessDeg = -140 # Guess because this system isn't valid at high slip angle and when corrnering stiffness is dynamic
    CF = frontCorneringStiffnessDeg * 180 / np.pi
    Fy = 0

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