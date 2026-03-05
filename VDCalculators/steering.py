# Steering model
import numpy as np

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
    # This model is based on Chapter 1 of Pacejka's 2012 book.
    # We treat all variables here as static to calculate virtual slip angle
    # This is entirely untrue. Every single variable is something to calculate every step.
    # But for now, we will guess
    # TODO: Improve every variable listed

    return 0# parameters["frontToe"]

    frontCorneringStiffnessDeg = -140 # Guess because this system isn't valid at high slip angle and when corrnering stiffness is dynamic
    CF = frontCorneringStiffnessDeg * 180 / np.pi
    Fy = 0

    # l = parameters["wheelBase"]
    # m = parameters["Mass"]
    # epsilon_i = parameters["rollSteerCoefficient"]
    # tau_i = parameters["rollCamberSteerCoefficient"]
    # hPrime = parameters["CoG-distanceToRollAxis"]
    # e_i = parameters["casterLength"]
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

def calculateYawRate(currYawRate, speed, stepSteerInput, timeSinceLastSteer, frontCorneringStiffnessDeg_, rearCorneringStiffnessDeg_, parameters):
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

    #print(c, Cc)

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

    #print("STEERING INPUT", normalizedR, r_inf, zeta, Cc, omega_n, r_dot_0, r_inf, C2, k, c, YR_v, NR_v, N_delta, N_beta, Y_delta, Y_beta)
    return normalizedR * r_inf

# parameters = {"Mass": 300, "polarMoment": 658.088580080000, "a": 0.853506, "wheelBase": 1.65471}
# for i in np.arange(0, 1, 0.02):
#     res = calculateYawRate(0, 35.76, 0.4, i, -1086.083, -890.0656, parameters)
#     print(i, res)
