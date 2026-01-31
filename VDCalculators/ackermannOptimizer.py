import os
import sys
import math
import json
sys.path.append("../FullVehicleSim/Mech")
from steering import *

import json
magic:dict
parameters:dict
with open('../FullVehicleSim/params.json', 'r') as file:
    params = json.load(file)
    Magic = params["Magic"]
    Parameters = params["Parameters"]
    del params


def calculateAckermann(steerAngle):
    return (steerAngle, steerAngle)

def calculateSlipAngle(steerAngle, velocity):
    Vx = velocity / math.cos(steerAngle)
    Vy = velocity * math.tan(steerAngle)
    return math.atan(Vy/Vx)

def calculateYawRate(steerAngle, ):
    CF = frontCorneringStiffnessDeg * 180 / np.pi
    CR = rearCorneringStiffnessDeg * 180 / np.pi
    a = parameters['a']0.853506
    b = 1.589 - a
    m = 277
    I = 658.088580080000
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
