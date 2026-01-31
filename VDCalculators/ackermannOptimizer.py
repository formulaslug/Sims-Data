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

def calculateYawRate(steerAngle
