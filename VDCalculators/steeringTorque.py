import os
import sys
import math
import json
import numpy
import traction
import tireLoad
import steering
import tireState
import math
from traction import getCorneringStiffness
from tireLoad import getLatLoadTransfer
from tireState import Tire
from steering import calculateSlipAngle #might want to take this from ackermann model
import json
magic:dict
parameters:dict
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #gpt-generated way to find params
params_path = os.path.join(BASE_DIR, '..', 'FullVehicleSim', 'params.json')

with open(params_path, 'r') as file:
    params = json.load(file)
    Magic = params["Magic"]
    Parameters = params["Parameters"]
    del params

# global variables
track = 1.234 #m
hcg = 0.3048 #m, from ground
