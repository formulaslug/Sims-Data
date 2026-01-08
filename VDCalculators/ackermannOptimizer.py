import os
import sys
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
