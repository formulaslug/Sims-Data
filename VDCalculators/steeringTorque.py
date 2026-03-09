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
track = 1.0833862 #m
hcg = 0.3048 #m, from ground
larm = 75.946 #steering ARM length (uprights, mm)
lrod = 383.211 #tie rod length (mm)
phiStatic = numpy.deg2rad(4.531) #fs4 degrees (KPI to toe rod pickup)
wheelRadius = 0 #steering wheel radius (m)
rackRatio = 0 #steering rack ratio
d = 32.905 #fs4 mm (sta to rack, longitudinal)
d_lat = 387.194 #fs4 mm (sta to rack, lateral)

def calculateSteerAngles(wheelInput): #INPUT MUST BE IN RAD, this is from equation 2
        # print(f"{wheelInput=}")
        
        leftSteerAngle = 0
        rightSteerAngle = 0
        # print(f"D_rax: {d}, TRL: {l_rod}")
        LS_repeatTerm = d_lat + (rackRatio*wheelInput)
        LS_firstTerm = numpy.atan(LS_repeatTerm/d)
        LS_secondTermNumerator = (d**2) + (LS_repeatTerm**2) + (larm**2) - (lrod**2)
        LS_sqrt = (numpy.sqrt((d**2) + (LS_repeatTerm**2)))
        LS_secondTermDenominator = (2*larm)*LS_sqrt
        LS_secondTerm = numpy.acos(LS_secondTermNumerator/LS_secondTermDenominator)
        leftSteerAngle = LS_firstTerm - LS_secondTerm - phiStatic
        RS_repeatTerm = d_lat - (rackRatio*wheelInput)
        RS_firstTerm = numpy.atan(RS_repeatTerm/d)
        RS_secondTermNumerator = (d**2) + (RS_repeatTerm**2) + (larm**2) - (lrod**2)
        RS_sqrt = (numpy.sqrt((d**2) + (RS_repeatTerm**2)))
        RS_secondTermDenominator = (2*larm)*RS_sqrt
        RS_secondTerm = numpy.acos(RS_secondTermNumerator/RS_secondTermDenominator)
        rightSteerAngle = phiStatic - RS_firstTerm + RS_secondTerm
        
        return leftSteerAngle, rightSteerAngle #OUTPUT WILL ALSO BE IN RAD

def solveSteeringTorque(steerAngle, casterAngle, KPI):
    #asd
    return 0
