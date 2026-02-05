import os
import sys
import math
import json
import numpy
sys.path.append("../FullVehicleSim/Mech")
import traction
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
    tw = 1.0833862 #fs4/3m (simplified track width from steering axis to steering axis [steering axis is also simplified to be A-arm knuckle to A-arm knuckle])
    rackRatio = 82.55/numpy.deg2rad(248) #fs4/3 mm rack displacement/deg pinion rotation
    l_rod = 383.211 #fs4 mm (length of "tie rod")
    d = 32.905 #fs4 mm (sta to rack, longitudinal)
    l_arm = 75.946 #fs4 mm (length of "steer arm", which is the distance from the center of the upright toe rod pickup to the KPA)
    LWB = 1.524 #fs4 m lwb
    phiStatic = numpy.deg2rad(4.531) #fs4 degrees
    d_lat = 387.194 #fs4 mm (sta to rack, lateral)
def calculateAckermannOther(steerAngle):
    import scipy
    import numpy
    import matplotlib.pyplot as plt


    def findAckermannFactor(leftAngle, rightAngle): #INPUT MUST BE IN RADIANS, this is from equation 1B
        angleIn = 0
        if (leftAngle < 0 or rightAngle > 0):
            angleIn = rightAngle
            ackermannIn = (1/ numpy.tan(angleIn)) + tw/LWB
            outIdeal = numpy.atan(1/ackermannIn)
        if (leftAngle > 0 or rightAngle < 0):
            angleIn = leftAngle
            ackermannIn = (1/ numpy.tan(angleIn)) + tw/LWB
            outIdeal = numpy.atan(1/ackermannIn)
        if (leftAngle == 0 or rightAngle == 0):
            angleIn = 0
            outIdeal = 0
            return "No steer input: Ackermann is undefined"
        numerator = 1 / (numpy.tan(leftAngle) - numpy.tan(rightAngle))
        denominator = 1 / (numpy.tan(outIdeal) - numpy.tan(angleIn))
        ackFactor = 100*(numpy.abs(numerator/denominator))
        return ackFactor #OUTPUT IS IN PERCENT.
    def findStaticSteeringGeo(): #this is from equation 1A
        value = (d - l_arm*numpy.cos(phiStatic))**2 + (d_lat - l_arm*numpy.sin(phiStatic))**2
        toeRodLength = numpy.positive(numpy.sqrt(value))
        print(f"Toe Rod Length: {l_rod}, Derived TRL: {toeRodLength}")
        return 0
    def calculateSteerAngles(wheelInput): #INPUT MUST BE IN RAD, this is from equation 2
        leftSteerAngle = 0
        rightSteerAngle = 0

        LS_repeatTerm = d_lat + (rackRatio*wheelInput)
        LS_firstTerm = numpy.atan(LS_repeatTerm/d)
        LS_secondTermNumerator = (d**2) + (LS_repeatTerm**2) + (l_arm**2) - (l_rod**2)
        LS_sqrt = (numpy.sqrt((d**2) + (LS_repeatTerm**2)))
        LS_secondTermDenominator = (2*l_arm)*LS_sqrt
        LS_secondTerm = numpy.acos(LS_secondTermNumerator/LS_secondTermDenominator)
        leftSteerAngle = LS_firstTerm - LS_secondTerm - phiStatic

        RS_repeatTerm = d_lat - (rackRatio*wheelInput)
        RS_firstTerm = numpy.atan(RS_repeatTerm/d)
        RS_secondTermNumerator = (d**2) + (RS_repeatTerm**2) + (l_arm**2) - (l_rod**2)
        RS_sqrt = (numpy.sqrt((d**2) + (RS_repeatTerm**2)))
        RS_secondTermDenominator = (2*l_arm)*RS_sqrt
        RS_secondTerm = numpy.acos(RS_secondTermNumerator/RS_secondTermDenominator)
        rightSteerAngle = phiStatic - RS_firstTerm + RS_secondTerm

        return leftSteerAngle, rightSteerAngle #OUTPUT WILL ALSO BE IN RAD
    inTire, outTire = calculateSteerAngles(steerAngle)
    return (inTire, outTire)
def updateSteerGeo(rackStep): #rackStep should be +/- ~5mm, this is from equation 1A
    global l_rod
    storeOldRod = l_rod
    global d 
    d += rackStep
    value = (d - l_arm*numpy.cos(phiStatic))**2 + (d_lat - l_arm*numpy.sin(phiStatic))**2
    l_rod = numpy.positive(numpy.sqrt(value))
    print(f"Old Toe Rod Length: {storeOldRod}, Updated TRL: {l_rod}")
    return 0
def calculateSlipAngle(steerAngle, velocity):
    Vx = velocity / math.cos(steerAngle)
    Vy = velocity * math.tan(steerAngle)
    return math.atan(Vy/Vx)
def calculateYawRate(steerAngle, frontCorneringStiffnessDeg, rearCorneringStiffnessDeg, speed, stepSteerInput):
    CF = frontCorneringStiffnessDeg * 180 / np.pi
    CR = rearCorneringStiffnessDeg * 180 / np.pi
    a = 0.853506
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
    return r_inf
def calculateUSG(steerAngle, yawRate, velocity):
    L_wb = 1.589989 #wheelbase length, in meters
    if (yawRate == 0): #just return 0 and break so as not to throw invalid division error
        return 0
    R_p = velocity/yawRate #cornering radius, in meter
    ay = velocity*yawRate #lateral acceleration, m/s
    rhoPerfect = L_wb/R_p #chalmer's formula for "perfect steering angle"
    usg = (steerAngle - rhoPerfect)/ay #chalmer's formula
    # ^^^ on the condition that your steer angle is MORE than rhoPerfect (turning more than you need to), K_u is positive
    return usg

def solver(minSteer, maxSteer, minVelocity, maxVelocity): #old solver for yaw rate
    carMass = [277 / 4, 277 / 4 , 277 / 4 , 277 / 4 ]
    res = []
    for steer in np.arange(minSteer, maxSteer, 0.1):
        currLine = []
        for velocity in np.arange(minVelocity, maxVelocity, 1):
            inTire, outTire = calculateAckermannOther(steer)
            inCorneringAngleSlip = calculateSlipAngle(inTire, velocity)
            outCorneringAngleSlip = calculateSlipAngle(outTire, velocity)
            inCorneringStiff = traction.getCorneringStiffness(carMass, (inCorneringAngleSlip, 0), 0.15, velocity, 80, 40, Parameters, Magic)[0]
            outCorneringStiff = traction.getCorneringStiffness(carMass, (outCorneringAngleSlip, 0),  0.15, velocity, 80, 40, Parameters, Magic)[0]
            netFrontCorneringStiffness = (inCorneringStiff + outCorneringStiff)/2
            netRearCornerningStiffness = -70 # lol who knows bruh
            yawRate = calculateYawRate(steer, netFrontCorneringStiffness, netRearCornerningStiffness, velocity, steer) * -1
            currLine.append(yawRate)
        res.append(currLine)
    return res
def solveUSG(minSteer, maxSteer, minVelocity, maxVelocity, steerStep=0.1, velocityStep=1): #usg-specific solver so we dont lose the old one in case this is cooked. mostly copy+paste from old solver
    carMass = [277 / 4, 277 / 4 , 277 / 4 , 277 / 4 ]
    USG_vals = []
    radius_vals = []

    for steer in np.arange(minSteer, maxSteer, steerStep):
        curveUSG = []
        curveRadius = []

        for velocity in np.arange(minVelocity, maxVelocity, velocityStep):
            inTire, outTire = calculateAckermannOther(steer)
            inSlip = calculateSlipAngle(inTire, velocity)
            outSlip = calculateSlipAngle(outTire, velocity)
            inCorneringStiff = traction.getCorneringStiffness(carMass, (inSlip, 0), 0.15, velocity, 80, 40, Parameters, Magic)[0]
            outCorneringStiff = traction.getCorneringStiffness(carMass, (outSlip, 0),  0.15, velocity, 80, 40, Parameters, Magic)[0]
            netCF = (inCorneringStiff + outCorneringStiff)/2
            netCR = -70 # i still dont know bruh xd
            yawRate = calculateYawRate(steer, netCF, netCR, velocity, steer) * -1
            if (yawRate == 0): #avoid invalid division error
                radius = 0
            else:
                radius = velocity/yawRate
            usg = calculateUSG(steer, yawRate, velocity)
            curveUSG.append(usg)
            curveRadius.append(radius)
        USG_vals.append(curveUSG)
        radius_vals.append(curveRadius)
    return USG_vals, radius_vals

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- solver inputs ---
minSteer = -1.75 #changed to 0.1 because graph was throwing some crazy values
maxSteer = 1.75
minVelocity = 10
maxVelocity = 30

# -run solver (USG graph)
steer_vals = np.arange(minSteer, maxSteer, 0.01)
velocity_vals = np.arange(minVelocity, maxVelocity, 1)

S, V, Z = [], [], []

USG_vals, R_vals = solveUSG(
    minSteer, maxSteer, minVelocity, maxVelocity
)

for i in range(len(USG_vals)):
    for j in range(len(USG_vals[0])):
        usg = USG_vals[i][j]
        if np.isfinite(usg):
            S.append(steer_vals[i])
            V.append(velocity_vals[j])
            Z.append(usg)

S = np.array(S)
V = np.array(V)
Z = np.array(Z)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_trisurf(
    V,              # x-axis
    S,              # y-axis
    Z,              # z-axis
    cmap='viridis',
    linewidth=0.2,
    antialiased=True
)

ax.set_xlabel("Velocity, m/s") 
ax.set_ylabel("Steering Angle, rad") #OLD: steering angle
ax.set_zlabel("Understeer Gradient, rad/g") #OLD: yaw rate
ax.set_title("USG vs. Speed and Steering Input") #OLD: SA&V

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Understeer Gradient") #old title = yaw rate

plt.tight_layout()
plt.show()

