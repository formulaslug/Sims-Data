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


def calculateAckermannOther(steerAngle):
    #steering wheel angle --> steering rack psition --> wheel steer angle (how static ackermann affects wheel angle function)
    # equations taken from "rack and pinion" section of: https://www.mathworks.com/help/vdynblks/ref/kinematicsteering.html
    import scipy
    import numpy
    import matplotlib.pyplot as plt
    #global variables
    #-----------------------------
    # THIS SCRIPT USES MOSTLY FS-3 VALUES. FS-3 values denoted by [3], any theoretical or FS-4 values denoted by [4]
    #-----------------------------
    tw = 1083.3862 #mm (simplified track width from steering axis to steering axis [steering axis is also simplified to be A-arm knuckle to A-arm knuckle])
    rackRatio = 82.55/248 #[4] mm rack displacement/deg pinion rotation
    wheelInput = steerAngle * 180/3.141592 #in degrees of steering wheel movement (CW + CCW -)
    rackShift = 0.0 # mm of movement of the rack from left to right (left is - right is +)
    l_rack = 292.1 #[4] mm (width of steering rack casing)
    l_rod = 378.9426 #[3] mm (length of tie rod as left in FS-3 master CAD)
    d = 109.7788 #[3] mm (plan view distance between front axis and rack. negative because we have a front steer setup)
    l_arm = 71.628 #[3] mm (length of "steer arm", which is the distance from the center of the upright toe rod pickup to the KPA)

    def rackMovement(): #returns the amount of L-R displacement (in mm) of the steering rack, with the right direction as "positive"
        rackShift: float = rackRatio*wheelInput
        return rackShift
    def calculateAckermann(): #calculates the steer angles of both wheels
        l1Left = (0.5*(tw-l_rack)) - rackMovement() #l1 is the instantaneous parallel distance from the rack knuckle to steering axis (KPA). 
        l1Right = (0.5*(tw-l_rack)) + rackMovement()
        l_nought = (0.5*(tw-l_rack))
        beta_nought = betaTrigSolver(l_nought) #used to find the initial "beta" geometry to determine the real steer angle at the wheels

        beta_L = betaTrigSolver(l1Left) - beta_nought #additionally, because there is a static "beta" (simply just arm geometry), we must find the difference to find the actual wheel angles
        beta_R = betaTrigSolver(l1Right) - beta_nought
    
        return beta_L, beta_R
        #return beta_nought, betaTrigSolver(l1Left), betaTrigSolver(l1Right)
    
    def betaTrigSolver(l1): #a separate function to solve the big bad trig equation
        l2 = numpy.sqrt((l1**2) + (d**2)) #l2 is the instantaneous direct distance from rack knuckle to steering axis (KPA)
        atan = numpy.arctan(d/l1) #first term of the "beta" equation

        num = (l_arm**2) + (l2**2) - (l_rod**2) #just simplifying the calculation of the second term 
        denom = 2*l_arm*l2
        frac = num/denom
        acos = numpy.arccos(frac)
        beta = (numpy.pi/2) - atan - acos
        return beta
        #return frac
    inTire, outTire = calculateAckermann()
    return (inTire, outTire)

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
def solveUSG(minSteer, maxSteer, minVelocity, maxVelocity): #usg-specific solver so we dont lose the old one in case this is cooked. mostly copy+paste from old solver
    carMass = [277 / 4, 277 / 4 , 277 / 4 , 277 / 4 ]
    USG_vals = []
    radius_vals = []

    for steer in np.arange(minSteer, maxSteer, 0.1):
        curveUSG = []
        curveRadius = []

        for velocity in np.arange(minVelocity, maxVelocity, 1):
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
steer_vals = np.arange(minSteer, maxSteer, 0.1)
velocity_vals = np.arange(minVelocity, maxVelocity, 1)

S, V, Z = [], [], []

USG_vals, R_vals = solveUSG(
    minSteer, maxSteer, minVelocity, maxVelocity
)

for i, steer in enumerate(steer_vals):
    for j, velocity in enumerate(velocity_vals):
        usg = USG_vals[i][j]
        if np.isfinite(usg):
            S.append(steer)
            V.append(velocity)
            Z.append(usg)

S = np.array(S)
V = np.array(V)
Z = np.array(Z)

# # --- run solver (YR-angle-v graph) ---
# yaw_rates = solver(minSteer, maxSteer, minVelocity, maxVelocity)

# # --- build coordinate lists for trisurf ---
# steer_vals = np.arange(minSteer, maxSteer, 0.1)
# velocity_vals = np.arange(minVelocity, maxVelocity, 1)

# S = []
# V = []
# Z = []

# for i, steer in enumerate(steer_vals):
#     for j, velocity in enumerate(velocity_vals):
#         S.append(steer)
#         V.append(velocity)
#         Z.append(yaw_rates[i][j])

# S = np.array(S)
# V = np.array(V)
# Z = np.array(Z)

# --- plot (also just ctrl c+v) ---
#also subplot for cornering stiffnesses
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

