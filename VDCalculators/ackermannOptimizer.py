import os
import sys
import math
import json
sys.path.append("../FullVehicleSim/Mech")
from steering import *
from traction import *

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

def calculateYawRate(steerAngle, frontCorneringStiffnessDeg, rearCorneringStiffnessDeg):
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

def solver(minSteer, maxSteer, minVelocity, maxVelocity):
    carMass = 277 / 4 
    res = []
    for steer in range (minSteer, maxSteer, 0.1):
        currLine = []
        for velocity in range(minVelocity, maxVelocity, 1):
            inTire, outTire = calculateAckermann(steer)
            inCorneringAngleSlip = calculateSlipAngle(inTire, velocity)
            outCorneringAngleSlip = calculateSlipAngle(outTire, velocity)
            inCorneringStiff = getCorneringStiffness(carMass,
                                                    inCorneringAngleSlip,
                                                    0.15,
                                                    velocity,
                                                    80,
                                                    40,
                                                    parameters,
                                                    magic)
           outCorneringStiff = getCorneringStiffness(carMass,
                                                    outCorneringAngleSlip,
                                                    0.15,
                                                    velocity,
                                                    80,
                                                    40,
                                                    parameters,
                                                    magic)
            netFrontCorneringStiffness = inCorneringStiffness + outCorneringStiffness
            netRearCornerningStiffness = -140 # lol who knows bruh

            yawRate = calculateYawRate(steer, inCornerningStiff, outCorneringStiff)
            currLine.append(yawRate)
        res.append(currLine)
    return res

        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    minSteer = 0
    maxSteer = 1.8
    minVelocity = 1
    maxVelocity = 30 

    yaw_data = solver(minSteer, maxSteer, minVelocity, maxVelocity)
    
    steering_angles = np.arange(minSteer, maxSteer, 0.1)
    velocities = np.arange(minVelocity, maxVelocity)
    S, V = np.meshgrid(steering_angles, velocities)
    YawRate = np.array(yaw_data)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(S, V, YawRate, cmap='viridis')
    ax.set_xlabel('Steering Angle (degrees)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_zlabel('Yaw Rate (rad/s)')
    plt.show()
