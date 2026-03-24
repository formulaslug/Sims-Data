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
import ackermannOptimizer as ack
from traction import getCorneringStiffness
from tireLoad import getLatLoadTransfer
from tireState import Tire
from steering import calculateSlipAngle #might want to take this from ackermann model
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
magic:dict
parameters:dict
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #gpt-generated way to find params
params_path = os.path.join(BASE_DIR, '..', 'FullVehicleSim', 'params.json')

with open(params_path, 'r') as file:
    params = json.load(file)
    Magic = params["Magic"]
    Parameters = params["Parameters"]
    del params
# CONSTS
LEFT = 1
RIGHT = 0
mass = 210.92 + 63.05 # (car weight + daniel weight)
wheelLoad = [mass/4, mass/4, mass/4, mass/4]

# GLOBAL VARIABLES -------------
#upright kinematics
splen = 0.0424183302 #spindle length, m
kpi = numpy.deg2rad(2.197) #kpi, deg

track = 1.0833862 #m
hcg = 0.3048 #m, from ground
larm1 = 56.515 * 0.001 #steering ARM length (uprights, mm)
larm = 56.515
lrod1 = 339.1154 * 0.001 #tie rod length (mm)
lrod = 339.1154
phiStatic = numpy.deg2rad(4.531) #fs4 degrees (KPI to toe rod pickup)
d1 = 68.9356 * 0.001 #fs4 mm (sta to rack, longitudinal)
d = 68.9356
d_lat1 = 319.7098 * 0.001 #fs4 mm (sta to rack, lateral)
d_lat = 319.7098

#ergo values
rackRatio = 82.55/numpy.deg2rad(248) #steering rack ratio
wheelRadius = 0.13335 #steering wheel radius (m)
pinionRadius = 0.020955 #m 

#tire values
tireRadius = 0.2032 #m
slipRatio = 0.15 #arbitrary value?
temp = 80 #idk
pressure = 40 #idk

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
        LS_secondTerm = numpy.acos(LS_secondTermNumerator/LS_secondTermDenominator) #acos error
        # print(f"LS error (acos): num = {LS_secondTermNumerator}, denom = {LS_secondTermDenominator}")
        # print(f"LS nominal value = {LS_secondTermNumerator/LS_secondTermDenominator}")
        leftSteerAngle = LS_firstTerm - LS_secondTerm - phiStatic
        RS_repeatTerm = d_lat - (rackRatio*wheelInput)
        RS_firstTerm = numpy.atan(RS_repeatTerm/d)
        RS_secondTermNumerator = (d**2) + (RS_repeatTerm**2) + (larm**2) - (lrod**2)
        RS_sqrt = (numpy.sqrt((d**2) + (RS_repeatTerm**2)))
        RS_secondTermDenominator = (2*larm)*RS_sqrt
        RS_secondTerm = numpy.acos(RS_secondTermNumerator/RS_secondTermDenominator) #acos error again
        # print(f"RS error (acos): num = {RS_secondTermNumerator}, denom = {RS_secondTermDenominator}")
        # print(f"RS nominal value = {RS_secondTermNumerator/RS_secondTermDenominator}")
        rightSteerAngle = phiStatic - RS_firstTerm + RS_secondTerm
        
        return leftSteerAngle, rightSteerAngle #OUTPUT WILL ALSO BE IN RAD
def solveSteerMoment(F_y, caster, side): # FOR A SINGLE SIDE! output in newtons. 
    #caster in deg, "side" = 1 or 0 for left or right
    trail = (splen * numpy.sin(caster) + tireRadius   * numpy.tan(caster)) #rudimentary mechanical trail, m
    scrub     = (splen * numpy.sin(kpi) + tireRadius   * numpy.tan(kpi)) #rudimentary scrub radius, m
    momentTrail = F_y * trail * numpy.cos(caster) #moment derived by mechanical trail, pneumatic trail omitted because i don't know how to model it
    if (side == LEFT):
         sign = 1
    if (side == RIGHT):
         sign = -1
    momentScrub = sign * F_y * scrub * numpy.cos(kpi)

    return momentTrail + momentScrub 
def solveRackForces(wheelInput, v_fwd, casterAngle, F_zL, F_zR): #need input LLT for Fz's, rest is self explanatory
    leftAngle, rightAngle = calculateSteerAngles(wheelInput) 
    leftSlip = ack.calculateSlipAngle(v_fwd, leftAngle)
    rightSlip = ack.calculateSlipAngle(v_fwd, rightAngle)
    rightTire = Tire(F_zL, slipRatio, leftSlip, v_fwd, pressure, temp, Parameters, Magic)
    leftTire = Tire(F_zR, slipRatio, rightSlip, v_fwd, pressure, temp, Parameters, Magic)
    F_yL = leftTire.getLateralForce()
    F_yR = rightTire.getLateralForce()
    # print(f"steer angles at {numpy.rad2deg(wheelInput)},  L/R: {numpy.rad2deg(leftAngle)}/{numpy.rad2deg(rightAngle)}")
    #calculation of Ls/Rs steering moments
    leftMomentComposite = solveSteerMoment(F_yL, casterAngle, LEFT)
    rightMomentComposite = solveSteerMoment(F_yR, casterAngle, RIGHT)
    #tie rod magic
    d_latL = d_lat1 + (rackRatio * wheelInput)
    d_latR = d_lat1 - (rackRatio * wheelInput)
    phi_rodL = numpy.arctan2(d1, d_latL)
    phi_rodR = numpy.arctan2(d1, d_latR)
    phi_armL = phiStatic + leftAngle
    phi_armR = phiStatic + rightAngle
    phi_includedL = phi_armL - phi_rodL
    phi_includedR = phi_armR - phi_rodR
    
    eff_armL = larm1 * numpy.sin(phi_includedL)
    eff_armR = larm1 * numpy.sin(phi_includedR)
    #forces and force projections, from rotational equilibrium [steer mom + TR force * effective mom arm = 0]
    tieRodForceL = leftMomentComposite / eff_armL #also if any of these moments are messed up or 0 the entire thing gets cooked, so watch out
    tieRodForceR = rightMomentComposite / eff_armR 
    #rack forces
    rackForceL = tieRodForceL * numpy.cos(phi_rodL)
    rackForceR = tieRodForceR * numpy.cos(phi_rodR)

    rackForce = rackForceL + rackForceR
    print(f"TLat L={F_yL:.1f}  TLat R={F_yR:.1f}  steer={numpy.rad2deg(wheelInput):.1f}deg")
    print(f"Mmt L={leftMomentComposite:.3f}  Mmt R={rightMomentComposite:.3f}")
    # print(f"eff_armL={eff_armL:.4f}  eff_armR={eff_armR:.4f}")
    print(f"Ftie L={tieRodForceL:.1f}  Ftie R={tieRodForceR:.1f}\n")
    # print(f"{rackForce} rack force at {numpy.rad2deg(wheelInput)} degrees steer and {numpy.rad2deg(casterAngle)} caster") 
    return rackForce
if __name__ == '__main__':
    velocity = 20 #m/s
    #step caster value [degs]
    #step steering wheel inputs [rads] 
    steerRange = numpy.arange(-0.7, 0.7, 0.1) #right is pos, left is neg
    casterRange = numpy.arange(0, 0.1658, 0.00873)  #in rad, 9.5deg max val, 0.5deg step
    
    cmap   = cm.plasma
    colours = [cmap(i / (len(casterRange) - 1)) for i in range(len(casterRange))]
    fig, ax = plt.subplots(figsize=(9, 6))
    
    #meat of the solver
    for casterStep, colour in zip(casterRange, colours):

        steer_data  = []
        torque_data = []

        for steerStep in steerRange:
            leftAngle, rightAngle = calculateSteerAngles(steerStep) 
            leftSlip = ack.calculateSlipAngle(velocity, leftAngle) 
            rightSlip = ack.calculateSlipAngle(velocity, rightAngle)

            inCorneringStiff = traction.getCorneringStiffness(wheelLoad, (leftSlip, 0), 0.15, velocity, 80, 40, Parameters, Magic)[0]
            outCorneringStiff = traction.getCorneringStiffness(wheelLoad, (rightSlip, 0),  0.15, velocity, 80, 40, Parameters, Magic)[0]
            stiff_Front = (inCorneringStiff + outCorneringStiff)/2 # ^^^i determine that in vs. out doesn't matter since it's just scalar addition in this line anyway
            stiff_Rear = -2000 # changing the arbitrary value to something larger 
            yR = ack.calculateYawRate(stiff_Front, stiff_Rear, velocity, steerStep)
            a_y = velocity*yR #lateral acceleration, m/s
            if (a_y > 0): #car is turning right?
                Fn_out, Fn_in = tireLoad.getLatLoadTransfer(Parameters, track, a_y, hcg) 
                # print(f"normal out, normal in:{Fn_out, Fn_in}, a_y = {a_y}")
                rackForce = solveRackForces(steerStep, velocity, casterStep, Fn_in, Fn_out)
            elif (a_y < 0): #car is turning left?
                Fn_out, Fn_in = tireLoad.getLatLoadTransfer(Parameters, track, a_y, hcg) 
                # print(f"normal out, normal in:{Fn_out, Fn_in}, a_y = {a_y}")
                rackForce = solveRackForces(steerStep, velocity, casterStep, Fn_out, Fn_in)
            else:
                rackForce = 0.0
            # print(f"rackForce {rackForce} a_y {a_y}")
            columnTorque = rackForce * pinionRadius
            steeringTorque = columnTorque / wheelRadius

            steer_data.append(numpy.rad2deg(steerStep))
            torque_data.append(steeringTorque)

        ax.plot(steer_data, torque_data,
            color=colour,
            linewidth=1.8,
            label=f'{numpy.degrees(casterStep):.1f}°')       

    #plot stuff
    

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)


    ax.set_xlabel('Turn Angle',     fontsize=12)
    ax.set_ylabel('Wheel Torque [N·m]',    fontsize=12)
    ax.set_title( 'Turn Angle vs. Wheel Torque, Caster Variations', fontsize=13)


    legend = ax.legend(
        title='Caster [deg]',
        loc='upper left',
        fontsize=9,
        title_fontsize=9,
        framealpha=0.85,
    )

    ax.grid(True, linestyle=':', alpha=0.5)
    fig.tight_layout()
    plt.show()