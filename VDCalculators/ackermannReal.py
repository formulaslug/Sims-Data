#steering wheel angle --> steering rack psition --> wheel steer angle (how static ackermann affects wheel angle function)
# equations taken from "rack and pinion" section of: https://www.mathworks.com/help/vdynblks/ref/kinematicsteering.html
import scipy
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets
from ipywidgets import interact, interactive

#global variables
#-----------------------------
# THIS SCRIPT USES MOSTLY FS-3 VALUES.
#-----------------------------
tw = 1.0833862 #fs4/3m (simplified track width from steering axis to steering axis [steering axis is also simplified to be A-arm knuckle to A-arm knuckle])
rackRatio = 82.55/248 #fs4/3 mm rack displacement/deg pinion rotation
wheelInput = 0.0 #in degrees of steering wheel movement (CW + CCW -)
rackShift = 0.0 # mm of movement of the rack from left to right (left is - right is +)
l_rack = 292.1 #fs4/3 mm (width of steering rack casing)
l_rod = 383.211 #fs4 mm (length of "tie rod" as left in FS-4 master CAD)
#l_rod = 378.434 #fs3
d = -33.642 #fs4 mm (plan view distance between front axis and rack. negative because we have a front steer setup)
#d = 122.598 #fs3
l_arm = 75.946 #fs4 mm (length of "steer arm", which is the distance from the center of the upright toe rod pickup to the KPA)
#l_arm = 91.04204 #fs3
#LWB = 1.589989 #fs3 m (length of wheelbase)
LWB = 1.524 #fs4 m lwb
m = 277.92 #[3] kg (mass of FS-3, with driver)
cornerRadius = 16.75 #meters, as per the rules
phiStatic = numpy.deg2rad(4.531) #fs4 degrees

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
    
def betaTrigSolver(l1): #a separate function to solve the trig equation (Gillespie's? Pulled from MATHWORKS site)
    l2 = numpy.sqrt((l1**2) + (d**2)) #l2 is the instantaneous direct distance from rack knuckle to steering axis (KPA)
    atan = numpy.arctan(d/l1) #first term of the "beta" equation

    num = (l_arm**2) + (l2**2) - (l_rod**2) #just simplifying the calculation of the second term 
    denom = 2*l_arm*l2
    frac = num/denom
    acos = numpy.arccos(frac)
    beta = (numpy.pi/2) - atan - acos
    return beta
    #return frac
def calculateAckermannIdeal(steerLeft, steerRight):
    angleOut = 0
    angleIn = 0
    if (steerLeft > 0):
        angleOut = steerRight
        angleIn = steerLeft
    if (steerLeft < 0):
        angleOut = steerLeft
        angleIn = steerRight
    inAckAngle = numpy.atan(numpy.tan(angleOut) + (LWB/tw))
    ackermannFactor = angleIn/inAckAngle
    return ackermannFactor

# def calebUSG(wiVal, yawRate, cornerRadius): #USG at fixed cornering radius, chalmer's formula
#     global wheelInput
#     wheelInput = wiVal
#     rho_Perfect = LWB/cornerRadius 
#     F_c = m*cornerRadius*yawRate**2
#     USG = (wheelInput - rho_Perfect)/F_c
#     return USG
#how does an ackermann value minimize or change the USG based on varying cornering speeds and a fixed cornering radius?
# - ackermann range 
# - varying yaw rate (minsteer/maxsteer from ackermann -> slip angles -> cornering stiffnesses - > (ackermann)step steer input + step velocity input))
# - return USG
def fixedCornerUSG(cornerRadius, wheelInput, vStep, steerInput, ackermannFactor): #USG plot with changing ackermann percentage, increasing velocity, fixed corner radius (for several cornering radii)
    return 0
def updateSusGeo(ackermannFactor): #a function that intakes the ackermann factor, and uses Dixon formula 5.3.9-10 to solve for new toe arm lenghts/rack placements.
    staI = ((numpy.pi)/2) * (tw/(2*LWB)) #pi/2 * T/2L for ideal ackermann
    staActual = ackermannFactor*staI #angle between steer arm and tie rod will determine rack location. 
    thetaAxis = phiStatic - staActual #angle of interest formed by the right triangle made by sketching together the tie rod knuckle and rack knuckle in plan view
    rackDisplacement = numpy.tan(thetaAxis)*tw #tan(thetaAxis) = delta_D/tw, tw known so use this relationship to solve for delta_D (rack displacement)
    return rackDisplacement, staI
angleIn, angleOut = calculateAckermann()
ackFac = calculateAckermannIdeal(angleIn, angleOut)
print(f"ackermann factor: {ackFac}")
# rack, staI = updateSusGeo(numpy.negative(0.05))
# print(f"rack distance: {rack}")
# print(f"toe arm angle to steer arm: {numpy.rad2deg(staI)}")
# def update(val): #update variables based off of interact()
#     global wheelInput
#     wheelInput = val
#     left_angle, right_angle = calculateAckermann()
#     #stat, bL, bR = calculateAckermann()
#     print(f"wheelInput = {wheelInput}")
#     print(f"rack movement = {rackMovement()}")
#     print("----------------------------")
#     print(f"left wheel radians = {left_angle}")
#     print(f"right wheel radians = {right_angle}")
#     print("----------------------------")
#     print(f"left wheel degrees = {numpy.rad2deg(left_angle)}")
#     print(f"right wheel degrees = {numpy.rad2deg(right_angle)}")
#     #print(f"Static value must be within [-1,1] = {stat}")
#     #print(f"Left value must be within [-1,1] = {bL}")
#     #print(f"Right value must be within [-1,1] = {bR}")

# interact( #ui
#     update,
#     val=widgets.FloatSlider(value=0.0,min=-90,max=90,step=1,description="Deg Wheel Input")
# )