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
# THIS SCRIPT USES MOSTLY FS-3 VALUES. FS-3 values denoted by [3], any theoretical or FS-4 values denoted by [4]
#-----------------------------
tw = 1083.3862 #mm (simplified track width from steering axis to steering axis [steering axis is also simplified to be A-arm knuckle to A-arm knuckle])
rackRatio = 82.55/248 #[4] mm rack displacement/deg pinion rotation
wheelInput = 0.0 #in degrees of steering wheel movement (CW + CCW -)
rackShift = 0.0 # mm of movement of the rack from left to right (left is - right is +)
l_rack = 292.1 #[4] mm (width of steering rack casing)
l_rod = 378.9426 #[3] mm (length of tie rod as left in FS-3 master CAD)
d = 109.7788 #[3] mm (plan view distance between front axis and rack. negative because we have a front steer setup)
l_arm = 71.628 #[3] mm (length of "steer arm", which is the distance from the center of the upright toe rod pickup to the KPA)
LWB = 1589.989 #[3] mm (length of wheelbase)
m = 277.92 #[3] kg (mass of FS-3, with driver)
cornerRadius = 16.75 #meters, as per the rules

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
def fixedCornerUSG(cornerRadius, wheelInput, vStep, steerInput): #USG plot with changing ackermann percentage, increasing velocity, fixed corner radius (for several cornering radii)
    return 0
def rackLocateEasy(ackermannFactor):
    StaI = #pi/2 * T/2L for ideal ackermann.
    angle = ackermannFactor*StaI #angle between steer arm and tie rod will determine rack location. 
    return angle





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