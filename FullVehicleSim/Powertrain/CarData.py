import math

#Track
Gravity:float = 9.8 # m/s^2
AirDensity:float = 1.293 # kg/m^3

#Mass
VehicleMass:int = 0
DriverMass:int = 0
ACCMass:int = 0
TotalMass:float = 300 #kgs should be VehicleMass + DriverMass + ACCMass. We measured vehicle plus acc to be 231 kg and assuming 70 kg driver gives ~300 kg
# FrontRearBalance:float = 0.5 # 0 indicates all weight at rear, 1 indicates all weight at front
LongitudinalDistanceFrontAxleCoM:float = 0.7477 # meters longitudinal distance from front axle to center of gravity
CenterOfMassHeight:float = 0.247718 # meters center of mass height
Wheelbase:float = 1.6547084 # meters wheelbase

#CenterOfGravity = []



WheelDiameter:float = 0.46 #From onshape, meters
#Drivetrain
MotorV:int
MotorA:int
DrivingGearTeeth:int = 11 #From onshape
DrivenGearTeeth:int = 40 #From onshape
GearingRatio:int = DrivenGearTeeth/DrivenGearTeeth
#reflected mass inertia?? https://www.motioncontroltips.com/how-do-gearmotors-impact-reflected-mass-inertia-from-the-load/


#Suspension
RollCenter=0


#Aerodynamics
FrontArea:float = 0.693 #  m^2, TBD from OptimumLap
DragCoef:float = 0.63   # TBD from OptimumLap


#Accumulator
CellsSeries:int = 28                # number of cells in series
CellsParallel:int = 20              # number of cells in parallel

#Accumulator Cell
InternalImpedance:float = 0.0013    # internal impedance in ohms (1.3mO)

CurrentVoltageChargeTable = (
    (0  , [(0, 4.2 ), (0.02, 4.16), (0.05, 4.14), (0.1, 4.12), (0.25, 4.07), (0.5, 3.96), (0.75, 3.87), (1, 3.78), (1.25, 3.68), (1.5, 3.61), (1.75, 3.56), (2, 3.49), (2.25, 3.37), (2.35, 3.33), (2.45, 3.27), (2.55, 2.91), (2.6, 2.5 )]),
    (0.2, [(0, 4.17), (0.02, 4.15), (0.05, 4.13), (0.1, 4.11), (0.25, 4.06), (0.5, 3.95), (0.75, 3.86), (1, 3.77), (1.25, 3.67), (1.5, 3.6 ), (1.75, 3.55), (2, 3.48), (2.25, 3.36), (2.35, 3.32), (2.45, 3.26), (2.55, 2.9 ), (2.6, 2.5 )]),
    (1  , [(0, 4.15), (0.02, 4.12), (0.05, 4.1 ), (0.1, 4.08), (0.25, 4.03), (0.5, 3.91), (0.75, 3.82), (1, 3.73), (1.25, 3.64), (1.5, 3.57), (1.75, 3.52), (2, 3.45), (2.25, 3.32), (2.35, 3.27), (2.45, 3.16), (2.55, 2.7 )]),
    (2  , [(0, 4.14), (0.02, 4.1 ), (0.05, 4.07), (0.1, 4.05), (0.25, 3.99), (0.5, 3.87), (0.75, 3.78), (1, 3.7 ), (1.25, 3.6 ), (1.5, 3.53), (1.75, 3.47), (2, 3.4 ), (2.25, 3.28), (2.35, 3.21), (2.45, 3.01), (2.55, 2.51)]),
    (3  , [(0, 4.12), (0.02, 4.07), (0.05, 4.05), (0.1, 4.02), (0.25, 3.96), (0.5, 3.84), (0.75, 3.75), (1, 3.67), (1.25, 3.57), (1.5, 3.5 ), (1.75, 3.43), (2, 3.36), (2.25, 3.24), (2.35, 3.15), (2.45, 2.91), (2.55, 2.5 )]),
    (5  , [(0, 4.09), (0.02, 4.03), (0.05, 4.0 ), (0.1, 3.97), (0.25, 3.9 ), (0.5, 3.79), (0.75, 3.7 ), (1, 3.61), (1.25, 3.51), (1.5, 3.44), (1.75, 3.38), (2, 3.3 ), (2.25, 3.18), (2.35, 3.09), (2.45, 2.87), (2.55, 2.5 )]),
    (10 , [(0, 4.0 ), (0.02, 3.93), (0.05, 3.89), (0.1, 3.85), (0.25, 3.79), (0.5, 3.69), (0.75, 3.59), (1, 3.5 ), (1.25, 3.41), (1.5, 3.34), (1.75, 3.27), (2, 3.19), (2.25, 3.09), (2.35, 3.01), (2.45, 2.87), (2.55, 2.5 )]),
    (20 , [(0, 3.82), (0.02, 3.74), (0.05, 3.7 ), (0.1, 3.66), (0.25, 3.58), (0.5, 3.49), (0.75, 3.4 ), (1, 3.32), (1.25, 3.24), (1.5, 3.18), (1.75, 3.12), (2, 3.05), (2.25, 2.95), (2.35, 2.89), (2.45, 2.76)]),
    (30.01 , [(0, 3.62), (0.02, 3.55), (0.05, 3.49), (0.1, 3.45), (0.25, 3.36), (0.5, 3.26), (0.75, 3.18), (1, 3.13), (1.25, 3.08), (1.5, 3.03), (1.75, 2.98), (2, 2.92), (2.25, 2.82), (2.35, 2.7 )])
) # ((current, [(discharge, voltage), ...]), ...)
# 30.01 accounts for floating point errors

def lookupLerpVoltage(current:float, discharge:float): # current in amps, discharge is Ahs discharged, this is a bilinear interpolation
    for i in range(1, len(CurrentVoltageChargeTable)):
        highCurrent = CurrentVoltageChargeTable[i]

        if highCurrent[0] > current:
            
            lowCurrent = CurrentVoltageChargeTable[i-1]
            lcVoltage:float = 0
            for j in range(1, len(lowCurrent[1])):
                highDischarge = lowCurrent[1][j]
                if highDischarge[0] > discharge:
                    lowDischarge = lowCurrent[1][j-1]
                    lcVoltage = lowDischarge[1] + (highDischarge[1]-lowDischarge[1])*((discharge-lowDischarge[0])/(highDischarge[0]-lowDischarge[0]))
                    break
            
            hcVoltage:float = 0
            for k in range(1, len(highCurrent[1])):
                highDischarge = highCurrent[1][k]
                if highDischarge[0] > discharge:
                    lowDischarge = highCurrent[1][k-1]
                    hcVoltage = lowDischarge[1] + (highDischarge[1]-lowDischarge[1])*((discharge-lowDischarge[0])/(highDischarge[0]-lowDischarge[0]))
                    break
            
            return lcVoltage + (hcVoltage-lcVoltage)*((current-lowCurrent[0])/(highCurrent[0]-lowCurrent[0]))
    return MinVoltage
            

MinVoltage:float = 2.5              # volts
MaxDischargeCurrent:float = 30      # amps
Capacity:float = 2.5                # Ah

#motor
RPMtorqueTable = [(0, 178.4), (4120, 178.4), (7500, 82)]  # data is [(RPM, Nm), ...]
def lookupLerpTorque(rpm:float):
    if (rpm < 0):
        return 0
    for i in range(1, len(RPMtorqueTable)):
        tup = RPMtorqueTable[i]
        if tup[0] == rpm:
            return tup[1]
        elif tup[0] > rpm:
            prevTup = RPMtorqueTable[i-1]
            return (prevTup[1] + ((tup[1]-prevTup[1]) * ((rpm-prevTup[0])/(tup[0]-prevTup[0]))))
    return 0

#tires
LongitudinalFrictionCoef = 1.5      # coef of static friction