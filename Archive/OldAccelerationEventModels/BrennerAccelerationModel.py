from CarData import *
import math


def IPStoMPH(InchesPerMin:int):
    return (60/1) * (1/63360) * InchesPerMin

def GetVelocity(wheelRPM, WheelDiameter):
    InchesPerMin= wheelRPM * 2 * math.py * (WheelDiameter/2)
    return IPStoMPH(InchesPerMin)

def GetDumbAcceleration(V1,V2,T1,T2):
    #acceleration of car = acceleration of motor * gear ratio then conversion
    #inputs are rudementary V1,V2 are motor velocity T1,T2 are times
    DeltaTime = T2-T1
    DeltaVelocity = V2-V1
    Acceleration = (DeltaVelocity*GearingRatio)
    DeltaMPH = IPStoMPH(Acceleration) #IPS^2 to MPH(s)
    #This needs to be validated





def GetForceOfDrag():
    ForceOfDrag = 0 #ignore air-resistance haha 
    #this will come from the aerodynamics model

def GetRollingResistance(CoeffOfResistance:list, NormalForceOnTire:list):
    #Coeff of resistance will be the "friction" of all mechanical components
    #each element in the array will be the resistance of the corner
    #it should go Front Left, Front Right, Rear Left, Rear Right
    #Normal force on tire should be the data we directly pull from the StrainGauges
    #In our model for FS-3 this force will be 1/4 mass + 1/4downforce, this is an assumption
    FrontLeftResistance=CoeffOfResistance[0] * NormalForceOnTire[0]
    FrontRightResistance=CoeffOfResistance[1] * NormalForceOnTire[1]
    RearLeftResistance=CoeffOfResistance[2] * NormalForceOnTire[2]
    RearRightResistance=CoeffOfResistance[3] * NormalForceOnTire[3]
    TotalRollingResistance = FrontLeftResistance + FrontRightResistance + RearLeftResistance + RearRightResistance
    return TotalRollingResistance

def GetForceOfGravity(weight:int, slope:int):
    TrackSlope:int = 0 #degrees
    #Track slope must not be confused with carslope, suspension travel must be accounted for
    weight * math.sin(TrackSlope) #make sure that this value can be negative for declines

# def GetForceAtWheels(MotorRpm:int, TorqueAtRPM:int, )

def GetSumOfForces(FWheels, FDrag, FGravity, FRollingResistance):
    #To account for limited traction we need to limit FWheels, this should be done through a tire model
    
    print(FWheels)


if __name__ == "__main__":

    #acceleration can be found by subtracting friction force at the wheels by all resisting forces put in terms of the car's weight

    WheelForce = 0
    drag = GetForceOfDrag()
    rollingResistance = GetRollingResistance([0,0,0,0],TotalMass/4)
    forceOfGravity = GetForceOfGravity(TotalMass,0)




    print("here")