import math 
from MBS import CarData
import copy

class MBS:
    time:float              # seconds since the sim started, at start of step
    deltaTime:float         # seconds, time until next step
    velocity:float          # m/s, at start of step
    batteryVoltage:float    # volts, at start of step
    motorSpeed:float        # rpm, at start of step
    throttleDemand:float    # 0 to 1, at start of step
    motorTorque:float       # Nm, at start of step
    motorPower:float        # kW, at start of step
    currentDraw:float       # Amperes, at start of step
    tractiveForce:float     # Newtons, positive is forward, at start of step
    netForce:float          # Newtons, positive is forward, at start of step
    totalDischarged:float   # Ah, at start of step
    deltaDischarged:float   # Ah, discharged until next step


    def __init__(self, throttle, initDischarged, maxPower):
        self.throttle = throttle       # percent of max torque to demand
        self.initDischarged = initDischarged # how many Ahs have already been expended on the car, 2.8 yeilds a voltage of about 115
        self.maxPower = maxPower     # 80kW max
        #return self 
    
    def calculateStep(self, prevStep, timeDelta, tractionForce):
        self.frictionForce = tractionForce
        self.deltaTime = timeDelta
        self.time = prevStep.time + self.deltaTime
        self.velocity = prevStep.velocity + self.deltaTime * prevStep.netForce / CarData.TotalMass
        self.totalDischarged = prevStep.totalDischarged + prevStep.deltaDischarged
        self.batteryVoltage = CarData.lookupLerpVoltage(prevStep.currentDraw / CarData.CellsParallel, self.totalDischarged / CarData.CellsParallel) * CarData.CellsSeries
        self.motorSpeed = self.calculateMotorRPMfromVelocity(self.velocity)
        self.throttleDemand = self.throttle
        self.torqueDemand = self.throttleDemand * CarData.lookupLerpTorque(self.motorSpeed)
        self.motorTorque = self.calculateRealTorque(self.torqueDemand, self.batteryVoltage, self.motorSpeed)
        self.motorPower = self.calculateMotorPower(self.motorTorque, self.motorSpeed)
        self.currentDraw = self.calculateCurrentDraw(self.motorPower, self.batteryVoltage)
        self.deltaDischarged = self.calculateDischargeDelta(self.currentDraw, self.deltaTime)

        self.tractiveForce = self.calculateTractiveForce(self.motorTorque) # assuming two driving wheels
        return copy.deepcopy(self) 
    
    def firstStep(self, initialDischarged, timeDelta, slipRatio):
        self.frictionForce = 0
        self.slipRatio = slipRatio
        self.time = 0
        self.deltaTime = timeDelta
        self.velocity = 0
        self.totalDischarged = initialDischarged
        self.batteryVoltage = CarData.lookupLerpVoltage(0, self.totalDischarged / CarData.CellsParallel) * CarData.CellsSeries
        self.motorSpeed = 0
        self.throttleDemand = self.throttle
        self.torqueDemand = self.throttleDemand * CarData.lookupLerpTorque(self.motorSpeed)
        self.motorTorque = self.calculateRealTorque(self.torqueDemand, self.batteryVoltage, self.motorSpeed)
        self.motorPower = self.calculateMotorPower(self.motorTorque, self.motorSpeed)
        self.currentDraw = self.calculateCurrentDraw(self.motorPower, self.batteryVoltage)
        self.deltaDischarged = self.calculateDischargeDelta(self.currentDraw, self.deltaTime)

        self.tractiveForce = self.calculateTractiveForce(self.motorTorque)

        print("GOT TO CREATION OF FIRST STEP")
        return self
    
    
    def calculateMaxTireFriction(self):
        return ((self.frictionForce * CarData.LongitudinalDistanceFrontAxleCoM / CarData.Wheelbase) / (1 - (CarData.CenterOfMassHeight / CarData.Wheelbase)* (self.frictionForce/9.8/CarData.TotalMass)))
    
    def calculateTractiveForce(self, motorTorque:float):
        axleTorque = motorTorque * CarData.DrivenGearTeeth / CarData.DrivingGearTeeth # torque for both rear wheels
        wheelForce = axleTorque / (CarData.WheelDiameter/2) # T = Fr so F = T/r

        return wheelForce
    
    def calculateVelocityfromMotorRPM(self, motorRPM:float):
        return (motorRPM * CarData.DrivingGearTeeth * math.pi * CarData.WheelDiameter)/(60*CarData.DrivenGearTeeth) # turns RPM into m/s
    
    def calculateMotorRPMfromVelocity(self, velocity:float):
        return (velocity*60*CarData.DrivenGearTeeth)/(CarData.DrivingGearTeeth * math.pi * CarData.WheelDiameter) # turns m/s into RPM
    
    def calculateMotorPower(self, motorTorque, motorRPM):
        return motorTorque*motorRPM/9550
    
    def calculateCurrentDraw(self, motorPower, motorVoltage):
        return 1000.0*motorPower/motorVoltage # times 1000 to account for power being kW

    def calculateRealTorque(self, torqueDemand, batVoltage, motorRPM):
        currentTorqueLim = torqueDemand
        if (motorRPM != 0):
            currentLim = min(CarData.MaxDischargeCurrent*CarData.CellsParallel, self.maxPower/batVoltage*1000)
            currentTorqueLim = currentLim * batVoltage * 9550 / (motorRPM * 1000)

        tractionTorqueLim = self.calculateMaxTireFriction() * (CarData.WheelDiameter/2) * CarData.DrivingGearTeeth / CarData.DrivenGearTeeth

        return min(torqueDemand, currentTorqueLim, tractionTorqueLim)
    
    def calculateDischargeDelta(self, currentDraw, timeDelta):
        return currentDraw * timeDelta / 3600 # divide by 3600 to convert from amp seconds to amp hours