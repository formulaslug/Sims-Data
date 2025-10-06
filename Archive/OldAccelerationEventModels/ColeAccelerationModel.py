import math
from AccelerationEventModels import CarData
# import matplotlib.pyplot as plt
# import numpy as np

#constants
simLength:float = 5.0       # how many seconds to run the simulation for
simStepsPerSec:int = 100    # how many simulation steps per second
throttle:float = 1          # percent of max torque to demand
initDischarged:float = 2.8  # how many Ahs have already been expended on the car, 2.8 yeilds a voltage of about 115
maxPower:float = 80.0       # 80kW max

def runSim(slipRatio):
    simulation = Simulation()
    simulation.runSim(slipRatio)
    simulation.exportResults()

class Simulation:
    
    steps = []

    timeTo100 = -1

    def __init__(self):
        return
    
    def runSim(self, slipRatio):

        timeDelta = (1.0/simStepsPerSec)

        initStep = SimulationStep(slipRatio)
        initStep.firstStep(initDischarged, timeDelta, slipRatio)
        self.steps.append(initStep)
        for i in range(1, math.floor(simLength*simStepsPerSec)):
            prevStep = self.steps[i-1]
            step = SimulationStep(slipRatio)
            step.calculateStep(prevStep, timeDelta)
            self.steps.append(step)

            if (self.timeTo100 == -1 and step.velocity * 3.6 >= 65):
                self.timeTo100 = step.time

            print(f"{round(step.time, 5)}s: motorTorque: {round(step.motorTorque, 3)}, rpm: {round(step.motorSpeed, 3)}, force: {round(step.netForce, 3)}, vel: {round(step.velocity, 3)}, pow: {round(step.motorPower, 3)}, current: {round(step.currentDraw, 3)}, voltage: {round(step.batteryVoltage, 3)}, charge: {round(100*(CarData.CellsParallel*CarData.Capacity - step.totalDischarged)/(CarData.CellsParallel*CarData.Capacity), 3)}")

        print(f"Time to 100 kmh: {self.timeTo100} seconds")
        return
    
    def exportResults(self):
        file = open("Cole_FS-2_Accel.csv", "w")
        data = "Time (s),Delta Time (s),Velocity (m/s),Battery Voltage (V),Motor Speed (RPM),Throttle Demand,Motor Torque (Nm),Motor Power (kW),Current Draw (A),Tractive Force (N),Drag Force (N),Net Force (N),Total Discharged (Ah),Delta Discharged (Ah)\n"
        for step in self.steps:
            data += f"{step.time},{step.deltaTime},{step.velocity},{step.batteryVoltage},{step.motorSpeed},{step.throttleDemand},{step.motorTorque},{step.motorPower},{step.currentDraw},{step.tractiveForce},{step.dragForce},{step.netForce},{step.totalDischarged},{step.deltaDischarged}\n"
        file.write(data)
        file.close
        return


class SimulationStep:

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
    dragForce:float         # Newtons, positive is backward, at start of step
    netForce:float          # Newtons, positive is forward, at start of step
    totalDischarged:float   # Ah, at start of step
    deltaDischarged:float   # Ah, discharged until next step
    slipRatio:float         # Unitless, slip ratio


    def __init__(self, slipRatio):
        self.slipRatio = slipRatio
        return
    
    def calculateStep(self, prevStep, timeDelta):
        self.deltaTime = timeDelta
        self.time = prevStep.time + self.deltaTime
        self.velocity = prevStep.velocity + self.deltaTime * prevStep.netForce / CarData.TotalMass
        self.totalDischarged = prevStep.totalDischarged + prevStep.deltaDischarged
        self.batteryVoltage = CarData.lookupLerpVoltage(prevStep.currentDraw / CarData.CellsParallel, self.totalDischarged / CarData.CellsParallel) * CarData.CellsSeries
        self.motorSpeed = self.calculateMotorRPMfromVelocity(self.velocity)
        self.throttleDemand = throttle
        self.torqueDemand = self.throttleDemand * CarData.lookupLerpTorque(self.motorSpeed)
        self.motorTorque = self.calculateRealTorque(self.torqueDemand, self.batteryVoltage, self.motorSpeed)
        self.motorPower = self.calculateMotorPower(self.motorTorque, self.motorSpeed)
        self.currentDraw = self.calculateCurrentDraw(self.motorPower, self.batteryVoltage)
        self.deltaDischarged = self.calculateDischargeDelta(self.currentDraw, self.deltaTime)

        self.tractiveForce = self.calculateTractiveForce(self.motorTorque) # assuming two driving wheels
        self.dragForce = self.calculateDragForce()
        
        self.netForce = self.calculateNetForce(self.tractiveForce, self.dragForce)

        return
    
    def firstStep(self, initialDischarged, timeDelta, slipRatio):
        self.slipRatio = slipRatio
        self.time = 0
        self.deltaTime = timeDelta
        self.velocity = 0
        self.totalDischarged = initialDischarged
        self.batteryVoltage = CarData.lookupLerpVoltage(0, self.totalDischarged / CarData.CellsParallel) * CarData.CellsSeries
        self.motorSpeed = 0
        self.throttleDemand = throttle
        self.torqueDemand = self.throttleDemand * CarData.lookupLerpTorque(self.motorSpeed)
        self.motorTorque = self.calculateRealTorque(self.torqueDemand, self.batteryVoltage, self.motorSpeed)
        self.motorPower = self.calculateMotorPower(self.motorTorque, self.motorSpeed)
        self.currentDraw = self.calculateCurrentDraw(self.motorPower, self.batteryVoltage)
        self.deltaDischarged = self.calculateDischargeDelta(self.currentDraw, self.deltaTime)

        self.tractiveForce = self.calculateTractiveForce(self.motorTorque)
        self.dragForce = self.calculateDragForce()
        
        self.netForce = self.calculateNetForce(self.tractiveForce, self.dragForce)

        return self
    
    # def calculateMaxTireFriction(self):
    #     wheelNormal = CarData.TotalMass * CarData.Gravity * (1-CarData.FrontRearBalance) # normal on both rear wheels
    #     return wheelNormal * CarData.LongitudinalFrictionCoef
    # this is before load transfer
    
    def calculateMaxTireFriction(self):
        #friction =  ((self.pacejka94(self.slipRatio) * CarData.TotalMass * CarData.Gravity * CarData.LongitudinalDistanceFrontAxleCoM / CarData.Wheelbase) / (1 - (CarData.CenterOfMassHeight / CarData.Wheelbase)*CarData.LongitudinalFrictionCoef))
        
        #print(friction)

        #return friction
        return ((CarData.LongitudinalFrictionCoef * CarData.TotalMass * CarData.Gravity * CarData.LongitudinalDistanceFrontAxleCoM / CarData.Wheelbase) / (1 - (CarData.CenterOfMassHeight / CarData.Wheelbase)*CarData.LongitudinalFrictionCoef))
    
    def calculateTractiveForce(self, motorTorque:float):
        axleTorque = motorTorque * CarData.DrivenGearTeeth / CarData.DrivingGearTeeth # torque for both rear wheels
        wheelForce = axleTorque / (CarData.WheelDiameter/2) # T = Fr so F = T/r

        return wheelForce

    def calculateDragForce(self):
        #uses FD = 1/2 * p * v^2 * Cd * A
        return 0.5 * CarData.AirDensity * math.pow(self.velocity, 2) * CarData.DragCoef * CarData.FrontArea

    def calculateNetForce(self, tractiveForce: float, dragForce: float):
        return tractiveForce - dragForce
    
    def calculateVelocityfromMotorRPM(self, motorRPM:float):
        return (motorRPM * CarData.DrivingGearTeeth * math.pi * CarData.WheelDiameter)/(60*CarData.DrivenGearTeeth) # turns RPM into m/s
    
    def calculateMotorRPMfromVelocity(self, velocity:float):
        return (velocity*60*CarData.DrivenGearTeeth)/(CarData.DrivingGearTeeth * math.pi * CarData.WheelDiameter) # turns m/s into RPM
    
    def calculateMotorPower(self, motorTorque, motorRPM):
        return motorTorque*motorRPM/9550
    
    def calculateCurrentDraw(self, motorPower, motorVoltage):
        return 1000.0*motorPower/motorVoltage # times 1000 to account for power being kW
    
    def pacejka94(self, slip_ratio):
        import numpy as np
        B = 10.0   # stiffness factor
        C = 1.9    # shape factor
        D = 1.0    # peak factor (maximum friction coefficient)
        E = 0.97   # curvature factor
        term1 = B * slip_ratio
        term2 = E * (term1 - np.arctan(term1))
        friction_force = D * np.sin(C * np.arctan(term1 - term2))

        #return 1.5
        return friction_force

    def calculateRealTorque(self, torqueDemand, batVoltage, motorRPM):
        currentTorqueLim = torqueDemand
        if (motorRPM != 0):
            currentLim = min(CarData.MaxDischargeCurrent*CarData.CellsParallel, maxPower/batVoltage*1000)
            currentTorqueLim = currentLim * batVoltage * 9550 / (motorRPM * 1000)

        tractionTorqueLim = self.calculateMaxTireFriction() * (CarData.WheelDiameter/2) * CarData.DrivingGearTeeth / CarData.DrivenGearTeeth

        print(f"{torqueDemand}, {currentTorqueLim}, {tractionTorqueLim}")
        return min(torqueDemand, currentTorqueLim, tractionTorqueLim)
    
    def calculateDischargeDelta(self, currentDraw, timeDelta):
        return currentDraw * timeDelta / 3600 # divide by 3600 to convert from amp seconds to amp hours
