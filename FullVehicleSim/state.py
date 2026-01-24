import numpy as np
from ramen import Parameters, Magic
from dataclasses import dataclass

# Our libraries
# import yogurt as stepWorld
# from TireModel import dumpling as tire
# from Mech.mechanical import *
from Powertrain.lionCellModel import *
from Mech.aero import calculateDrag
from Mech.braking import *
from Mech.steering import *
from Mech.tireLoad import *
from Mech.traction import *

@dataclass
class VehicleState:
    def __init__(self, stepSize, position:np.ndarray, speed:float, acceleration:np.ndarray, heading, charge, lastCurrent, throttle, brakes, yawRate, steerAngle, brakeTemperature, timeSinceLastSteer, initSpeed):
        self.stepSize:float = stepSize
        self.initYawRate:float = yawRate
        self.steerAngle:float = steerAngle
        # self.brakes:float = brakes
        # self.throttle:float = throttle
        self.position:np.ndarray = position
        self.speed:float = speed
        self.initAcceleration:np.ndarray = acceleration
        self.heading:np.ndarray = heading
        self.charge:float = charge
        self.lastCurrent:float = lastCurrent
        self.WheelCircumference:float = Parameters["wheelCircumferance"]
        self.WheelRadius:float = Parameters["wheelRadius"]
        self.GearRatio:float = Parameters["gearRatio"]
        self.TorqueMax:float = Parameters["maxTorque"]
        self.tractiveIMax:float = Parameters["tractiveIMax"]
        self.brakeTemperature:float = brakeTemperature
        self.timeSinceLastSteer:float = timeSinceLastSteer
        self.initSpeed:float = initSpeed

        #self.wheelRPM: np.array = np.asarray([0,0,0,0], dtype=np.float32)
        #self.wheelRotationsHz: float = self.speed / self.WheelCircumference * 2.0 * np.pi
        self.tires:np.ndarray = np.asarray([None, None, None, None])#, dtype=tire.Tire) # [FL, FR, BL, BR]

    ## Not a property, fix.
    @property
    def yawRate(self):
        tireLoad = getloadTransfer(Parameters, self.initAcceleration * self.heading[0], self.initAcceleration * self.heading[1], self.initYawRate)
        slipAngle = calculateSlipAngle(self.initYawRate, self.velocity, self.steerAngle, Parameters)
        slipRatio = 0.15
        corneringStiffness = getCorneringStiffness(tireLoad, slipAngle, slipRatio, self.speed, 80, 40, Parameters, Magic) # Works but unused
        res = calculateYawRate(self.initYawRate, self.initSpeed, self.steerAngle, self.timeSinceLastSteer,corneringStiffness[0], corneringStiffness[1], Parameters)

        return res

    # @property
    # def speed(self):
    #     return np.sqrt(np.sum(self.velocity**2))

    @property
    def velocity(self):
        return self.heading * self.speed

    ## Not a property, fix.
    @property
    def drag(self):
        return calculateDrag(self.heading, self.speed)

    ## Not a property, fix.
    @property
    def resistiveForces(self):
        if self.speed <= 1e-5: # Floating point error
            return 0
        elif self.brakes == 0:
            return self.drag
        else:
            brakeForce, self.brakeTemperature = getBrakeForce(self.speed, self.brakeTemperature, self.stepSize, Parameters)
            return -1 * (self.drag + brakeForce)

    ## Not a property, fix.
    @property
    def cooledBrakeTemperature(self):
        return calculateBrakeCooling(self.brakeTemperature, self.stepSize, Parameters)

    @property
    def calcWheelRPM(self):
        return self.speed / self.WheelCircumference * 60.0

    @property
    def wheelRotationsHZ(self):
        return self.speed / self.WheelCircumference * 2.0 * np.pi

    @property
    def rpm(self):
        return self.calcWheelRPM * self.GearRatio

    @property
    def motorRotationsHZ(self):
        return self.wheelRotationsHZ * self.GearRatio

    @property
    def maxPower(self):
        return self.tractiveIMax * self.voltage

    @property
    def maxWheelTorque(self):
        '''
        maxMotorTorque * gear rato
        '''
        return self.maxMotorTorque * self.GearRatio

    @property
    def maxMotorTorque(self):
        '''
        Motor Torque at the wheel
        
        minimum(rpm limited torque, power limited torque, perfect traction torque)
        '''
        ## RPM Limited Torque (Motor Controller limits it to ~ this in practice. Maybe something more like 7490ish)
        if self.rpm > 7490:
            return -1 * self.resistiveForces * self.WheelRadius
        if self.motorRotationsHZ != 0: ## If rolling, torque may be power limited. 
            maxPowerTorque = self.maxPower / self. motorRotationsHZ * self.GearRatio
        else: ## Avoid divide by 0 error but it's just the same as the max torque that the motor can deliver (180 Nm)
            maxPowerTorque = 180.0 # Nm at 0 rpm
        perfectTractionTorque = self.TorqueMax
        torque = min(perfectTractionTorque, maxPowerTorque, self.maxTractionTorqueAtWheel/self.GearRatio)
        return torque

    @property
    def voltage(self):
        return 28.0 * lookup(self.charge, self.lastCurrent)

    @property
    def power(self):
        return np.linalg.norm(self.maxMotorTorque) * self.motorRotationsHZ

    @property
    def current(self):
        if (self.power / self.voltage) > self.tractiveIMax:
            return self.tractiveIMax
        return self.power / self.voltage

    @property
    def maxTraction(self):
        tireLoad = getloadTransfer(Parameters, self.initAcceleration * self.heading[0], self.initAcceleration * self.heading[1], self.initYawRate) # yaw velocity is currently set to 0

        slipAngle = calculateSlipAngle(self.initYawRate, self.velocity, self.steerAngle, Parameters)
        slipRatio = 0.15
        tireTraction = getTraction(tireLoad, slipAngle, slipRatio, self.speed, 80, 40, Parameters, Magic)
        longTraction = 0
        latTraction = 0
        for x, y in tireTraction:
            longTraction += x
            latTraction += y
        return np.sqrt(longTraction**2 + latTraction**2)

        #tempTire = tire.Tire(500 , 0.15, 0, self.speed, 80, 40, Parameters, Magic)
        #return  ((tempTire.getLongForce()/500 * self.weight * 0.7477)/1.6547084)/(1.0-(0.247718 * tempTire.getLongForce()/500 / 1.6547084))

    @property
    def maxTractionTorqueAtWheel(self):
        return self.maxTraction * self.WheelRadius

    @property
    def motorForce(self):
        return (self.maxWheelTorque / self.WheelRadius)

    @property
    def netForce(self):
        return self.motorForce + self.resistiveForces

    @property
    def acceleration(self):
        return self.netForce / Parameters["Mass"]

    def logProperties(self):
        return [self.position[0], self.position[1],
                self.velocity[0], self.velocity[1],
                self.speed, self.acceleration,
                self.heading[0], self.heading[1],
                self.yawRate,
                self.steerAngle, self.throttle,
                self.brakes,
                self.drag, self.resistiveForces,
                self.motorForce, self.netForce,
                self.maxWheelTorque, self.maxMotorTorque,
                self.maxTraction, self.maxTractionTorqueAtWheel,
                self.cooledBrakeTemperature,
                self.calcWheelRPM, self.wheelRotationsHZ,
                self.rpm, self.motorRotationsHZ,
                self.charge, self.voltage,
                self.current, self.power,
                self.maxPower,
                self.stepSize,
                self.timeSinceLastSteer
            ]
