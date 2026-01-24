import numpy as np
from paramLoader import Parameters, Magic
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
    def __init__(self, stepSize, position:np.ndarray, speed:float, acceleration:np.ndarray, heading, charge, yawRate, steerAngle, brakeTemperature, timeSinceLastSteer, initSpeed):
        self.position:np.ndarray = position
        self.speed:float = speed
        self.heading:np.ndarray = heading
        self.charge:float = charge
        self.brakeTemperature:float = brakeTemperature
        self.yawRate:float = yawRate

        #self.wheelRPM: np.array = np.asarray([0,0,0,0], dtype=np.float32)
        #self.wheelRotationsHz: float = self.speed / self.WheelCircumference * 2.0 * np.pi
        self.tires:np.ndarray = np.asarray([None, None, None, None])#, dtype=tire.Tire) # [FL, FR, BL, BR]

    @property
    def velocity(self):
        return self.heading * self.speed

    ## Not a property, fix.
    @property
    def resistiveForces(self):
        if self.speed <= 1e-5: # Floating point error
            return 0
        elif self.brakes == 0:
            return calculateDrag(self.heading, self.speed)
        else:
            brakeForce, self.brakeTemperature = getBrakeForce(self.speed, self.brakeTemperature, self.stepSize, Parameters)
            return -1 * (calculateDrag(self.heading, self.speed) + brakeForce)

    ## Not a property, fix.
    @property
    def cooledBrakeTemperature(self):
        return calculateBrakeCooling(self.brakeTemperature, self.stepSize, Parameters)

    @property
    def wheelRPM(self):
        return self.speed / Parameters["wheelCircumferance"] * 60.0

    @property
    def wheelRotationsHZ(self):
        return self.speed / Parameters["wheelCircumferance"] * 2.0 * np.pi
    @property
    def motorRPM(self):
        return self.wheelRPM * Parameters["gearRatio"]

    @property
    def motorRotationsHZ(self):
        return self.wheelRotationsHZ * Parameters["gearRatio"]

    @property
    def maxPower(self):
        return Parameters["tractiveIMax"] * self.voltage

    @property
    def maxWheelTorque(self):
        '''
        maxMotorTorque * gear rato
        '''
        return self.maxMotorTorque * Parameters["gearRatio"]

    @property
    def maxMotorTorque(self):
        '''
        Motor Torque at the wheel
        
        minimum(rpm limited torque, power limited torque, perfect traction torque)
        '''
        ## RPM Limited Torque (Motor Controller limits it to ~ this in practice. Maybe something more like 7490ish)
        if self.motorRPM > 7490:
            return -1 * self.resistiveForces * Parameters["wheelRadius"]
        if self.motorRotationsHZ != 0: ## If rolling, torque may be power limited. 
            maxPowerTorque = self.maxPower / self.motorRotationsHZ * Parameters["gearRatio"]
        else: ## Avoid divide by 0 error but it's just the same as the max torque that the motor can deliver (180 Nm)
            maxPowerTorque = 180.0 # Nm at 0 rpm
        perfectTractionTorque = Parameters["maxTorque"]
        torque = min(perfectTractionTorque, maxPowerTorque, self.maxTractionTorqueAtWheel/Parameters["gearRatio"])
        return torque

    @property
    def voltage(self):
        # return 28.0 * lookup(self.charge, self.lastCurrent)
        return 120.0 # Placeholder voltage. Will be a function of SOC, Temp, and Current Histeresis

    @property
    def power(self):
        # return np.linalg.norm(self.maxMotorTorque) * self.motorRotationsHZ --> Why was this norm?
        return self.maxMotorTorque * self.motorRotationsHZ

    @property
    def current(self):
        if (self.power / self.voltage) > Parameters["tractiveIMax"]:
            return Parameters["tractiveIMax"]
        return self.power / self.voltage

    @property
    def maxTractionTorqueAtWheel(self):
        return Parameters["maxTractionTorque"] * Parameters["wheelRadius"]

    @property
    def motorForce(self):
        return (self.maxWheelTorque / Parameters["wheelRadius"])
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
                self.wheelRPM, self.wheelRotationsHZ,
                self.motorRPM, self.motorRotationsHZ,
                self.charge, self.voltage,
                self.current, self.power,
                self.maxPower,
                self.stepSize,
                self.timeSinceLastSteer
            ]

class SF():
    '''
    Static Method Simulation Functions
    '''
    @staticmethod
    def calculateYawRate(initAcceleration:float, heading:np.ndarray, initYawRate:float, velocity:np.ndarray, steerAngle:float, speed:float, timeSinceLastSteer:float):
        """Calculate the yaw rate of the vehicle at the current state.
        This function computes the yaw rate by calculating tire loads, slip angles,
        cornering stiffness, and then applying the vehicle dynamics equations.
        heading : np.ndarray
            Unit heading vector of the vehicle [x, y] components.
            Initial yaw rate of the vehicle before this time step, in rad/s.
            The velocity vector of the vehicle, in m/s.
            The steering angle of the vehicle, in radians.
            The speed of the vehicle, in m/s.
        Returns
        -------
        float
            The yaw rate of the vehicle, in rad/s.
        Notes
        -----
        Slip ratio is fixed at 0.15.
        """
        tireLoad = getloadTransfer(Parameters, initAcceleration * heading[0], initAcceleration * heading[1], initYawRate)
        slipAngle = calculateSlipAngle(initYawRate, velocity, steerAngle, Parameters)
        slipRatio = 0.15
        corneringStiffness = getCorneringStiffness(tireLoad, slipAngle, slipRatio, speed, 80, 40, Parameters, Magic) # Works but unused
        res = calculateYawRate(initYawRate, speed, steerAngle, timeSinceLastSteer, corneringStiffness[0], corneringStiffness[1], Parameters)
        return res
    
    @staticmethod
    def maxTraction(initAcceleration:float, heading:np.ndarray, initYawRate:float, velocity:np.ndarray, steerAngle:float, speed:float):
        """Calculate the maximum traction available for the vehicle at the current state.
        This function computes the total traction magnitude by calculating tire loads,
        slip angles, and individual tire tractions, then combining them into a resultant
        traction vector.
        heading : np.ndarray
            Unit heading vector of the vehicle [x, y] components.
            Initial yaw rate of the vehicle before this time step, in rad/s.
            The velocity vector of the vehicle, in m/s.
            The steering angle of the vehicle, in radians.
            The speed of the vehicle, in m/s.
        Returns
        -------
        np.ndarray[np.float32]
            The magnitude of the maximum available traction force, in Newtons.
        Notes
        -----
        Yaw velocity is currently set to 0 in tire load calculations.
        Slip ratio is fixed at 0.15.
        """
        tireLoad = getloadTransfer(Parameters, initAcceleration * heading[0], initAcceleration * heading[1], initYawRate) # yaw velocity is currently set to 0

        slipAngle = calculateSlipAngle(initYawRate, velocity, steerAngle, Parameters)
        slipRatio = 0.15
        tireTraction = getTraction(tireLoad, slipAngle, slipRatio, speed, 80, 40, Parameters, Magic)
        longTraction = 0
        latTraction = 0
        for x, y in tireTraction:
            longTraction += x
            latTraction += y
        return np.sqrt(longTraction**2 + latTraction**2)
    
        #tempTire = tire.Tire(500 , 0.15, 0, self.speed, 80, 40, Parameters, Magic)
        #return  ((tempTire.getLongForce()/500 * self.weight * 0.7477)/1.6547084)/(1.0-(0.247718 * tempTire.getLongForce()/500 / 1.6547084))