import numpy as np
from paramLoader import Parameters, Magic
from dataclasses import dataclass

# Our libraries
# import yogurt as stepWorld
# from TireModel import dumpling as tire
# from Mech.mechanical import *
from Powertrain.lionCellModel import *
from Mech.aero import calcDrag, calcDownForce
from Mech.braking import calcBrakeForce, calcBrakeHeating, calcBrakeCooling
from Mech.steering import calcSlipAngle, calcYawRate, calcVirtualSlipAngle
from Mech.tireLoad import calcLoadTransfer, calcWeightTransfer
from Mech.traction import calcCorneringStiffness, calcTraction

@dataclass
class VehicleState:
    def __init__(self, position, speed, heading, charge, yawRate, frontBrakeTemperature, rearBrakeTemperature):
        self.position:np.ndarray = position
        self.speed:float = speed
        self.heading:np.ndarray = heading
        self.charge:float = charge
        self.frontBrakeTemperature:float = frontBrakeTemperature
        self.rearBrakeTemperature:float = rearBrakeTemperature
        self.yawRate:float = yawRate

        #self.wheelRPM: np.array = np.asarray([0,0,0,0], dtype=np.float32)
        #self.wheelRotationsHz: float = self.speed / self.WheelCircumference * 2.0 * np.pi
        self.tires:np.ndarray = np.asarray([None, None, None, None])#, dtype=tire.Tire) # [FL, FR, BL, BR]

    @property
    def velocity(self):
        return self.heading * self.speed

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
    def maxTractionTorqueAtWheel(self):
        return Parameters["maxTractionTorque"] * Parameters["wheelRadius"]

class SF():
    '''
    Static Method Simulation Functions
    '''
    @staticmethod
    def calculateYawRate(prevWorld:VehicleState, steerAngle:float, initAcceleration:float, heading:np.ndarray, initYawRate:float, timeSinceLastSteer:float):
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
        tireLoad = calcLoadTransfer(initAcceleration * heading[0], initAcceleration * heading[1], initYawRate)
        slipAngle = calcSlipAngle(initYawRate, prevWorld.velocity, steerAngle, Parameters)
        slipRatio = 0.15
        corneringStiffness = calcCorneringStiffness(tireLoad, slipAngle, slipRatio, prevWorld.speed, 80, 40, Parameters, Magic) # Works but unused
        res = calcYawRate(initYawRate, prevWorld.speed, steerAngle, timeSinceLastSteer, corneringStiffness[0], corneringStiffness[1], Parameters)
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
        np.float32
            The magnitude of the maximum available traction force, in Newtons.
        Notes
        -----
        Yaw velocity is currently set to 0 in tire load calculations.
        Slip ratio is fixed at 0.15.
        """
        tireLoad = calcLoadTransfer(Parameters, initAcceleration * heading[0], initAcceleration * heading[1], initYawRate) # yaw velocity is currently set to 0

        slipAngle = calcSlipAngle(initYawRate, velocity, steerAngle, Parameters)
        slipRatio = 0.15
        tireTraction = calcTraction(tireLoad, slipAngle, slipRatio, speed, 80, 40, Parameters, Magic)
        longTraction = 0
        latTraction = 0
        for x, y in tireTraction:
            longTraction += x
            latTraction += y
        return np.sqrt(longTraction**2 + latTraction**2)
    
        #tempTire = tire.Tire(500 , 0.15, 0, self.speed, 80, 40, Parameters, Magic)
        #return  ((tempTire.getLongForce()/500 * self.weight * 0.7477)/1.6547084)/(1.0-(0.247718 * tempTire.getLongForce()/500 / 1.6547084))

    @staticmethod
    def resistiveForces(worldPrev:VehicleState, brakes:float):
        if worldPrev.speed <= 1e-5: # Floating point error
            return 0
        elif brakes == 0:
            return calcDrag(worldPrev)
        else:
            brakeForce = calcBrakeForce(worldPrev)
            return -1 * (calcDrag(worldPrev) + brakeForce)
        
    @staticmethod
    def maxMotorTorque(worldPrev:VehicleState, resistiveForces:float, maxPower:float, maxTractionTorqueAtWheel:float):
        '''
        Motor Torque at the wheel
        
        minimum(rpm limited torque, power limited torque, perfect traction torque)
        '''
        ## RPM Limited Torque (Motor Controller limits it to ~ this in practice. Maybe something more like 7490ish)
        if worldPrev.motorRPM > 7490:
            return -1 * resistiveForces * Parameters["wheelRadius"]
        if worldPrev.motorRotationsHZ != 0: ## If rolling, torque may be power limited. 
            maxPowerTorque = maxPower / worldPrev.motorRotationsHZ * Parameters["gearRatio"]
        else: ## Avoid divide by 0 error but it's just the same as the max torque that the motor can deliver (180 Nm)
            maxPowerTorque = 180.0 # Nm at 0 rpm
        perfectTractionTorque = Parameters["maxTorque"]
        torque = min(perfectTractionTorque, maxPowerTorque, maxTractionTorqueAtWheel/Parameters["gearRatio"])
        return torque
    
    @staticmethod
    def current(power, voltage):
        if (power / voltage) > Parameters["tractiveIMax"]:
            return Parameters["tractiveIMax"]
        return power / voltage
    
    @staticmethod
    def maxWheelTorque(maxMotorTorque):
        '''
        maxMotorTorque * gear rato
        '''
        return maxMotorTorque * Parameters["gearRatio"]
    
    @staticmethod
    def motorForce(maxWheelTorque):
        return (maxWheelTorque / Parameters["wheelRadius"])
    
    @staticmethod
    def maxPower(voltage):
        return Parameters["tractiveIMax"] * voltage
    @staticmethod
    def voltage():
        # return 28.0 * lookup(self.charge, self.lastCurrent)
        return 120.0 # Placeholder voltage. Will be a function of SOC, Temp, and Current Histeresis
    @staticmethod
    def log(prevWorldArray):
        cols = ["x", "y", "z", "vX", "vY", "vZ", "speed", 
                "headingX", "headingY", "headingZ", 
                "yawRate", "brakeTemperature", 
                "charge", "drag", "resistiveForces", 
                "motorTorque", "motorForce", "netForce", 
                "maxTraction", "wheelRotationsHZ", "motorRPM",
                "motorRotationsHZ", "maxTractionAtWheel", "current", 
                "maxWheelTorque", "maxPower", "power", "voltage",
                "downForce", "brakeForce", "slipAngleFront", "slipAngleRear"]
        for world in prevWorldArray:
            x = world.position[0]
            y = world.position[1]
            z = world.position[2]
            vX = world.velocity[0]
            vY = world.velocity[1]
            vZ = world.velocity[2]
            speed = world.speed
            headingX = world.heading[0]
            headingY = world.heading[1]
            headingZ = world.heading[2]
