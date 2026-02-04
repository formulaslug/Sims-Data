import numpy as np
from paramLoader import Parameters
from dataclasses import dataclass

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
