import numpy as np
from ramen import Parameters, Magic

# Our libraries
from MBS.lionCellModel import *
from Mech.aero import calculateDrag
from Mech.braking import *
from Mech.steering import *
from Mech.tireLoad import *
from Mech.traction import *


class VehicleState:
    def __init__(
        self,
        stepSize,
        position: np.ndarray,
        speed: float,
        acceleration: np.ndarray,
        heading,
        charge,
        lastCurrent,
        throttle,
        brakes,
        yawRate,
        steerAngle,
        brakeTemperature,
        timeSinceLastSteer,
        initSpeed,
        current_history=None,
        cell_voltage: float = 4.2,     # cell voltage [V] (NOT pack)
        batt_v_rc: float = 0.0,        # RC polarization voltage [V]
        batt_hyst: float = 0.0,        # hysteresis state
        batt_temp_c: float = 25.0      # temperature [C]
    ):
        # original
        self.stepSize = stepSize
        self.initYawRate = yawRate
        self.steerAngle = steerAngle
        self.brakes = brakes
        self.throttle = throttle
        self.position = position
        self.speed = speed
        self.initAcceleration = acceleration
        self.heading = heading
        self.charge = charge
        self.lastCurrent = lastCurrent

        self.WheelCircumference = Parameters["wheelCircumferance"]
        self.WheelRadius = Parameters["wheelRadius"]
        self.GearRatio = Parameters["gearRatio"]
        self.TorqueMax = Parameters["maxTorque"]
        self.tractiveIMax = Parameters["tractiveIMax"]

        self.brakeTemperature = brakeTemperature
        self.timeSinceLastSteer = timeSinceLastSteer
        self.initSpeed = initSpeed

        # ---- current history buffer (Gaussian uses this) ----
        self.history_steps = int(10.0 / self.stepSize) if self.stepSize > 0 else 10

        if current_history is not None:
            self.current_history = np.asarray(current_history, dtype=float)
            # ensure correct length (optional but helps stability)
            if self.current_history.size != self.history_steps:
                # resize preserving most recent samples
                tmp = np.zeros(self.history_steps, dtype=float)
                n = min(self.history_steps, self.current_history.size)
                tmp[-n:] = self.current_history[-n:]
                self.current_history = tmp
        else:
            self.current_history = np.zeros(self.history_steps, dtype=float)

        # ---- NEW: store battery model states (persist across steps) ----
        self._cell_voltage = float(cell_voltage)
        self.batt_v_rc = float(batt_v_rc)
        self.batt_hyst = float(batt_hyst)
        self.batt_temp_c = float(batt_temp_c)

        # tires container
        self.tires: np.ndarray = np.asarray([None, None, None, None])  # [FL, FR, BL, BR]

    @property
    def yawRate(self):
        tireLoad = getloadTransfer(
            Parameters,
            self.initAcceleration * self.heading[0],
            self.initAcceleration * self.heading[1],
            self.initYawRate
        )
        slipAngle = calculateSlipAngle(self.initYawRate, self.velocity, self.steerAngle, Parameters)
        slipRatio = 0.15
        corneringStiffness = getCorneringStiffness(
            tireLoad, slipAngle, slipRatio, self.speed, 80, 40, Parameters, Magic
        )
        res = calculateYawRate(
            self.initYawRate,
            self.initSpeed,
            self.steerAngle,
            self.timeSinceLastSteer,
            corneringStiffness[0],
            corneringStiffness[1],
            Parameters
        )
        return res

    @property
    def velocity(self):
        return self.heading * self.speed

    @property
    def drag(self):
        return calculateDrag(self.heading, self.speed)

    @property
    def resistiveForces(self):
        if self.speed <= 1e-5:
            return 0
        elif self.brakes == 0:
            return self.drag
        else:
            brakeForce, self.brakeTemperature = getBrakeForce(
                self.speed, self.brakeTemperature, self.stepSize, Parameters
            )
            return -1 * (self.drag + brakeForce)

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
    def voltage(self):
        if hasattr(self, "pack_voltage"):
            return float(self.pack_voltage)
        return 28.0 * float(self._cell_voltage)

    @property
    def maxPower(self):
        return self.tractiveIMax * self.voltage

    @property
    def torque(self):
        return self.motorTorque * self.GearRatio

    @property
    def motorTorque(self):
        if self.rpm > 7500:
            return -1 * self.resistiveForces * self.WheelRadius
        if self.motorRotationsHZ != 0:
            maxPowerTorque = self.maxPower / self.motorRotationsHZ * self.GearRatio
        else:
            maxPowerTorque = 1e6
        perfectTractionTorque = self.TorqueMax * self.throttle
        torque = min(perfectTractionTorque, maxPowerTorque, self.maxTractionTorqueAtWheel / self.GearRatio)
        return torque

    @property
    def power(self):
        return np.linalg.norm(self.motorTorque) * self.motorRotationsHZ

    @property
    def current(self):
        if (self.power / self.voltage) > self.tractiveIMax:
            return self.tractiveIMax
        return self.power / self.voltage

    def update_history(self, value: float):
        """Shift history left and append a new current sample."""
        self.current_history[:-1] = self.current_history[1:]
        self.current_history[-1] = float(value)

    @property
    def maxTraction(self):
        tireLoad = getloadTransfer(
            Parameters,
            self.initAcceleration * self.heading[0],
            self.initAcceleration * self.heading[1],
            self.initYawRate
        )

        slipAngle = calculateSlipAngle(self.initYawRate, self.velocity, self.steerAngle, Parameters)
        slipRatio = 0.15
        tireTraction = getTraction(tireLoad, slipAngle, slipRatio, self.speed, 80, 40, Parameters, Magic)

        longTraction = 0
        latTraction = 0
        for x, y in tireTraction:
            longTraction += x
            latTraction += y
        return np.sqrt(longTraction ** 2 + latTraction ** 2)

    @property
    def maxTractionTorqueAtWheel(self):
        return self.maxTraction * self.WheelRadius

    @property
    def motorForce(self):
        return (self.torque / self.WheelRadius)

    @property
    def netForce(self):
        return self.motorForce + self.resistiveForces

    @property
    def acceleration(self):
        return self.netForce / Parameters["Mass"]

    def logProperties(self):
        return [
            self.position[0], self.position[1],
            self.velocity[0], self.velocity[1],
            self.speed, self.acceleration,
            self.heading[0], self.heading[1],
            self.yawRate,
            self.steerAngle, self.throttle,
            self.brakes,
            self.drag, self.resistiveForces,
            self.motorForce, self.netForce,
            self.torque, self.motorTorque,
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

