import numpy as np
from paramLoader import Parameters, Magic
from dataclasses import dataclass

@dataclass
class VehicleState:
    def __init__(self, position, speed, heading, charge, yawRate, frontBrakeTemperature, rearBrakeTemperature, tractiveBatteryTemperature):
        self.position:np.ndarray = position
        self.speed:float = speed
        self.heading:np.ndarray = heading
        self.charge:float = charge
        self.frontBrakeTemperature:float = frontBrakeTemperature
        self.rearBrakeTemperature:float = rearBrakeTemperature
        self.yawRate:float = yawRate
        self.tractiveBatteryTemperature:float = tractiveBatteryTemperature

        # ---- current history buffer (Gaussian uses this) ----
        self.history_steps = int(10.0 * Parameters["stepsPerSecond"])

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
        # self._cell_voltage = float(cell_voltage)
        # self.batt_v_rc = float(batt_v_rc)
        # self.batt_hyst = float(batt_hyst)
        # self.batt_temp_c = float(batt_temp_c)

        # tires container
        self.tires: np.ndarray = np.asarray([None, None, None, None])  # [FL, FR, BL, BR]

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
