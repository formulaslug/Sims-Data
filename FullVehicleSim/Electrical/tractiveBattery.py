from paramLoader import Parameters, Magic
from state import VehicleState
import numpy as np

def calcVoltage(prevWorld: VehicleState) -> float:
    delta = 1 / Parameters["stepsPerSecond"]
    capacity_Ah = Parameters["cellCapacity_Ah"]
    soc = prevWorld.charge

    sigma = Parameters["cellModelSigma"]
    hystGain = Parameters["hysteresisGain"]

    # Sliding window: last 10 seconds of current
    I_hist = prevWorld.current_history

    # Hysteresis kernel
    t = np.arange(prevWorld.current_history.shape[0])
    kernel = np.exp(-(t**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)

    def ocv_from_soc(self, soc):
        return 3.0 + 0.9 * soc + 0.25 * np.exp(-12 * (1 - soc))

    def sag(self, current):
        return 0.02 * current + 0.004 * (current ** 1.3)

    def step(self, current):

        # Update SOC
        self.SOC -= (current * self.dt) / (3600 * self.capacity_Ah)
        self.SOC = np.clip(self.SOC, 0.0, 1.0)

        # -------- Sliding array logic --------
        self.I_hist[:-1] = self.I_hist[1:]   # shift old values
        self.I_hist[-1] = current             # add new current

        # Hysteresis voltage
        V_hyst = self.hyst_gain * np.sum(self.I_hist * self.kernel)

        # Terminal voltage
        voltage = (
            self.ocv_from_soc(self.SOC)
            - self.sag(current) * (1 - self.SOC)
            - V_hyst
        )

        return voltage