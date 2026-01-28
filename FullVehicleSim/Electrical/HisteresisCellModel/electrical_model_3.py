import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# Accumulator Voltage Model
# =====================================================
class AccumulatorVoltageModel:
    def __init__(self, dt=1.0):
        self.dt = dt
        self.capacity_Ah = 2.6
        self.SOC = 1.0

        # Sliding window: last 10 seconds of current
        self.I_hist = np.zeros(10)

        # Hysteresis kernel
        t = np.arange(10)
        sigma = 3.0
        self.kernel = np.exp(-(t**2) / (2 * sigma**2))
        self.kernel /= np.sum(self.kernel)

        self.hyst_gain = 0.015

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


# =====================================================
# Vehicle Current Template
# (Can be replaced with vehicle state logic)
# =====================================================
current_profile = (
    [5]*10 +     # cruise
    [20]*10 +    # acceleration
    [10]*10 +    # steady
    [0]*10       # idle / regen
)

# =====================================================
# Simulation
# =====================================================
model = AccumulatorVoltageModel()

voltage_log = []
soc_log = []
I_hist_log = []

for I in current_profile:
    V = model.step(I)
    voltage_log.append(V)
    soc_log.append(model.SOC)
    I_hist_log.append(model.I_hist.copy())

I_hist_log = np.array(I_hist_log)

# =====================================================
# Plots
# =====================================================
plt.figure(figsize=(14,10))

# Voltage
plt.subplot(2,2,1)
plt.plot(voltage_log)
plt.title("Accumulator Voltage")
plt.xlabel("Time step")
plt.ylabel("Voltage [V]")
plt.grid(True)

# SOC
plt.subplot(2,2,2)
plt.plot(soc_log)
plt.title("State of Charge")
plt.xlabel("Time step")
plt.ylabel("SOC")
plt.grid(True)

# Sliding current window
plt.subplot(2,2,3)
plt.imshow(I_hist_log.T, aspect='auto')
plt.title("Sliding 10-Second Current Window")
plt.xlabel("Time step")
plt.ylabel("History index (old → new)")
plt.colorbar(label="Current [A]")

# Input current
plt.subplot(2,2,4)
plt.plot(current_profile)
plt.title("Vehicle Current Input")
plt.xlabel("Time step")
plt.ylabel("Current [A]")
plt.grid(True)

plt.tight_layout()
plt.show()
