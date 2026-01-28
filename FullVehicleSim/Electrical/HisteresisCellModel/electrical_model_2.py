import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# Accumulator Voltage Model (Stateful + Hysteresis)
# =====================================================

class AccumulatorVoltageModel:
    def __init__(self, dt=1.0):
        self.dt = dt                      # time step [s]
        self.capacity_Ah = 2.6            # cell capacity [Ah]
        self.SOC = 1.0                    # initial SOC

        # 10-second current history buffer
        self.I_hist = np.zeros(10)

        # Gaussian kernel for hysteresis
        t = np.arange(10)
        sigma = 3.0
        self.kernel = np.exp(-(t**2) / (2 * sigma**2))
        self.kernel /= np.sum(self.kernel)

        self.hyst_gain = 0.015             # hysteresis strength

    # -------------------------------------------------
    # Open Circuit Voltage from SOC (datasheet-like)
    # -------------------------------------------------
    def ocv_from_soc(self, soc):
        return 3.0 + 0.9*soc + 0.25*np.exp(-12*(1 - soc))


    def sag(self, current):
        return 0.02*current + 0.004*(current**1.3)

  
    def step(self, current):
        # Update SOC
        self.SOC -= (current * self.dt) / (3600 * self.capacity_Ah)
        self.SOC = np.clip(self.SOC, 0.0, 1.0)

        # Slide current history window
        self.I_hist[:-1] = self.I_hist[1:]
        self.I_hist[-1] = current

        # Hysteresis voltage term
        V_hyst = self.hyst_gain * np.sum(self.I_hist * self.kernel)

        # Voltage calculation
        V = (
            self.ocv_from_soc(self.SOC)
            - self.sag(current) * (1 - self.SOC)
            - V_hyst
        )

        return V


model = AccumulatorVoltageModel(dt=1.0)

time = []
voltage = []
soc = []
current_log = []

current_profile = (
    [5]*20 +     # cruise
    [20]*10 +    # acceleration
    [10]*20 +    # steady drive
    [0]*10       # regen / idle
)

for t, I in enumerate(current_profile):
    V = model.step(I)
    time.append(t)
    voltage.append(V)
    soc.append(model.SOC)
    current_log.append(I)

# =====================================================
# Plots
# =====================================================

plt.figure(figsize=(14,10))

# Voltage vs Time
plt.subplot(2,2,1)
plt.plot(time, voltage)
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.title("Accumulator Voltage vs Time")
plt.grid(True)

# SOC vs Time
plt.subplot(2,2,2)
plt.plot(time, soc)
plt.xlabel("Time [s]")
plt.ylabel("SOC")
plt.title("State of Charge vs Time")
plt.grid(True)

# Current Profile
plt.subplot(2,2,3)
plt.plot(time, current_log)
plt.xlabel("Time [s]")
plt.ylabel("Current [A]")
plt.title("Vehicle Current Profile")
plt.grid(True)

# Voltage vs SOC
plt.subplot(2,2,4)
plt.plot(soc, voltage)
plt.xlabel("SOC")
plt.ylabel("Voltage [V]")
plt.title("Voltage vs SOC")
plt.grid(True)

plt.tight_layout()
plt.show()

# =====================================================
# Hysteresis Demonstration
# =====================================================

model = AccumulatorVoltageModel()

# High-current history
for _ in range(10):
    model.step(20)
V_high_hist = model.step(5)

# Low-current history
model = AccumulatorVoltageModel()
for _ in range(10):
    model.step(0)
V_low_hist = model.step(5)

print("Voltage with high-current history:", round(V_high_hist, 3), "V")
print("Voltage with low-current history :", round(V_low_hist, 3), "V")
