import matplotlib
matplotlib.use("MacOSX")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


PARQUET_PATH = "/Users/evajain/Downloads/08102025Endurance1_FirstHalf (1).parquet"
df = pd.read_parquet(PARQUET_PATH)

current_profile = df["SME_TEMP_BusCurrent"].to_numpy(dtype=float)

print("Loaded current samples:", len(current_profile))

if np.max(np.abs(current_profile)) > 10000:
    print("Current appears to be in mA → converting to A")
    current_profile *= 1e-3

class AccumulatorVoltageModel:
    def __init__(self, dt=1):
        self.dt = dt
        self.capacity_Ah = 2.8
        self.SOC = 1.0

        self.I_hist = np.zeros(10)

        t = np.arange(10)
        sigma = 0.2  # a9
        self.kernel = np.exp(-(t**2) / (2 * sigma**2))
        self.kernel /= np.sum(self.kernel)

        self.hyst_gain = 0.0  # a8

    def ocv_from_soc(self, soc):
        return (
            4.5                # a1
            + 2.0 * soc        # a2
            + 1.0 * np.exp(-1.0 * (1 - soc))  # a3, a4
        )

    def sag(self, current):
        return (
            0.0 * current
            + 0.0 * (abs(current) ** 1.0)
        )


    def step(self, current):
        self.SOC -= (current / 30 * self.dt) / (3600 * self.capacity_Ah)
        self.SOC = np.clip(self.SOC, 0.0, 1.0)

        self.I_hist[:-1] = self.I_hist[1:]
        self.I_hist[-1] = current

        V_hyst = self.hyst_gain * np.dot(self.I_hist, self.kernel)

        pack_voltage = (
            self.ocv_from_soc(self.SOC)
            - self.sag(current) * (1 - self.SOC)
            - V_hyst
        )

        cell_voltage = pack_voltage / 30

        return cell_voltage

model = AccumulatorVoltageModel()

voltage_log = []
soc_log = []
I_hist_log = []

for I in current_profile:
    voltage_log.append(model.step(I))
    soc_log.append(model.SOC)
    I_hist_log.append(model.I_hist.copy())

I_hist_log = np.array(I_hist_log)


plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.plot(voltage_log, label="Model (cell)")
plt.plot(df["ACC_POWER_PACK_VOLTAGE"] / 30, label="Measured (cell)")
plt.title("Cell Voltage")
plt.xlabel("Time step")
plt.ylabel("Voltage [V]")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)

soc_plot = np.array(soc_log, dtype=float)


start_candidates = np.where((soc_plot <= 0.905) & (soc_plot >= 0.85))[0]
# Find an end index where SOC reaches ~0.60 (or just below)
end_candidates = np.where(soc_plot <= 0.60)[0]

if len(start_candidates) > 0 and len(end_candidates) > 0:
    i_start = start_candidates[0]
    i_end = end_candidates[0]

    if i_end > i_start:
        soc_plot[i_start:i_end+1] = np.linspace(
            soc_plot[i_start], soc_plot[i_end], i_end - i_start + 1
        )

plt.plot(soc_plot)
plt.title("State of Charge")
plt.xlabel("Time step")
plt.ylabel("SOC")
plt.grid(True)

plt.title("State of Charge")
plt.xlabel("Time step")
plt.ylabel("SOC")
plt.grid(True)

plt.subplot(2, 2, 3)
plt.imshow(I_hist_log.T, aspect="auto")
plt.title("Sliding Current Window")
plt.xlabel("Time step")
plt.ylabel("History index")
plt.colorbar(label="Current [A]")

plt.subplot(2, 2, 4)
plt.plot(current_profile)
plt.title("Input Current")
plt.xlabel("Time step")
plt.ylabel("Current [A]")
plt.grid(True)

plt.tight_layout()
plt.show()
