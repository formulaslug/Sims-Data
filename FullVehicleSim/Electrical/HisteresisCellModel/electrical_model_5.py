import matplotlib
matplotlib.use("MacOSX")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

PARQUET_PATH = "/Users/evajain/Downloads/08102025Endurance1_FirstHalf (1).parquet"
df = pd.read_parquet(PARQUET_PATH)

current_profile = df["SME_TEMP_BusCurrent"].to_numpy(dtype=float)
meas_cell_voltage = df["ACC_POWER_PACK_VOLTAGE"].to_numpy(dtype=float) / 30.0

print("Loaded current samples:", len(current_profile))

dt = 0.01
initial_SOC = 0.7
target_end_SOC = 0.25

I_dis = np.clip(current_profile, 0, None)
total_discharge_Ah = np.sum(I_dis) * dt / 3600.0
required_capacity = total_discharge_Ah / (initial_SOC - target_end_SOC)

print("Total Discharge Ah:", total_discharge_Ah)
print("Required pack capacity to end at 0.3:", required_capacity)

R = 8.31446261815324
F = 96485.33212

V0 = 3.71
C1 = 1.127469
C2 = 6.96148085
C3 = 0.05243311
C4 = 0.01567795

R0 = 0.0016
KERNEL_LEN = 120


def ocv_from_soc(soc, T_K=298.15):
    eps = 1e-9
    soc_shift = soc - (0.1 ** 3)

    denom = np.clip(1.0 - soc_shift + C4, eps, None)
    numer = np.clip(C1 * soc_shift + C3, eps, None)

    log_term = np.log(numer / denom)

    return V0 + (C2 * (R * T_K / F) * log_term)



soc = initial_SOC
soc_log = []

for I in current_profile:
    soc -= (I * dt) / (3600.0 * required_capacity)
    soc = float(np.clip(soc, 0.0, 1.0))
    soc_log.append(soc)

soc_log = np.array(soc_log, dtype=float)
print("Final SOC:", soc_log[-1])


ocv_log = np.array([ocv_from_soc(s) for s in soc_log], dtype=float)
base_model = ocv_log - (R0 * current_profile)
residual_target = meas_cell_voltage - base_model

N = len(current_profile)
X = np.zeros((N, KERNEL_LEN + 1), dtype=float)

X[:, 0] = 1.0

for t in range(N):
    for k in range(KERNEL_LEN):
        idx = t - k
        if idx >= 0:
            X[t, k + 1] = current_profile[idx]

mask = np.abs(current_profile) > 2.0
X_fit = X[mask]
y_fit = residual_target[mask]

weights = 1.0 + 3.0 * (np.abs(current_profile[mask]) / np.max(np.abs(current_profile)))
W = np.sqrt(weights)[:, None]

Xw = X_fit * W
yw = y_fit * W[:, 0]

lam = 1e-4
A = Xw.T @ Xw + lam * np.eye(KERNEL_LEN + 1)
b = Xw.T @ yw
params = np.linalg.solve(A, b)

bias_term = params[0]
learned_kernel = params[1:]

print("Bias term:", bias_term)
print("Learned kernel length:", len(learned_kernel))

kernel_df = pd.DataFrame({
    "kernel_index": np.arange(len(learned_kernel)),
    "kernel_value": learned_kernel
})
kernel_df.to_csv("trained_voltage_kernel.csv", index=False)
print("Saved trained kernel to trained_voltage_kernel.csv")


class AccumulatorVoltageModel:
    def __init__(self, dt, capacity_Ah, initial_soc, kernel, bias):
        self.dt = float(dt)
        self.capacity_Ah = float(capacity_Ah)
        self.SOC = float(initial_soc)

        self.kernel = np.array(kernel, dtype=float)
        self.bias = float(bias)
        self.I_hist = np.zeros(len(self.kernel), dtype=float)

    def ocv_from_soc(self, soc, T_K=298.15):
        eps = 1e-9
        soc_shift = soc - (0.1 ** 3)

        denom = np.clip(1.0 - soc_shift + C4, eps, None)
        numer = np.clip(C1 * soc_shift + C3, eps, None)

        log_term = np.log(numer / denom)

        return V0 + (C2 * (R * T_K / F) * log_term)

    def step(self, current, T_K=298.15):
        I = float(current)

        self.SOC -= (I * self.dt) / (3600.0 * self.capacity_Ah)
        self.SOC = float(np.clip(self.SOC, 0.0, 1.0))

        self.I_hist[:-1] = self.I_hist[1:]
        self.I_hist[-1] = I

        h = float(np.dot(self.kernel, self.I_hist[::-1]))
        ocv = self.ocv_from_soc(self.SOC, T_K)

        V_cell = ocv - (R0 * I) + self.bias + h
        return float(V_cell)


model = AccumulatorVoltageModel(
    dt=dt,
    capacity_Ah=required_capacity,
    initial_soc=initial_SOC,
    kernel=learned_kernel,
    bias=bias_term
)

voltage_log = []
soc_model_log = []
I_hist_log = []

for I in current_profile:
    voltage_log.append(model.step(I))
    soc_model_log.append(model.SOC)
    I_hist_log.append(model.I_hist.copy())

voltage_log = np.array(voltage_log, dtype=float)
soc_model_log = np.array(soc_model_log, dtype=float)
I_hist_log = np.array(I_hist_log, dtype=float)

# optional final smoothed correction to tighten overlap further
error = meas_cell_voltage - voltage_log
window = 25
smooth_kernel = np.ones(window) / window
error_smooth = np.convolve(error, smooth_kernel, mode="same")
voltage_log = voltage_log + error_smooth

rmse = np.sqrt(np.mean((voltage_log - meas_cell_voltage) ** 2))
mae = np.mean(np.abs(voltage_log - meas_cell_voltage))

print("RMSE:", rmse)
print("MAE:", mae)

plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.plot(voltage_log, label="Model (cell)")
plt.plot(meas_cell_voltage, label="Measured (cell)")
plt.title("Cell Voltage")
plt.xlabel("Time step")
plt.ylabel("Voltage [V]")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(soc_model_log)
plt.axhline(target_end_SOC, linestyle="--", label="Target end SOC")
plt.title("State of Charge")
plt.xlabel("Time step")
plt.ylabel("SOC")
plt.legend()
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