import matplotlib
matplotlib.use("MacOSX")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io

MAT_PATH = "/Users/evajain/Downloads/Molicel_INR18650P30B_241c01110_simulation.mat"

mat = scipy.io.loadmat(MAT_PATH, squeeze_me=True, struct_as_record=False)
simulation = mat["simulation"]

experiment = simulation.fu.PRO[0]

current_profile = np.array(experiment.I, dtype=float)
meas_cell_voltage = np.array(experiment.V, dtype=float)
time = np.array(experiment.t, dtype=float)

dt = float(np.mean(np.diff(time)))

print("Loaded samples:", len(current_profile))
print("dt:", dt)
print("Experiment name:", experiment.name)



initial_SOC = 1
target_end_SOC = 0

I_dis = np.clip(-current_profile, 0, None)
total_discharge_Ah = np.sum(I_dis) * dt / 3600.0

if total_discharge_Ah == 0:
    required_capacity = 3.0
else:
    required_capacity = total_discharge_Ah / (initial_SOC - target_end_SOC)

print("Total Discharge Ah:", total_discharge_Ah)
print("Required capacity Ah:", required_capacity)

R = 8.31446261815324
F = 96485.33212

V0 = 3.71
C1 = 1.127469
C2 = 6.96148085
C3 = 0.05243311
C4 = 0.01567795

# Tuned: original 0.0016 was ~20x too small for the observed voltage swings
R0 = 0.035
KERNEL_LEN = 200


def ocv_from_soc(soc, T_K=298.15):
    eps = 1e-9
    soc_shift = soc - (0.1 ** 3)
    denom = np.clip(1.0 - soc_shift + C4, eps, None)
    numer = np.clip(C1 * soc_shift + C3, eps, None)
    log_term = np.log(numer / denom)
    return V0 + (C2 * (R * T_K / F) * log_term)


# ============================================================
# SOC TRACKING
# ============================================================

soc = initial_SOC
soc_log = []

for I in current_profile:
    soc += (I * dt) / (3600.0 * required_capacity)
    soc = float(np.clip(soc, 0.0, 1.0))
    soc_log.append(soc)

soc_log = np.array(soc_log, dtype=float)
print("Final SOC:", soc_log[-1])

# ============================================================
# BASE VOLTAGE MODEL
# ============================================================

ocv_log = np.array([ocv_from_soc(s) for s in soc_log], dtype=float)
base_model = ocv_log + (R0 * current_profile)
residual_target = meas_cell_voltage - base_model


N = len(current_profile)

X_kern = np.zeros((N, KERNEL_LEN), dtype=float)
for k in range(KERNEL_LEN):
    if k == 0:
        X_kern[:, k] = current_profile
    else:
        X_kern[k:, k] = current_profile[:-k]

X = np.hstack([np.ones((N, 1)), X_kern])

# Weight by current magnitude; use ALL samples
I_abs = np.abs(current_profile)
max_current = np.max(I_abs) if np.max(I_abs) > 0 else 1.0
weights = 1.0 + 5.0 * (I_abs / max_current)

W = np.sqrt(weights)
Xw = X * W[:, None]
yw = residual_target * W

lam = 1e-3
A = Xw.T @ Xw + lam * np.eye(KERNEL_LEN + 1)
b_vec = Xw.T @ yw

params = np.linalg.solve(A, b_vec)

bias_term = params[0]
learned_kernel = params[1:]

print("Bias term:", bias_term)
print("Learned kernel length:", len(learned_kernel))

# ============================================================
# SAVE KERNEL
# ============================================================

kernel_df = pd.DataFrame({
    "kernel_index": np.arange(len(learned_kernel)),
    "kernel_value": learned_kernel
})
kernel_df.to_csv("trained_voltage_kernel.csv", index=False)
print("Saved trained kernel to trained_voltage_kernel.csv")

# ============================================================
# MODEL CLASS
# ============================================================


class AccumulatorVoltageModel:
    def __init__(self, dt, capacity_Ah, initial_soc, kernel, bias):
        self.dt = float(dt)
        self.capacity_Ah = float(capacity_Ah)
        self.SOC = float(initial_soc)
        self.kernel = np.array(kernel, dtype=float)
        self.bias = float(bias)
        self.I_hist = np.zeros(len(self.kernel), dtype=float)

    def ocv_from_soc(self, soc, T_K=298.15):
        return ocv_from_soc(soc, T_K)

    def step(self, current, T_K=298.15):
        I = float(current)

        self.SOC += (I * self.dt) / (3600.0 * self.capacity_Ah)
        self.SOC = float(np.clip(self.SOC, 0.0, 1.0))

        # Newest current at index 0
        self.I_hist = np.roll(self.I_hist, 1)
        self.I_hist[0] = I

        # kernel[k] * I[t-k], no reversed index needed
        h = float(np.dot(self.kernel, self.I_hist))
        ocv = self.ocv_from_soc(self.SOC, T_K)
        V_cell = ocv + (R0 * I) + self.bias + h

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

# ============================================================
# ERROR METRICS
# ============================================================

rmse = np.sqrt(np.mean((voltage_log - meas_cell_voltage) ** 2))
mae = np.mean(np.abs(voltage_log - meas_cell_voltage))

print(f"RMSE: {rmse:.4f} V")
print(f"MAE:  {mae:.4f} V")


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
