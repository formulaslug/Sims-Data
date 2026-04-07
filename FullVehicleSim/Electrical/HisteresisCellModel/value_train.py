import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --------------------------------------------------
# Load data
# --------------------------------------------------
df = pl.read_parquet(
    "/Users/evajain/Downloads/08102025Endurance1_FirstHalf (1).parquet"
)

temps = [f"ACC_SEG{i}_TEMPS_CELL{j}" for i in range(5) for j in range(6)]

df = df.with_columns(
    pl.col("ACC_POWER_PACK_VOLTAGE").alias("Voltage"),
    pl.col("ACC_POWER_CURRENT").alias("Current"),
    pl.col("ACC_POWER_SOC").alias("SOC"),
    df.select(temps).mean_horizontal().alias("Temperature"),
)

voltage = df["Voltage"].to_numpy()
current = df["Current"].to_numpy()
soc = df["SOC"].to_numpy()

# --------------------------------------------------
# 🔧 FIX 1: Normalize SOC to [0, 1]
# --------------------------------------------------
soc = np.clip(soc / 100.0, 0.0, 1.0)

# --------------------------------------------------
# 🔧 FIX 2: Remove NaNs / infinities
# --------------------------------------------------
mask = (
    np.isfinite(voltage) &
    np.isfinite(current) &
    np.isfinite(soc)
)

voltage = voltage[mask]
current = current[mask]
soc = soc[mask]

# --------------------------------------------------
# Models
# --------------------------------------------------
def ocv_from_soc(soc, a1, a2, a3, a4):
    exponent = np.clip(-a4 * (1 - soc), -50, 50)  # 🔧 FIX 3
    return a1 + a2 * soc + a3 * np.exp(exponent)


def sag(current, a5, a6, a7):
    return a5 * current + a6 * (np.abs(current) ** a7)

# --------------------------------------------------
# Hysteresis setup
# --------------------------------------------------
dt = 0.01
kernel_duration = 10.0
kernel_size = int(kernel_duration / dt)
t = np.arange(0, kernel_size * dt, dt)

def voltage_model(x, a1, a2, a3, a4, a5, a6, a7, a8, a9):
    soc = x[:, 0]
    current = x[:, 1]

    sigma = a9
    kernel = np.exp(-(t**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)

    prev_curr = np.zeros((len(current), kernel_size))

    for i in range(len(current)):
        start = max(0, i - kernel_size)
        prev = current[start:i]
        if len(prev) > 0:
            prev_curr[i, -len(prev):] = prev

    V_hys = a8 * np.sum(prev_curr * kernel, axis=1)
    V_ocv = ocv_from_soc(soc, a1, a2, a3, a4)
    V_sag = sag(current, a5, a6, a7)

    return V_ocv - V_sag - V_hys

# --------------------------------------------------
# Fit with bounds
# --------------------------------------------------
X = np.column_stack((soc, current))

initial_guess = [3.7, 0.5, 0.1, 8.0, 0.01, 0.002, 1.2, 0.01, 2.0]

bounds = (
    [3.0, 0.0, 0.0, 1.0,  0.0, 0.0, 1.0,  0.0, 0.2],
    [4.5, 2.0, 1.0, 50.0, 0.1, 0.1, 3.0,  0.1, 10.0],
)

params, _ = curve_fit(
    voltage_model,
    X,
    voltage,
    p0=initial_guess,
    bounds=bounds,
    maxfev=40000
)

a1, a2, a3, a4, a5, a6, a7, a8, a9 = params

print("\nLearned parameters:")
print(f"a1={a1:.6f}, a2={a2:.6f}, a3={a3:.6f}, a4={a4:.6f}")
print(f"a5={a5:.6f}, a6={a6:.6f}, a7={a7:.6f}")
print(f"a8={a8:.6f}, a9={a9:.6f}")

# --------------------------------------------------
# Plot (sorted)
# --------------------------------------------------
predicted_voltage = voltage_model(X, *params)
idx = np.argsort(soc)

plt.figure(figsize=(10, 6))
plt.scatter(soc[idx], voltage[idx], s=5, alpha=0.4, label="Measured Voltage")
plt.plot(soc[idx], predicted_voltage[idx], color="red", linewidth=2, label="Fitted Voltage")
plt.xlabel("SOC")
plt.ylabel("Voltage (V)")
plt.legend()
plt.grid(True)
plt.show()
