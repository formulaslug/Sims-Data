## When we train from data, use this

from scipy.optimize import curve_fit
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from Data.FSLib.IntegralsAndDerivatives import integrate_with_Scipy_tCol
from Data.FSLib.AnalysisFunctions import simpleTimeCol

# df = pl.read_csv("C:/Projects/FormulaSlug/fs-data/FS-3/voltageTableVTC5A.csv")

temps = []

for i in range(5):
    for j in range(6):
        temps.append(f"ACC_SEG{i}_TEMPS_CELL{j}")

df = pl.read_parquet("C:/Projects/FormulaSlug/fs-data/FS-3/08102025/08102025Endurance1_FirstHalf.parquet")


charge = integrate_with_Scipy_tCol(df["Current"] * -1, simpleTimeCol(df))/3600/30/2.6 # Coulombs --> Ah, 30 cells --> 1 cell, 2.6Ah per cell

df = df.with_columns(
    pl.col("ACC_POWER_PACK_VOLTAGE").alias("Voltage"),
    df.select(temps).mean_horizontal().alias("Temperature"),
    pl.col("SME_TEMP_BusCurrent").alias("Current")
)


# plt.scatter(df["Charge"], df["Voltage"],c=df["Current"], label="Current")
# plt.xlabel("Charge (Ah)")
# plt.legend()
# plt.show()

dt = 0.01
kernel_duration = 10.0
kernel_size = int(kernel_duration / dt)
t = np.arange(0, kernel_size*dt, dt)

def ocv_from_soc(soc, a1, a2, a3, a4):
    return a1 + a2 * soc + a3 * np.exp(-a4 * (1 - soc))

def sag(current, a5, a6, a7):
    return a5 * current + a6 * (current ** a7)

def voltage_model(x, a1, a2, a3, a4, a5, a6, a7, a8, a9):
    charge = x[:,0]
    current = x[:,1]
    hyst_gain = a8
    sigma = a9
    kernel = np.exp(-(t**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    prev_curr = np.zeros((charge.shape[0], kernel_size))
    for i in range(charge.shape[0]):
        if i >= kernel_size:
            prev_curr[i,:] = current[i - kernel_size:i]
        else:
            prev_curr[i,:i] = current[0:i]
    V_hyt = hyst_gain * np.sum(prev_curr * t, axis=1)
    V_ocv = ocv_from_soc(charge / 2.6, a1, a2, a3, a4)
    V_sag = sag(current, a5, a6, a7)
    return V_ocv - V_sag - V_hyt

args = curve_fit(voltage_model, np.column_stack((df["Charge"], df["Current"])), df["Voltage"], p0=[3.0, 0.9, 0.25, 12.0, 0.02, 0.004, 1.3, 0.015, 3.0], maxfev=10000)
args[0]

a1, a2, a3, a4, a5, a6, a7, a8, a9 = args[0]
plt.figure(figsize=(10,6))
plt.scatter(df["Charge"], df["Voltage"], c='blue', label="Measured Voltage", alpha=0.5)
predicted_voltage = voltage_model(np.column_stack((df["Charge"], df["Current"])), a1, a2, a3, a4, a5, a6, a7, a8, a9)
plt.scatter(df["Charge"], predicted_voltage, c='red', label="Fitted Voltage", alpha=0.5)
plt.xlabel("Charge (Ah)")
plt.ylabel("Voltage (V)")
plt.legend()
plt.show()
    
    