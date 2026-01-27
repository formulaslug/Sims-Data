## When we train from data, use this

from scipy.optimize import curve_fit
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

df = pl.read_csv("C:/Projects/FormulaSlug/fs-data/FS-3/voltageTableVTC5A.csv")

plt.scatter(df["Charge"], df["Voltage"],c=df["Current"], label="Current")
plt.xlabel("Charge (Ah)")
plt.legend()
plt.show()

dt = 0.01
kernel_duration = 10.0
kernel_size = int(kernel_duration / dt)
t = np.arange(0, kernel_size*dt, dt)

def ocv_from_soc(soc, a1, a2, a3, a4):
    return a1 + a2 * soc + a3 * np.exp(-a4 * (1 - soc))

def sag(current, a5, a6, a7):
    return a5 * current + a6 * (current ** a7)

def voltage_model(x, a1, a2, a3, a4, a5, a6, a7, a8):
    charge = x[:,0]
    current = x[:,1]
    hyst_gain = a8
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

args = curve_fit(voltage_model, np.column_stack((df["Charge"], df["Current"])), df["Voltage"], p0=[3.0, 0.9, 0.25, 12.0, 0.02, 0.004, 1.3, 0.015])

    