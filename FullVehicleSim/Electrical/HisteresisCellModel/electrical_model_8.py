import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, differential_evolution
from scipy.interpolate import cubic_spline

import scipy.io
from scipy.integrate import cumulative_simpson
from scipy.signal import convolve
from functools import reduce

matData = scipy.io.loadmat("../fs-data/FS-4/P30B Cell Data/Molicel_INR18650P30B_measurement.mat", squeeze_me=True, struct_as_record=False)
measurements = matData["measurement"]

# (measurements.fu.DCC[1].t[1:]-measurements.fu.DCC[1].t[:-1]).mean()

fullData = pl.DataFrame(schema={
    "I":pl.Float64,
    "V":pl.Float64,
    "t":pl.Float64,
    "T_surf1":pl.Float64,
    "T_surf2":pl.Float64,
    "SOC":pl.Float64,
})

def func_SOC(I, t):
    return np.concatenate([np.array([3.0]), 3.0 - cumulative_simpson(y=I, x=t)/3600.0])

for sim in np.concatenate([measurements.fu.DCC, measurements.fu.CHC, measurements.fu.DCP, measurements.fu.CHP]):
    I = sim.I
    V = sim.V
    t = sim.t
    T_surf = sim.T_surf
    T_surf1 = T_surf[0, :]
    T_surf2 = T_surf[1, :]
    SOC = func_SOC(I, t)
    fullData = fullData.vstack(pl.DataFrame({
        "I": I,
        "V": V,
        "t": t,
        "T_surf1": T_surf1,
        "T_surf2": T_surf2,
        "SOC": SOC,
    }))

delta = 0.01
kernel_duration = 30
kernel_size = int(kernel_duration / delta)

def func_V_h (SOC, M, I, kernel):
    padded_I = np.pad(I, (kernel_size - 1, 0), mode='constant')
    convolution = convolve(padded_I, kernel[::-1], mode='valid')
    return M * SOC * convolution

def func_V_X (tauX, RX, I):
    VX_arr = np.zeros_like(I)
    # reduce(lambda prev, i: VX_arr.__setitem__(i, VX_arr[i - 1] * np.exp(-delta / tauX) + (RX * I[i] * (1 - np.exp(-delta / tauX)))) or VX_arr[i], range(len(I)), None)
    for i in range(len(I)):
        if i == 0:
            VX_arr[i] = 0
        else:
            VX_arr[i] = VX_arr[i - 1] * np.exp(-delta / tauX) + (RX * I[i] * (1 - np.exp(-delta / tauX)))
    return VX_arr

def func_V_OCV (SOC, T, a0, a1, a2, a3, a4, a5, a6, K_T):
    return (a0 + a1 * SOC + a2 * SOC**2 + a3 * SOC**3 + a4 * SOC**4 + a5 * SOC**5 + a6 * SOC**6) * (1 + K_T * (T - 25))

def terminal_voltage(SOC, T, I, R0_discharge, R0_charge, R1, C1, R2, C2, tau_H, M, a0, a1, a2, a3, a4, a5, a6, K_T):
    tau1 = R1 * C1
    tau2 = R2 * C2
    kernal = np.exp(-np.arange(0, kernel_size) * delta / tau_H) / np.sum(np.exp(-np.arange(0, kernel_size) * delta / tau_H))
    V_h = func_V_h(SOC, M, I, kernal)
    V_X1 = func_V_X(tau1, R1, I)
    V_X2 = func_V_X(tau2, R2, I)
    V_OCV = func_V_OCV(SOC, T, a0, a1, a2, a3, a4, a5, a6, K_T)
    V_R0 = np.where(I >= 0, R0_discharge * I, R0_charge * I)
    V_terminal = V_OCV - V_R0 + V_X1 + V_X2 + V_h
    # print(f"V_OCV: {V_OCV[-1]}, V_R0: {V_R0[-1]}, V_X1: {V_X1[-1]}, V_X2: {V_X2[-1]}, V_h: {V_h[-1]}, V_terminal: {V_terminal[-1]}")
    # print(f"Params: R0_discharge: {R0_discharge}, R0_charge: {R0_charge}, R1: {R1}, C1: {C1}, R2: {R2}, C2: {C2}, tau_H: {tau_H}, M: {M}, a0: {a0}, a1: {a1}, a2: {a2}, a3: {a3}, a4: {a4}, a5: {a5}, a6: {a6}, K_T: {K_T}")
    print(f"Params: R0_discharge: {R0_discharge}, R0_charge: {R0_charge}, R1: {R1}, C1: {C1}, R2: {R2}, C2: {C2}, tau_H: {tau_H}, M: {M}")
    if len(V_terminal) == len(fullData["V"].to_numpy()):
        print(f"Mean Squared Error: {np.mean((V_terminal - fullData['V'].to_numpy()) ** 2)}")
    return V_terminal

def wrapper(X, R0_discharge, R0_charge, R1, C1, R2, C2, tau_H, M, a0, a1, a2, a3, a4, a5, a6, K_T):
    SOC = X[0]
    T = X[1]
    I = X[2]
    return terminal_voltage(SOC, T, I, R0_discharge, R0_charge, R1, C1, R2, C2, tau_H, M, a0, a1, a2, a3, a4, a5, a6, K_T)
    
args = curve_fit(wrapper, (fullData["SOC"].to_numpy(), fullData["T_surf1"].to_numpy(), fullData["I"].to_numpy()), fullData["V"].to_numpy(), p0=[0.01, 0.01, 1000, 0.01, 0.01, 1000, 10, 1, 3.7, -0.5, 0.1, -0.01, 0.001, -0.0001, 0.01, 0.01], maxfev=10000)
R0_discharge, R0_charge, R1, C1, R2, C2, tau_H, M, a0, a1, a2, a3, a4, a5, a6, K_T = args[0]

run = measurements.fu.DCC[6]
fig = plt.figure()
plt.plot(run.t, run.V, label="Measured Voltage")
plt.plot(run.t, wrapper((func_SOC(run.I, run.t), run.T_surf[0, :], run.I), *args[0]), label="Fitted Voltage")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.legend()
plt.show()

measurements.fu.DCC[0].name

def buildDF(measurement_set):
    df = pl.DataFrame(schema={
        "I":pl.Float64,
        "V":pl.Float64,
        "t":pl.Float64,
        "T_surf1":pl.Float64,
        "T_surf2":pl.Float64,
        "SOC":pl.Float64,
    })
    for sim in measurement_set:
        I = sim.I
        V = sim.V
        t = sim.t
        T_surf = sim.T_surf
        T_surf1 = T_surf[0, :]
        T_surf2 = T_surf[1, :]
        SOC = func_SOC(I, t)
        df = df.vstack(pl.DataFrame({
            "I": I,
            "V": V,
            "t": t,
            "T_surf1": T_surf1,
            "T_surf2": T_surf2,
            "SOC": SOC,
        }))
    return df

lowCurrMeasurements = [m for m in measurements.fu.DCC if "-0.1" in m.name]
lowCurrentDF = buildDF(lowCurrMeasurements)

def OCV_terminal_voltage(SOC, T, a0, a1, a2, a3, a4, a5, a6, K_T):
    return (a0 + a1 * SOC + a2 * SOC**2 + a3 * SOC**3 + a4 * SOC**4 + a5 * SOC**5 + a6 * SOC**6) * (1 + K_T * (T - 25))


ocv_args = curve_fit(lambda X, a0, a1, a2, a3, a4, a5, a6, K_T: OCV_terminal_voltage(X[0], X[1], a0, a1, a2, a3, a4, a5, a6, K_T), (lowCurrentDF["SOC"].to_numpy(), lowCurrentDF["T_surf1"].to_numpy()), lowCurrentDF["V"].to_numpy(), p0=[3.7, -0.5, 0.1, -0.01, 0.001, -0.0001, 0.01, 0.01], maxfev=10000)
ocv_args[0]

# OCV Args
# array([-1.23630823e+02,  1.87749488e+02, -1.14010230e+02,  3.67068235e+01,
#        -6.61877171e+00,  6.33739870e-01, -2.51814884e-02,  5.90437546e-04])

lowCurrMeasurements

for run in lowCurrMeasurements:
    fig = plt.figure()
    plt.plot(run.t, run.V, label="Measured Voltage")
    plt.plot(run.t, OCV_terminal_voltage(func_SOC(run.I, run.t), run.T_surf[0, :], *ocv_args[0]), label="Fitted OCV Voltage")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.legend()
    plt.show()

# Params: R0_discharge: -1.3235393250096301, 
# R0_charge: 0.15299343818330174, R1: 0.16123612685040048, 
# C1: 0.034272989366562326, R2: -126.42453895480814, C2: 1091372.6474212944, 
# tau_H: 4.7170565122757395, M: 0.0037977786728519702,

def wrapper_post_OCV(X, R0_discharge, R0_charge, R1, C1, R2, C2, tau_H, M):
    SOC = X[0]
    T = X[1]
    I = X[2]
    a0, a1, a2, a3, a4, a5, a6, K_T = ocv_args[0]
    return terminal_voltage(SOC, T, I, R0_discharge, R0_charge, R1, C1, R2, C2, tau_H, M, a0, a1, a2, a3, a4, a5, a6, K_T)

bounds = [
    (0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 1, 0.001), # Lower bounds for R0_discharge, R0_charge, R1, C1, R2, C2, tau_H, M
    (2, 0.5, 1, 1, 1, 1, 100, 1), # Upper bounds for R0_discharge, R0_charge, R1, C1, R2, C2, tau_H, M
]

args2 = curve_fit(wrapper_post_OCV, (fullData["SOC"].to_numpy(), fullData["T_surf1"].to_numpy(), fullData["I"].to_numpy()), fullData["V"].to_numpy(), p0=[0.01, 0.01, 0.16, 0.03, 0.01, 0.01, 10, 0.003], bounds=bounds, maxfev=10000)
