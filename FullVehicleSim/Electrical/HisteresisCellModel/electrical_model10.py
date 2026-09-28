import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.integrate import cumulative_simpson
from scipy.signal import convolve
from scipy.interpolate import CubicSpline as cs
from scipy.interpolate import LinearNDInterpolator as Lnd_interp
from scipy.optimize import curve_fit
from numba import njit

# 1. Load Data
matData = scipy.io.loadmat("../fs-data/FS-4/P30B Cell Data/Molicel_INR18650P30B_measurement.mat", squeeze_me=True, struct_as_record=False)
measurements = matData["measurement"]

DELTA = 0.1  # Stick to 0.1s to capture high-rate dynamic transients accurately
KERNEL_DURATION = 30
KERNEL_SIZE = int(KERNEL_DURATION / DELTA)
Q_NOMINAL = 3.0  

def func_SOC(I, t, starting_SOC):
    integrated_ah = cumulative_simpson(y=I, x=t, initial=0) / 3600.0
    soc = starting_SOC - (integrated_ah / Q_NOMINAL)  # Discharge decreases SOC
    return np.clip(soc, 0.0, 1.0)

lowCurrMeasurements = [m for m in measurements.fu.DCC if "-0.1" in m.name]
lowVs = np.concatenate([sim.V for sim in lowCurrMeasurements])
lowTs = np.concatenate([sim.T_surf[0, :] for sim in lowCurrMeasurements])
lowSOC = np.concatenate([func_SOC(sim.I, sim.t, 1.0) for sim in lowCurrMeasurements])

voltage_curve = Lnd_interp(np.array([lowVs, lowTs]).T, lowSOC, rescale=True)

def SOC_lookup(V, T):
    SOC = voltage_curve(V, T)
    if np.isnan(SOC):
        if V > 4.1: return 1.0
        if T < -15.0: return voltage_curve(V, -15.0)
        if T > 39.0: return voltage_curve(V, 39.0)
    return np.clip(SOC, 0.0, 1.0)

def buildDF(measurement_set):
    frames = []
    for sim in measurement_set:
        cubic = cs(sim.t, np.array([sim.I, sim.V, sim.T_surf[0, :], sim.T_surf[1, :]]), axis=1)
        t = np.arange(sim.t[0], sim.t[-1], DELTA)
        I, V, T_surf1, T_surf2 = cubic(t)
        starting_SOC = SOC_lookup(V[0], T_surf1[0])
        SOC = func_SOC(I, t, starting_SOC)
        frames.append(pl.DataFrame({
            "I": I, "V": V, "t": t, 
            "T_surf1": T_surf1, "T_surf2": T_surf2, "SOC": SOC,
        }))
    return pl.concat(frames)

all_sims = np.concatenate([measurements.fu.DCC, measurements.fu.CHC, measurements.fu.DCP, measurements.fu.CHP])
fullData = buildDF(all_sims)

# OCV Arguments
ocv_args = np.array([2.33586300e+00, 8.70615991e+00, -3.26789721e+01, 7.63836093e+01, -9.98902031e+01, 6.83036001e+01, -1.90310137e+01, 5.90707089e-04])

def OCV_terminal_voltage(SOC, T, a0, a1, a2, a3, a4, a5, a6, K_T):
    return (a0 + a1 * SOC + a2 * SOC**2 + a3 * SOC**3 + a4 * SOC**4 + a5 * SOC**5 + a6 * SOC**6) * (1 + K_T * (T - 25))

@njit
def func_V_X_fast_dynamic(tauX, RX_arr, I, delta):
    VX_arr = np.zeros_like(I)
    if tauX <= 1e-5: return VX_arr
    alpha = np.exp(-delta / tauX)
    for i in range(1, len(I)):
        beta = RX_arr[i] * (1.0 - alpha)
        VX_arr[i] = VX_arr[i - 1] * alpha + I[i] * beta
    return VX_arr

def func_V_h(SOC, M, I, kernel):
    # Fix: Hysteresis tracks current DIRECTION (sign), not raw multi-ampere values
    I_sign = np.sign(I)
    padded_I = np.pad(I_sign, (KERNEL_SIZE - 1, 0), mode='constant')
    convolution = convolve(padded_I, kernel[::-1], mode='valid')
    return M * SOC * convolution

def terminal_voltage_dynamic(SOC, T, I, 
                             r0_d0, r0_d1, r0_d2,  
                             r0_c0, r0_c1, r0_c2,  
                             r1_0, r1_1,           
                             k_r, C1, R2, C2, tau_H, M, ocv_params):
    
    temp_factor = (1.0 + k_r * (T - 25.0))
    
    R0_d_stream = (r0_d0 + r0_d1 * SOC + r0_d2 / (SOC + 0.01)) * temp_factor
    R0_c_stream = (r0_c0 + r0_c1 * SOC + r0_c2 / (SOC + 0.01)) * temp_factor
    R1_stream = (r1_0 + r1_1 * (1.0 - SOC)) * temp_factor
    
    tau1 = np.mean(R1_stream) * C1  
    tau2 = R2 * C2
    
    exponent = -np.arange(0, KERNEL_SIZE) * DELTA / tau_H
    kernel = np.exp(exponent) / np.sum(np.exp(exponent))
    
    V_h = func_V_h(SOC, M, I, kernel)
    V_X1 = func_V_X_fast_dynamic(tau1, R1_stream, I, DELTA)
    V_X2 = func_V_X_fast_dynamic(tau2, np.full_like(I, R2), I, DELTA)
    
    V_OCV = OCV_terminal_voltage(SOC, T, *ocv_params)
    V_R0 = np.where(I >= 0, R0_d_stream * I, R0_c_stream * I)
    
    # FIX: Subtract transient polarization voltages (V_X1, V_X2) from V_OCV
    V_est = V_OCV - V_R0 - V_X1 - V_X2 + V_h
    
    print(f"Current Loop Step MAE: {np.mean(np.abs(V_est - fullData['V'].to_numpy())) * 1000.0:.2f} mV")
    return V_est

def wrapper_dynamic(X, r0_d0, r0_d1, r0_d2, r0_c0, r0_c1, r0_c2, r1_0, r1_1, k_r, C1, R2, C2, tau_H, M):
    return terminal_voltage_dynamic(X[0], X[1], X[2], r0_d0, r0_d1, r0_d2, r0_c0, r0_c1, r0_c2, r1_0, r1_1, k_r, C1, R2, C2, tau_H, M, ocv_args)

# Strictly enforce physical boundaries to prevent optimization divergence
# Layout: [r0_d0, r0_d1, r0_d2, r0_c0, r0_c1, r0_c2, r1_0, r1_1, k_r, C1, R2, C2, tau_H, M]
low_b = [1e-4, -0.05, 1e-5, 1e-4, -0.05, 1e-5, 1e-4, 1e-4, -0.05, 10.0, 1e-4, 100.0, 0.5, 1e-4]
upr_b = [0.1,   0.05,  0.01, 0.1,   0.05,  0.01, 0.05, 0.05,  0.0,   5000, 0.1,  50000, 70.0, 0.1]

p0_dynamic = [0.012, 0.0, 0.0005, 0.010, 0.0, 0.0005, 0.004, 0.002, -0.01, 200, 0.008, 4000, 25, 0.015]

print("Starting bounded parameter estimation...")
args_dynamic, _ = curve_fit(
    wrapper_dynamic,
    (fullData["SOC"].to_numpy(), fullData["T_surf1"].to_numpy(), fullData["I"].to_numpy()),
    fullData["V"].to_numpy(),
    p0=p0_dynamic,
    bounds=(low_b, upr_b),  # Crucial bounds restored
    maxfev=5000
)

print("\nFinal Optimized Parameters Array:")
print(args_dynamic)