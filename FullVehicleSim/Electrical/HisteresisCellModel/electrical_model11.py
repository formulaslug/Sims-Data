import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, differential_evolution
from scipy.integrate import cumulative_simpson
from scipy.signal import convolve
import scipy.io
from numba import njit
from scipy.interpolate import LinearNDInterpolator as Lnd_interp
from scipy.interpolate import CubicSpline as cs

# ==========================================
# 1. GLOBAL BASELINE & DATA LOADING
# ==========================================
matData = scipy.io.loadmat("../fs-data/FS-4/P30B Cell Data/Molicel_INR18650P30B_measurement.mat", squeeze_me=True, struct_as_record=False)
measurements = matData["measurement"]

delta = 0.1
kernel_duration = 30
kernel_size = int(kernel_duration / delta)

lowCurrMeasurements = [m for m in measurements.fu.DCC if "-0.1" in m.name]

def func_SOC(I, t, starting_SOC):
    Q_nominal = 3.0  
    integrated_ah = cumulative_simpson(y=I, x=t, initial=0) / 3600.0
    soc = starting_SOC + (integrated_ah / Q_nominal)
    return np.clip(soc, 0.0, 1.0)

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
    for i, sim in enumerate(measurement_set):
        cubic = cs(sim.t, np.array([sim.I, sim.V, sim.T_surf[0, :], sim.T_surf[1, :]]), axis=1)
        t = np.arange(sim.t[0], sim.t[-1], delta)
        I, V, T_surf1, T_surf2 = cubic(t)
        starting_SOC = SOC_lookup(V[0], T_surf1[0])
        SOC = func_SOC(I, t, starting_SOC)
        frames.append(pl.DataFrame({
            "I": I, "V": V, "t": t, 
            "T_surf1": T_surf1, "T_surf2": T_surf2, "SOC": SOC,
            "run": np.ones(len(I)) * i
        }))
    return pl.concat(frames)

all_sims = np.concatenate([measurements.fu.DCC, measurements.fu.CHC, measurements.fu.DCP, measurements.fu.CHP])
aboveFreezingSims = [m for m in all_sims if m.T_surf[0, 0] > 5]
notHighPulse = [m for m in aboveFreezingSims if not ("-40.000C" in m.name or "-30.000C" in m.name)]
fullData = buildDF(notHighPulse)

# ==========================================
# 2. CORE PHYSICS & ECM FUNCTIONS
# ==========================================
ocv_args = np.array([2.33586300e+00, 8.70615991e+00, -3.26789721e+01, 7.63836093e+01, -9.98902031e+01, 6.83036001e+01, -1.90310137e+01, 5.90707089e-04])

def OCV_terminal_voltage(SOC, T, a0, a1, a2, a3, a4, a5, a6, K_T):
    return (a0 + a1 * SOC + a2 * SOC**2 + a3 * SOC**3 + a4 * SOC**4 + a5 * SOC**5 + a6 * SOC**6) * (1 + K_T * (T - 25))

def func_V_h(SOC, M, I, kernel):
    padded_I = np.pad(I, (kernel_size - 1, 0), mode='constant')
    convolution = convolve(padded_I, kernel, mode='valid')
    return M * SOC * convolution

@njit
def func_V_X_fast(tauX, RX, I, delta):
    VX_arr = np.zeros_like(I)
    alpha = np.exp(-delta / tauX)
    beta = RX * (1.0 - alpha)
    for i in range(1, len(I)):
        VX_arr[i] = VX_arr[i - 1] * alpha + I[i] * beta
    return VX_arr

def terminal_voltage(SOC, T, I, R0_discharge, R0_charge, R1, C1, R2, C2, tau_H, M, ocv_params):
    tau1 = R1 * C1
    tau2 = R2 * C2
    
    exponent = -np.arange(0, kernel_size) * delta / tau_H
    kernel = np.exp(exponent) / np.sum(np.exp(exponent))
    
    V_h = func_V_h(SOC, M, I, kernel)
    V_X1 = func_V_X_fast(tau1, R1, I, delta)
    V_X2 = func_V_X_fast(tau2, R2, I, delta)
    
    V_OCV = OCV_terminal_voltage(SOC, T, *ocv_params)
    V_R0 = np.where(I <= 0, R0_discharge * I, R0_charge * I)
    
    return V_OCV - V_R0 + V_X1 + V_X2 + V_h

# ==========================================
# 3. CLEAN WRAPPERS FOR OPTIMIZATION
# ==========================================
def wrapper_post_OCV(X, R0_discharge, R0_charge, R1, C1, R2, C2, tau_H, M):
    modelVoltages = []
    for run_id in np.unique(X[3]):
        mask = X[3] == run_id
        modelVoltage = terminal_voltage(X[0, mask], X[1, mask], X[2, mask], R0_discharge, R0_charge, R1, C1, R2, C2, tau_H, M, ocv_args)
        modelVoltages.append(modelVoltage)
    return np.concatenate(modelVoltages)

def evolution_wrapper(params, X):
    R0_discharge, R0_charge, R1, C1, R2, C2, tau_H, M = params
    v = wrapper_post_OCV(X, R0_discharge, R0_charge, R1, C1, R2, C2, tau_H, M)
    return np.mean((v - X[4]) ** 2)

# ==========================================
# 4. WINDOWS-SAFE EXECUTION GUARANTEE
# ==========================================
if __name__ == '__main__':
    # 1. Filter out data chunks for optimization
    lessData = fullData.filter(pl.col("t") < 1000)
    
    # Pack array structure cleanly
    data_payload = np.array((
        lessData["SOC"].to_numpy(), 
        lessData["T_surf1"].to_numpy(), 
        lessData["I"].to_numpy(), 
        lessData["run"].to_numpy(), 
        lessData["V"].to_numpy()
    ))

    # 2. Set up parameter boundary configurations
    bounds = (
        [1e-7, 1e-7, 1e-7, 1e-3,   1e-7, 10.0,   1e-5, -1], # Lower
        [0.2,  0.2,  0.2,  5000.0, 0.2,  50000.0, 1e4,  1e3] # Upper
    )
    boundsTupled = tuple(zip(*bounds))

    print("Launching Multi-Threaded Differential Evolution Loop...")
    
    # 3. Run Global Optimization Engine Safely
    argsGlobal = differential_evolution(
        evolution_wrapper,
        boundsTupled,
        args=(data_payload,),
        maxiter=35,       # Balance iteration density on dynamic pools
        popsize=10,       # Initial generation seed size
        workers=-1,       # Utilizes all available logical processor cores
        updating='deferred'
    )

    print("\nOptimization Complete!")
    print("Optimal Fit Found Parameters:")
    print(argsGlobal.x)