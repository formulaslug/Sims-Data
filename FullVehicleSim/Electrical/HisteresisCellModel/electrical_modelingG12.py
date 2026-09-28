import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_simpson
from scipy.signal import convolve
import scipy.io
from numba import njit
from scipy.interpolate import LinearNDInterpolator as Lnd_interp
from scipy.interpolate import CubicSpline as cs

# 1. Load Data
matData = scipy.io.loadmat("../fs-data/FS-4/P30B Cell Data/Molicel_INR18650P30B_measurement.mat", squeeze_me=True, struct_as_record=False)
measurements = matData["measurement"]

delta = 0.1
kernel_duration = 60
kernel_size = int(kernel_duration / delta)

lowCurrMeasurements = [m for m in measurements.fu.DCC if "-0.1" in m.name]

def func_SOC(I, t, starting_SOC):
    Q_nominal = 3.0  # Ah capacity of Molicel P30B
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
        t = np.arange(sim.t, sim.t[-1], delta)
        I, V, T_surf1, T_surf2 = cubic(t)
        starting_SOC = SOC_lookup(V, T_surf1)
        SOC = func_SOC(I, t, starting_SOC)
        frames.append(pl.DataFrame({
            "I": I, "V": V, "t": t, 
            "T_surf1": T_surf1, "T_surf2": T_surf2, "SOC": SOC,
            "run": np.ones(len(I)) * i
        }))
    return pl.concat(frames)

all_sims = np.concatenate([measurements.fu.DCC, measurements.fu.CHC, measurements.fu.DCP, measurements.fu.CHP])
aboveFreezingSims = [m for m in all_sims if m.T_surf > -2]
notHighPulse = [m for m in aboveFreezingSims if not ("-40.000C" in m.name or "-30.000C" in m.name)]
fullData = buildDF(notHighPulse)

# 2. Extract and Freeze OCV
def OCV_terminal_voltage(SOC, T, a0, a1, a2, a3, a4, a5, a6, K_T):
    return (a0 + a1 * SOC + a2 * SOC**2 + a3 * SOC**3 + a4 * SOC**4 + a5 * SOC**5 + a6 * SOC**6) * (1 + K_T * (T - 25.0))

ocv_args = np.array([2.33586300e+00, 8.70615991e+00, -3.26789721e+01, 7.63836093e+01, 
                     -9.98902031e+01, 6.83036001e+01, -1.90310137e+01, 5.90707089e-04])

# 3. Enhanced Hysteresis Tracking
def func_V_h(SOC, M, I, kernel):
    current_direction = np.sign(I)
    padded_I = np.pad(current_direction, (kernel_size - 1, 0), mode='constant')
    convolution = convolve(padded_I, kernel, mode='valid')
    return M * convolution

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
    
    return V_OCV + V_R0 + V_X1 + V_X2 + V_h

# 4. Scaled Optimization Wrapper
def wrapper_post_OCV(X, R0_d_scaled, R0_c_scaled, R1_scaled, C1_scaled, R2_scaled, C2_scaled, tau_H, M_scaled):
    # Scale variables back up to their physical magnitudes
    R0_discharge = R0_d_scaled * 1e-3
    R0_charge    = R0_c_scaled * 1e-3
    R1           = R1_scaled * 1e-3
    C1           = C1_scaled * 1e2
    R2           = R2_scaled * 1e-3
    C2           = C2_scaled * 1e4
    M            = M_scaled * 1e-2
    
    modelVoltages = []
    run_ids = X[3, :] 
    
    for id in np.unique(run_ids):
        mask = run_ids == id
        modelVoltage = terminal_voltage(
            X[0, mask], X[1, mask], X[2, mask], 
            R0_discharge, R0_charge, R1, C1, R2, C2, tau_H, M, ocv_args
        )
        modelVoltages.append(modelVoltage)
    
    actual_voltages = X[4, :]
    current_mae = np.mean(np.abs(np.concatenate(modelVoltages) - actual_voltages))
    print(f"Iteration MAE: {current_mae:.5f} V")
    
    return np.concatenate(modelVoltages)

# 5. Scaled Parameter Boundaries & Guess Configuration
# Enforces strict parameter separation: 
# Tau1 (R1*C1) bounds: 0.1s to 8s (Fast Dynamics)
# Tau2 (R2*C2) bounds: 15s to 300s (Slow Diffusion)

# Parameters mapped: [R0_d, R0_c, R1, C1, R2, C2, tau_H, M] (all values normalized around ~1-10)
scaled_bounds = (
    [2.0,  2.0,  0.5,  1.0,  1.0,  0.5,  0.5,  0.01],  # Lower Bounds
    [40.0, 40.0, 20.0, 10.0, 30.0,  5.0,  120.0, 5.00]   # Upper Bounds
)

p0_scaled = [12.0, 15.0, 6.0, 4.0, 12.0, 1.5, 15.0, 0.8]

# Downsample the whole time profile using polars instead of throwing away tail data
lessData = fullData.gather_every(10)

X_data = np.array((
    lessData["SOC"].to_numpy(), 
    lessData["T_surf1"].to_numpy(), 
    lessData["I"].to_numpy(), 
    lessData["run"].to_numpy(), 
    lessData["V"].to_numpy()
))

print(f"Starting dynamic tuning sequence with {X_data.shape} data points...")
optimized_scaled_args, _ = curve_fit(
    wrapper_post_OCV, 
    X_data, 
    lessData["V"].to_numpy(), 
    p0=p0_scaled, 
    bounds=scaled_bounds, 
    maxfev=3000,
    method='trf',
    x_scale='jac'
)

# Unpack and rescale final optimized values back to true SI units
final_params = {
    "R0_discharge": optimized_scaled_args * 1e-3,
    "R0_charge":    optimized_scaled_args * 1e-3,
    "R1":           optimized_scaled_args * 1e-3,
    "C1":           optimized_scaled_args * 1e2,
    "R2":           optimized_scaled_args * 1e-3,
    "C2":           optimized_scaled_args * 1e4,
    "tau_H":        optimized_scaled_args,
    "M":            optimized_scaled_args * 1e-2
}

print("\nOptimized Dynamic Parameters (Rescaled to Physical Units):")
for k, v in final_params.items():
    print(f"{k}: {v:.6e}")