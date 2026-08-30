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
from functools import partial


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
        if V > 4.1:
            return 1.0
        if T < -15.0:
            return voltage_curve(V, -15.0)
        if T > 39.0:
            return voltage_curve(V, 39.0)
    return np.clip(SOC, 0.0, 1.0)
# Fixed SOC calculation tracking 0.0 to 1.0 bounds safely

# plt.plot(lowCurrMeasurements[0].t, func_SOC(lowCurrMeasurements[0].I, lowCurrMeasurements[0].t, SOC_lookup(lowCurrMeasurements[0].V[0], lowCurrMeasurements[0].T_surf[0, 0])))
# lowCurrMeasurements[0].V[0], lowCurrMeasurements[0].T_surf[0, 0]

## Validation of SOC!!
# X, Y = np.meshgrid(np.arange(2.5, 4.2, 0.01), np.arange(-20, 45, 1))
# Z = SOC_lookup(X, Y)
# plt.contourf(X, Y, Z, levels=50, cmap='viridis')
# plt.colorbar(label='SOC')
# plt.xlabel('Voltage (V)')
# plt.ylabel('Temperature (°C)')
# plt.title('SOC Lookup Table')
# plt.show()

# measurements.fu.DCC[1].V[0]
# measurements.fu.DCC[1].T_surf[0, 0]
# SOC_lookup(4.169, -19.37)

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

# lowCurrentDF = buildDF(lowCurrMeasurements)
# Gather Datasets



# Use everything for dynamic fitting
all_sims = np.concatenate([measurements.fu.DCC, measurements.fu.CHC, measurements.fu.DCP, measurements.fu.CHP])
# for sim in all_sims:
#     startingSOC = SOC_lookup(sim.V[0], sim.T_surf[0, 0])
#     print(f"Run: {sim.name}, Starting SOC: {startingSOC:.4f}")

aboveFreezingSims = [m for m in all_sims if m.T_surf[0, 0] > 5]
notHighPulse = [m for m in aboveFreezingSims if not ("-40.000C" in m.name or "-30.000C" in m.name)]
# for sim in notHighPulse:
    # print(sim.name)
fullData = buildDF(notHighPulse)

# 2. Extract OCV
def OCV_terminal_voltage(SOC, T, a0, a1, a2, a3, a4, a5, a6, K_T):
    return (a0 + a1 * SOC + a2 * SOC**2 + a3 * SOC**3 + a4 * SOC**4 + a5 * SOC**5 + a6 * SOC**6) * (1 + K_T * (T - 25))

# OCV Args
# array([ 2.33727919e+00,  8.66472120e+00, -3.23286606e+01,  7.50959382e+01,
#        -9.75689762e+01,  6.62872924e+01, -1.83568111e+01,  5.90434471e-04])

# ocv_args = np.array([ 2.33727919e+00,  8.66472120e+00, -3.23286606e+01,  7.50959382e+01, -9.75689762e+01,  6.62872924e+01, -1.83568111e+01,  5.90434471e-04])
ocv_args = np.array([ 2.33586300e+00,  8.70615991e+00, -3.26789721e+01,  7.63836093e+01, -9.98902031e+01,  6.83036001e+01, -1.90310137e+01,  5.90707089e-04])

# ocv_args, _ = curve_fit(
#     lambda X, a0, a1, a2, a3, a4, a5, a6, K_T: OCV_terminal_voltage(X[0], X[1], a0, a1, a2, a3, a4, a5, a6, K_T),
#     (lowCurrentDF["SOC"].to_numpy(), lowCurrentDF["T_surf1"].to_numpy()), 
#     lowCurrentDF["V"].to_numpy(), 
#     p0=[3.7, 0.5, -0.1, 0.01, -0.001, 0.0, 0.0, 0.001], 
#     maxfev=20000
# )

# 3. Dynamic Optimization Functions (Accelerated)
# @njit
def func_V_h(SOC, M, I, kernel):
    padded_I = np.pad(I, (kernel_size - 1, 0), mode='constant')
    convolution = convolve(padded_I, kernel, mode='valid')
    # return M * convolution
    return M * SOC * convolution


@njit
def func_V_X_fast(tauX, RX, I, delta):
    VX_arr = np.zeros_like(I)
    alpha = np.exp(-delta / tauX)
    beta = RX * (1.0 - alpha)
    for i in range(1, len(I)):
        VX_arr[i] = VX_arr[i - 1] * alpha + I[i] * beta
    return VX_arr

example_tests = ["DCC / 25°C / -10.000C", "CHC / 25°C / 3.000C", "DCP / 25°C / -30.000C", "CHP / 40°C / 4.000C"]
example_runs = [m for m in measurements.fu.DCC if m.name in example_tests] + \
                [m for m in measurements.fu.CHC if m.name in example_tests] + \
                [m for m in measurements.fu.DCP if m.name in example_tests] + \
                [m for m in measurements.fu.CHP if m.name in example_tests]

# DCC / 25°C / -10.000C
# CHC / 25°C / 3.000C
# DCP / 25°C / -30.000C
# CHP / 40°C / 4.000C

def terminal_voltage(SOC, T, I, R0_discharge, R0_charge, R1, C1, R2, C2, tau_H, M, ocv_params, piece=False, verbose=False, superVerbose=True):
    tau1 = R1 * C1
    tau2 = R2 * C2
    
    # Kernel calculation
    exponent = -np.arange(0, kernel_size) * delta / tau_H
    kernel = np.exp(exponent) / np.sum(np.exp(exponent))
    
    V_h = func_V_h(SOC, M, I, kernel)
    V_X1 = func_V_X_fast(tau1, R1, I, delta)
    V_X2 = func_V_X_fast(tau2, R2, I, delta)
    
    V_OCV = OCV_terminal_voltage(SOC, T, *ocv_params)
    V_R0 = np.where(I <= 0, R0_discharge * I, R0_charge * I)
    # if superVerbose:
        # print(f"R0_discharge: {R0_discharge:.5f}, R0_charge: {R0_charge:.5f}, R1: {R1:.5f}, C1: {C1:.5f}, R2: {R2:.5f}, C2: {C2:.5f}, tau_H: {tau_H:.5f}, M: {M:.5f}")
    if not piece:
        print(f"MAE: {np.mean(np.abs(V_OCV + V_R0 + V_X1 + V_X2 + V_h - fullData['V'].to_numpy()))}")
    # print(f"MSE: {np.mean((V_OCV + V_R0 + V_X1 + V_X2 + V_h - fullData['V'].to_numpy()) ** 2)}")
    if verbose:
        print(f"V_OCV: {V_OCV[-1]}, V_R0: {V_R0[-1]}, V_X1: {V_X1[-1]}, V_X2: {V_X2[-1]}, V_h: {V_h[-1]}, V_terminal: {V_OCV[-1] + V_R0[-1] + V_X1[-1] + V_X2[-1] + V_h[-1]}")
    
    return V_OCV + V_R0 + V_X1 + V_X2 + V_h

def debug_terminal_voltage(SOC, T, I, R0_discharge, R0_charge, R1, C1, R2, C2, tau_H, M, ocv_params, piece=False, verbose=False, superVerbose=True):
    tau1 = R1 * C1
    tau2 = R2 * C2
    
    # Kernel calculation
    exponent = -np.arange(0, kernel_size) * delta / tau_H
    kernel = np.exp(exponent) / np.sum(np.exp(exponent))
    
    V_h = func_V_h(SOC, M, I, kernel)
    V_X1 = func_V_X_fast(tau1, R1, I, delta)
    V_X2 = func_V_X_fast(tau2, R2, I, delta)
    
    V_OCV = OCV_terminal_voltage(SOC, T, *ocv_params)
    V_R0 = np.where(I <= 0, R0_discharge * I, R0_charge * I)
    if superVerbose:
        print(f"R0_discharge: {R0_discharge:.5f}, R0_charge: {R0_charge:.5f}, R1: {R1:.5f}, C1: {C1:.5f}, R2: {R2:.5f}, C2: {C2:.5f}, tau_H: {tau_H:.5f}, M: {M:.5f}")
    if not piece:
        print(f"MAE: {np.mean(np.abs(V_OCV + V_R0 + V_X1 + V_X2 + V_h - fullData['V'].to_numpy()))}")
    # print(f"MSE: {np.mean((V_OCV + V_R0 + V_X1 + V_X2 + V_h - fullData['V'].to_numpy()) ** 2)}")
    if verbose:
        print(f"V_OCV: {V_OCV[-1]}, V_R0: {V_R0[-1]}, V_X1: {V_X1[-1]}, V_X2: {V_X2[-1]}, V_h: {V_h[-1]}, V_terminal: {V_OCV[-1] + V_R0[-1] + V_X1[-1] + V_X2[-1] + V_h[-1]}")
    
    return V_OCV, V_R0, V_X1, V_X2, V_h


# fig = plt.figure()
# axes = fig.subplots(2, 2)
# fig.show()

def wrapper_post_OCV(X, R0_discharge, R0_charge, R1, C1, R2, C2, tau_H, M):
    # print(X.shape)
    # for ax, run in zip(axes.flatten(), example_runs):
    #     run_soc = func_SOC(run.I, run.t, SOC_lookup(run.V[0], run.T_surf[0, 0]))
    #     fitted_v = terminal_voltage(run_soc, run.T_surf[0, :], run.I, R0_discharge, R0_charge, R1, C1, R2, C2, tau_H, M, ocv_args, piece=True)
    #     ax.clear()
    #     ax.plot(run.t, run.V, label="Measured Voltage", alpha=0.8)
    #     ax.plot(run.t, fitted_v, '--', label="Fitted Performance ECM", alpha=0.9)
    #     ax.set_xlabel("Time (s)")
    #     ax.set_ylabel("Voltage (V)")
    #     ax.set_title(f"Run: {run.name}")
    #     ax.legend()
    #     ax.grid(True)
    # fig.canvas.draw()
    # fig.canvas.flush_events()
    modelVoltages = []
    for id in np.unique(X[3]):
        mask = X[3] == id
        modelVoltage = terminal_voltage(X[0, mask], X[1, mask], X[2, mask], R0_discharge, R0_charge, R1, C1, R2, C2, tau_H, M, ocv_args, piece=True)
        measuredVoltage = X[4, mask]
        # print(modelVoltage.shape, measuredVoltage.shape)
        # print(f"Run ID: {id}, MAE: {np.mean(np.abs(modelVoltage - measuredVoltage))}")
        modelVoltages.append(modelVoltage)
    print(f"Overall MAE: {np.mean(np.abs(np.concatenate(modelVoltages) - X[4]))}")
    return np.concatenate(modelVoltages)

def wrapper_pre_OCV(X, R0_discharge, R0_charge, R1, C1, R2, C2, tau_H, M, a0, a1, a2, a3, a4, a5, a6, K_T):
    SOC = X[0]
    T = X[1]
    I = X[2]
    modelVoltages = []
    for id in np.unique(X[3]):
        mask = X[3] == id
        modelVoltage = terminal_voltage(X[0, mask], X[1, mask], X[2, mask], R0_discharge, R0_charge, R1, C1, R2, C2, tau_H, M, (a0, a1, a2, a3, a4, a5, a6, K_T), piece=True)
        modelVoltages.append(modelVoltage)
        # measuredVoltage = X[4, mask]
        # print(f"Run ID: {id}, MAE: {np.mean(np.abs(modelVoltage - measuredVoltage))}")
    print(f"Overall MAE: {np.mean(np.abs(np.concatenate(modelVoltages) - X[4]))}")
    print(f"R0_discharge: {R0_discharge:.3f}, R0_charge: {R0_charge:.3f}, R1: {R1:.3f}, C1: {C1:.3f}, R2: {R2:.3f}, C2: {C2:.3f}, tau_H: {tau_H:.3f}, M: {M:.3f}, a0: {a0:.3f}, a1: {a1:.3f}, a2: {a2:.3f}, a3: {a3:.3f}, a4: {a4:.3f}, a3: {a5:.3f}, a6: {a6:.3f}, K_T: {K_T:.3f}")
    return np.concatenate(modelVoltages)

def evolution_wrapper(params, X):
    R0_discharge, R0_charge, R1, C1, R2, C2, tau_H, M = params
    v = wrapper_post_OCV(X, R0_discharge, R0_charge, R1, C1, R2, C2, tau_H, M)
    MSE = np.mean((v - X[4]) ** 2)
    return MSE


# X = np.array([fullData["SOC"].to_numpy(), fullData["T_surf1"].to_numpy(), fullData["I"].to_numpy(), fullData["run"].to_numpy(), fullData["V"].to_numpy()])
# np.unique(X[3])
# X[:, np.where(X[3] == 0)]

# np.where(X[3] == 0)
# currentRun.shape
# currentRun = X[np.where(X[3] == id)]
# X

# R0_discharge: 0.00010000241368768962, 
# R0_charge: 0.0015092018205301212, 
# R1: 0.008623195393932841, 
# C1: 37.86392127904164, 
# R2: 0.013221128224458535, 
# C2: 761.6878322739899, 
# tau_H: 53.00341711702909, 
# M: 0.0018520981883959234

# Correct physical boundaries
bounds = (
    [1e-7, 1e-7, 1e-7, 1e-3,   1e-7, 10.0,   1e-5, -1], # Lower Bounds
    [0.2,  0.2,  0.2,  5000.0, 0.2,  50000.0, 1e4, 1e3]   # Upper Bounds
)

boundsTuppled = tuple(zip(*bounds))

# Initial dynamic guesses
p0 = [0.002728, 0.00010000, 0.00757, 10.0, 0.011, 735, 0.833, 0.00010000]

p0 = [ 7.54812179e-03,  1.00000000e-07,  2.44930101e-02,  5.74870831e+01,
  6.77082684e-04,  4.41486533e+04,  2.97718020e+03, -5.27874313e-03]

# Conditions generated post hysteresis function fix.
p0 = [1.00000000e-07, 1.00000000e-07, 1.40292451e-02, 3.15517635e+03,
       1.51833942e-03, 2.91537408e+04, 1.5, 2.86674119e-03]

preOCV_p0 = p0 + list(ocv_args)

lessData = fullData.filter(pl.col("t") < 1000)

args = np.array([5.30000214e-03, 9.20377540e-03,  8.07481539e-03,  1.41824989e+03,
        3.29520249e-03,  2.59134546e+02, 1.5, 1e-03,
        2.39587985e+00,  1.27417190e+01, -6.73434102e+01,  1.89554380e+02,
       -2.78887472e+02,  2.05193796e+02, -5.95230431e+01,  9.40568639e-05])

args = np.array([1.21786569e-02, 1.76761209e-02, 1.0, 1.0,
       1.0, 1.0,  9.62831610e+00,  1.03801131e-03,
        2.39250764e+00,  1.28854012e+01, -6.78772374e+01,  1.89652101e+02,
       -2.76828230e+02,  2.02226116e+02, -5.83091863e+01, -8.11863308e-05])

args = np.array([ 9.97263998e-03,  5.00002331e-03,  1.60732764e-02,  0.01,
        7.31018241e-03,  5.72244145e+02,  1.00000000e+00,  1.00000000e-07,
        2.38962033e+00,  1.29375218e+01, -6.85304383e+01,  1.92513553e+02,
       -2.82364531e+02,  2.07151056e+02, -5.99607550e+01, -1.23234655e-05])

args = np.array([0.006, 0.011, 0.001, 0.010, 0.006, 792.945, 7.433, 0.000001, 2.390, 12.938, -68.531, 192.513, -282.365, 207.151, -59.960, -0.000001])

conservative_bounds = (
# R0_discharge, R0_charge, R1, C1, R2, C2, tau_H, M, a0, a1, a2, a3, a4, a5, a6, K_T
    [0.005, 0.005, 0.0,    0.0,    0.0,    0.0,    1.0,  1e-7,  -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf], # Lower Bounds
    [0.02,  0.02,  0.02, np.inf, 0.02, np.inf, 20.0, np.inf, np.inf, np.inf,  np.inf,   np.inf,  np.inf,  np.inf,  np.inf,  np.inf]   # Upper Bounds
)

# Optimize dynamic parameters
args2, _ = curve_fit(
    wrapper_post_OCV, 
    np.array((lessData["SOC"].to_numpy(), lessData["T_surf1"].to_numpy(), lessData["I"].to_numpy(), lessData["run"].to_numpy(), lessData["V"].to_numpy())), 
    lessData["V"].to_numpy(), 
    # p0=p0, 
    # bounds=bounds, 
    maxfev=15000,
    method='trf',
    x_scale='jac'
)

args3, _ = curve_fit(
    wrapper_pre_OCV, 
    np.array((lessData["SOC"].to_numpy(), lessData["T_surf1"].to_numpy(), lessData["I"].to_numpy(), lessData["run"].to_numpy(), lessData["V"].to_numpy())), 
    lessData["V"].to_numpy(), 
    p0=args, 
    bounds=conservative_bounds,
    maxfev=1500,
    method='trf',
    x_scale='jac'
)


np.mean(np.abs(lessData["V"].to_numpy() - wrapper_pre_OCV(np.array((lessData["SOC"].to_numpy(), lessData["T_surf1"].to_numpy(), lessData["I"].to_numpy(), lessData["run"].to_numpy(), lessData["V"].to_numpy())), *args)))

args3

# argsGlobal = differential_evolution(
#     evolution_wrapper,
#     boundsTuppled,
#     args=(np.array((lessData["SOC"].to_numpy(), lessData["T_surf1"].to_numpy(), lessData["I"].to_numpy(), lessData["run"].to_numpy(), lessData["V"].to_numpy())),),
#     maxiter=100,
#     popsize=15,
#     workers=-1,
#     updating='deferred',
# )

# R0_discharge = -0.014174362096127368; R0_charge = -0.023739566602921935; R1 = 327.6950042682508; C1 = 2134.853515766918; R2 = -171.55438396480167; C2 = -2132.612618294229; tau_H = 0.8719469562101347; M = -0.00022039527420898293

tau_H = 1.0
exponent = -np.arange(0, kernel_size) * delta / tau_H
kernel = np.exp(exponent) / np.sum(np.exp(exponent))
# plt.plot(kernel)
# plt.show()

# Make sure the kernel faces the right way
plt.plot(convolve(np.pad(np.concatenate([np.ones(kernel_size), np.ones(kernel_size)*-1]), (kernel_size-1, 0), mode='constant'), kernel, mode="valid"), label="Convolution Result")
plt.plot(np.concatenate([np.ones(kernel_size), np.ones(kernel_size)*-1]), label="Input Current")
plt.legend()
plt.show()

arr = np.array((lessData["SOC"].to_numpy(), lessData["T_surf1"].to_numpy(), lessData["I"].to_numpy(), lessData["run"].to_numpy(), lessData["V"].to_numpy(), lessData["t"].to_numpy()))
for id, name in zip(np.unique(arr[3]), notHighPulse):
    currentRun = arr[:, np.where(arr[3] == id)][:, 0, :]
    V_OCV, V_R0, V_X1, V_X2, V_h = debug_terminal_voltage(currentRun[0], currentRun[1], currentRun[2], *args[:6], 10, args[7], args[8:], piece=True, verbose=True, superVerbose=False)
    measuredVoltage = currentRun[4]
    print(f"Run ID: {id}, MAE: {np.mean(np.abs(V_OCV + V_R0 + V_X1 + V_X2 + V_h - measuredVoltage))}")
    plt.plot(currentRun[5], measuredVoltage, label="Measured Voltage", alpha=0.8)
    plt.plot(currentRun[5], V_OCV + V_R0 + V_X2 + V_h, '--', label="Fitted Performance ECM", alpha=0.9)
    plt.plot(currentRun[5], V_OCV, label="OCV", alpha=0.8)
    plt.plot(currentRun[5], V_R0, label="R0 Voltage Drop", alpha=0.8)
    plt.plot(currentRun[5], V_X1, label="X1 Voltage", alpha=0.8)
    plt.plot(currentRun[5], V_X2, label="X2 Voltage", alpha=0.8)
    plt.plot(currentRun[5], V_h, label="Hysteresis Voltage", alpha=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title(name.name)
    plt.legend()
    plt.grid(True)
    plt.show()










# print("Optimized Parameters (R0_d, R0_c, R1, C1, R2, C2, tau_H, M):")
# print(args2)

# # 4. Plot Verification Validating Against One Dynamic Run
# run = measurements.fu.DCC[6]
# run_soc = func_SOC(run.I, run.t, SOC_lookup(run.V[0], run.T_surf[0, 0]))
# fitted_v = terminal_voltage(run_soc, run.T_surf[0, :], run.I, args2[0], args2[1], args2[2], args2[3], args2[4], args2[5], args2[6], args2[7], ocv_args, piece=True)

# np.abs(terminal_voltage(run_soc, run.T_surf[0, :], run.I, args2[0], args2[1], args2[2], args2[3], args2[4], args2[5], args2[6], args2[7], ocv_args, piece=True) - run.V).mean()


# plt.figure(figsize=(10, 5))
# plt.plot(run.t, run.V, label="Measured Voltage", alpha=0.8)
# plt.plot(run.t, fitted_v, '--', label="Fitted Performance ECM", alpha=0.9)
# plt.xlabel("Time (s)")
# plt.ylabel("Voltage (V)")
# plt.title(f"Model Validation Execution - Run: {run.name}")
# plt.legend()
# plt.grid(True)
# plt.show()

# plt.plot(fullData["SOC"].to_numpy())
# plt.show()

# for run in aboveFreezingSims:
#     plt.plot(func_SOC(run.I, run.t, SOC_lookup(run.V[0], run.T_surf[0, 0])))
#     plt.show()

# for run in aboveFreezingSims:
#     plt.plot(run.t, run.V, label="Measured Voltage", alpha=0.8)
#     plt.plot(run.t, terminal_voltage(func_SOC(run.I, run.t, SOC_lookup(run.V[0], run.T_surf[0, 0])), run.T_surf[0, :], run.I, p0[0], p0[1], p0[2], p0[3], p0[4], p0[5], p0[6], p0[7], ocv_args, piece=True), '--', label="SOC", alpha=0.9)
#     plt.xlabel("Time (s)")
#     plt.ylabel("Voltage (V)")
#     plt.title(f"Run: {run.name}")
#     plt.legend()
#     plt.grid(True)
#     plt.show()