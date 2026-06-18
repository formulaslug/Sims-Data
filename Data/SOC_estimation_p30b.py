import numpy as np
import scipy.io
from scipy.optimize import curve_fit, differential_evolution
from scipy.integrate import cumulative_simpson
from scipy.signal import convolve
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator as Lnd_interp

# --- 1. Cell Parameters (Approximated for a typical Li-ion cell like P30B) ---
# In practice, map these to your specific OCV-SOC curve and characterization data
R0 = 0.00686  # Ohms (Internal resistance)
RC_R1 = 0.015  # Ohms (RC circuit resistance)
RC_C1 = 2000.0 # Farads (RC circuit capacitance)
Q_max = 3.0 # Ah (Nominal capacity)
dt = 0.01    # Seconds (Sampling time)

args = np.array([ 9.97263998e-03,  5.00002331e-03,  1.60732764e-02,  0.01,
        7.31018241e-03,  5.72244145e+02,  1.00000000e+00,  1.00000000e-07,
        2.38962033e+00,  1.29375218e+01, -6.85304383e+01,  1.92513553e+02,
       -2.82364531e+02,  2.07151056e+02, -5.99607550e+01, -1.23234655e-05])

R0_discharge, R0_charge, R1, C1, R2, C2, tau_H, M, a0, a1, a2, a3, a4, a5, a6, K_T = args

def OCV_terminal_voltage(SOC, T):
    return (a0 + a1 * SOC + a2 * SOC**2 + a3 * SOC**3 + a4 * SOC**4 + a5 * SOC**5 + a6 * SOC**6) * (1 + K_T * (T - 25))

def d_OCV_terminal_voltage(SOC, T):
    return (a1 + 2 * a2 * SOC + 3 * a3 * SOC**2 + 4 * a4 * SOC**3 + 5 * a5 * SOC**4 + 6 * a6 * SOC**5) * (1 + K_T * (T - 25))

matData = scipy.io.loadmat("../fs-data/FS-4/P30B Cell Data/Molicel_INR18650P30B_measurement.mat", squeeze_me=True, struct_as_record=False)
measurements = matData["measurement"]
all_sims = np.concatenate([measurements.fu.DCC, measurements.fu.CHC, measurements.fu.DCP, measurements.fu.CHP])
aboveFreezingSims = [m for m in all_sims if m.T_surf[0, 0] > 5]
notHighPulse = [m for m in aboveFreezingSims if not ("-40.000C" in m.name or "-30.000C" in m.name)]

notHighPulse[0].name
# def terminal_voltage(SOC, T, I, R0_discharge, R0_charge, R1, C1, R2, C2, tau_H, M, ocv_params, piece=False, verbose=False, superVerbose=True):
#     tau1 = R1 * C1
#     tau2 = R2 * C2
    
#     # Kernel calculation
#     exponent = -np.arange(0, kernel_size) * delta / tau_H
#     kernel = np.exp(exponent) / np.sum(np.exp(exponent))
    
#     V_h = func_V_h(SOC, M, I, kernel)
#     V_X1 = func_V_X_fast(tau1, R1, I, delta)
#     V_X2 = func_V_X_fast(tau2, R2, I, delta)
    
#     V_OCV = OCV_terminal_voltage(SOC, T, *ocv_params)
#     V_R0 = np.where(I <= 0, R0_discharge * I, R0_charge * I)
#     # if superVerbose:
#         # print(f"R0_discharge: {R0_discharge:.5f}, R0_charge: {R0_charge:.5f}, R1: {R1:.5f}, C1: {C1:.5f}, R2: {R2:.5f}, C2: {C2:.5f}, tau_H: {tau_H:.5f}, M: {M:.5f}")
#     if not piece:
#         print(f"MAE: {np.mean(np.abs(V_OCV + V_R0 + V_X1 + V_X2 + V_h - fullData['V'].to_numpy()))}")
#     # print(f"MSE: {np.mean((V_OCV + V_R0 + V_X1 + V_X2 + V_h - fullData['V'].to_numpy()) ** 2)}")
#     if verbose:
#         print(f"V_OCV: {V_OCV[-1]}, V_R0: {V_R0[-1]}, V_X1: {V_X1[-1]}, V_X2: {V_X2[-1]}, V_h: {V_h[-1]}, V_terminal: {V_OCV[-1] + V_R0[-1] + V_X1[-1] + V_X2[-1] + V_h[-1]}")
    
#     return V_OCV + V_R0 + V_X1 + V_X2 + V_h

def get_ocv(soc):
    """Simple OCV-SOC polynomial approximation."""
    return OCV_terminal_voltage(soc, 25)

def get_d_ocv_d_soc(soc):
    """Derivative of OCV wrt SOC (Jacobian element)."""
    return d_OCV_terminal_voltage(soc, 25)

# --- 2. AEKF Core Steps (Functional Style) ---

def aekf_predict(state, P, I, Q_mat):
    """Predicts the next state and error covariance."""
    # soc, v1 = state[0, 0], state[1, 0]
    
    # State transition matrix (A)
    A = np.array([[1.0, 0.0],
                  [0.0, np.exp(-dt / (RC_R1 * RC_C1))]])
    
    # Control input matrix (B)
    B = np.array([[-dt / (Q_max * 3600)],
                  [RC_R1 * (1 - np.exp(-dt / (RC_R1 * RC_C1)))]])
    
    # State equation: x_k = A * x_{k-1} + B * u_k
    next_state = A @ state + B * I
    next_P = A @ P @ A.T + Q_mat
    return next_state, next_P

def aekf_update(state, P, I, V_meas, R_mat, H_jac, alpha=0.3):
    """Updates the state using the measurement and adapts R."""
    soc, v1 = state[0, 0], state[1, 0]
    
    # Predicted terminal voltage
    V_pred = get_ocv(soc) - v1 - I * R0
    
    # Innovation (residual)
    residual = V_meas - V_pred
    
    # Innovation covariance
    S = H_jac @ P @ H_jac.T + R_mat
    
    # Kalman Gain
    K = P @ H_jac.T @ np.linalg.inv(S)
    
    # Update State and Covariance
    updated_state = state + K * residual
    updated_P = (np.eye(2) - K @ H_jac) @ P
    
    # Adaptive Step: Update Measurement Noise Covariance (R)
    # R_adaptive = alpha * R_previous + (1 - alpha) * (residual^2 - H*P*H^T)
    R_innov = (residual ** 2) - (H_jac @ P @ H_jac.T)
    updated_R = alpha * R_mat + (1 - alpha) * np.maximum(R_innov, 1e-5) 
    
    return updated_state, updated_P, updated_R

# --- 3. Execution Pipeline ---

def estimate_soc(current_profile, voltage_profile, initial_soc=0.8):
    """Functional pipeline to iterate through time series data."""
    # State vector: [SOC, V1]^T
    state = np.array([[initial_soc], [0.0]])
    P = np.diag([1e-4, 1e-4])      # State covariance matrix
    Q_mat = np.diag([1e-6, 1e-5])  # Process noise covariance
    R_mat = np.array([[0.01]])     # Initial measurement noise covariance
    
    soc_history = []
    
    for I, V_meas in zip(current_profile, voltage_profile):
        # 1. Predict
        state, P = aekf_predict(state, P, I, Q_mat)
        
        # 2. Compute Jacobian matrix H based on predicted SOC
        dh_dsoc = get_d_ocv_d_soc(state[0, 0])
        H_jac = np.array([[dh_dsoc, -1.0]])
        
        # 3. Update & Adapt
        state, P, R_mat = aekf_update(state, P, I, V_meas, R_mat, H_jac)
        
        soc_history.append(state[0, 0])
        
    return np.array(soc_history)

# --- Example Usage with Dummy Data ---
# Simulate 100 seconds of constant discharge (3A) with random voltage noise
time_length = 100
time_steps = int(time_length / dt)
time = np.arange(time_steps) * dt
mock_current = np.ones_like(time) * 0.0001
v0 = 4.2
v1 = 2.5

mock_voltage = np.linspace(v0, v1, time_steps) #+ np.random.normal(0, 0.02, time_steps)

estimated_soc = estimate_soc(mock_current, mock_voltage, initial_soc=0.9)
print(f"Initial SOC Estimate: {estimated_soc[0]:.4f}")
print(f"Final SOC Estimate: {estimated_soc[-1]:.4f}")

plt.plot(time, estimated_soc, label="Estimated SOC")
plt.xlabel("Time Steps")
plt.ylabel("State of Charge (SOC)")
plt.title("AEKF SOC Estimation for P30B Cell")
plt.legend()
plt.grid(True)
plt.show()

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
    return SOC

# plt.scatter(lowVs, lowSOC, label="Data Points")
V_grid = np.linspace(2.5, 4.2, 100)
T_grid = np.linspace(-20, 40, 100)
V_mesh, T_mesh = np.meshgrid(V_grid, T_grid)
SOC_mesh = voltage_curve(V_mesh, T_mesh)
plt.contourf(V_mesh, T_mesh, SOC_mesh, levels=50, cmap='viridis', alpha=0.7)
plt.colorbar(label="Estimated SOC")
plt.xlabel("Voltage (V)")
plt.ylabel("Temperature (°C)")
plt.title("SOC Estimation from Voltage and Temperature")
plt.legend()
plt.grid(True)
plt.show()

plt.plot((1 - lowSOC[65148:99925])*3, lowVs[65148:99925], label="Data")
plt.plot((1 - np.array([SOC_lookup(V, 25) for V in V_grid]))*3, V_grid, label="Curve Fit")
plt.legend()
plt.xlabel("Discharged (Ah)")
plt.ylabel("Voltage (V)")
plt.title("OCV-SOC Curve for P30B Cell at 25°C")
plt.show()


