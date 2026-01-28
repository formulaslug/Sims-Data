from paramLoader import Parameters, Magic
from state import VehicleState
import numpy as np

def update_pack_voltage_template(prev_current: float, prevWorld: VehicleState) -> float:

    # --- dt ---
    # dt = float(getattr(prevWorld, "stepSize", 1.0))

    # # --- initialize battery memory states if missing ---
    # if not hasattr(prevWorld, "batt_v_rc"):
    #     prevWorld.batt_v_rc = 0.0
    # if not hasattr(prevWorld, "batt_hyst"):
    #     prevWorld.batt_hyst = 0.0
    # if not hasattr(prevWorld, "batt_temp_c"):
    #     prevWorld.batt_temp_c = 25.0

    
    # PARAMETERS (Murata VTC5A-ish ECM)

    Q_rated = 2.5 * 3600       # Coulombs (2.5Ah cell)
    R0 = 0.015
    R1 = 0.02
    C1 = 200.0

    lambda_hyst = 0.98
    gamma_hyst = 0.03
    T_ref = 25.0
    alpha_R = 0.004
    beta_V = 0.0008
    m, c, hA = 0.045, 1000.0, 0.6

    # Gaussian settings (matches sims-battery_voltage.py idea)
    kernel_size = 9   # must be odd
    sigma = 2.0
    gauss_weight = 0.2  # tune like your script

    # --- helper functions ---
    def f_OCV(SOC):
        SOC = float(np.clip(SOC, 0.0, 1.0))
        return 3.0 + 1.2*SOC - 0.3*(SOC**2) + 0.1*(SOC**3)

    def R_int(T):
        return R0 * (1 + alpha_R * (T - T_ref))

    
    SOC = float(np.clip(prevWorld.charge / 2.5, 0.0, 1.0))

    # --- OCV ---
    V_OCV = f_OCV(SOC)

    # --- RC polarization update ---
    exp_factor = np.exp(-dt / (R1 * C1))
    V_RC = float(prevWorld.batt_v_rc * exp_factor + R1 * (1 - exp_factor) * prev_current)

    # --- Gaussian current from history array ---
    # Use the array from state; if missing, just use prev_current.
    hist = np.asarray(getattr(prevWorld, "current_history", np.array([prev_current])), dtype=float)

    # build kernel
    if kernel_size % 2 == 0:
        kernel_size += 1
    x = np.linspace(-3, 3, kernel_size)
    gaussian_kernel = np.exp(-(x**2) / (2 * sigma**2))
    gaussian_kernel /= gaussian_kernel.sum()

    # window and trimmed kernel (like your sim script)
    I_window = hist[-kernel_size:] if hist.size > 0 else np.array([prev_current], dtype=float)
    k_window = gaussian_kernel[-len(I_window):]
    I_gauss = float(np.sum(I_window * k_window))

    # --- hysteresis update (with gaussian term) ---
    H = float(
        lambda_hyst * prevWorld.batt_hyst
        + (1 - lambda_hyst) * prev_current
        + gauss_weight * I_gauss
    )

    # --- thermal update ---
    T_prev = float(prevWorld.batt_temp_c)
    T = float(
        T_prev + (dt / (m * c)) * (prev_current**2 * R_int(T_prev) - hA * (T_prev - T_ref))
    )

    # --- temperature corrections ---
    R0_eff = R0 * (1 + alpha_R * (T - T_ref))
    V_OCV_T = V_OCV + beta_V * (T - T_ref)

   # --- cell terminal voltage ---
    V_cell = float(V_OCV_T - prev_current * R0_eff - V_RC + gamma_hyst * H)
    V_cell = float(np.clip(V_cell, 2.5, 4.25))

    prevWorld._cell_voltage = V_cell

    prevWorld.batt_v_rc = V_RC
    prevWorld.batt_hyst = H
    prevWorld.batt_temp_c = T

    return float(Parameters["seriesCells"] * V_cell)