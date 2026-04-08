import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ======================================================
# PARAMETERS
# ======================================================
Q_rated = 2.5 * 3600       # Coulombs (2.5 Ah cell)
dt = 1.0                  # timestep [s]
N = 300                   # number of steps

# ======================================================
# VEHICLE / BATTERY STATE
# ======================================================
@dataclass
class VehicleState:
    soc: float             # State of Charge [0..1]
    voltage: float         # Terminal voltage [V]

    # Reserved for future revised model
    temp_c: float = 25.0
    hyst: float = 0.0
    v_rc: float = 0.0


# ======================================================
# VOLTAGE UPDATE TEMPLATE  ← THIS IS THE DELIVERABLE
# ======================================================
def update_voltage_template(prev_current: float, state: VehicleState) -> float:
    """
    Template for revised voltage updating (ECM baseline).

    Model:
      V = OCV(SOC) - I*R_internal

    Inputs:
      prev_current: current from previous timestep [A] (+ discharge, - charge)
      state.soc: SOC in [0,1]

    Output:
      new terminal voltage [V]

    TODO later:
      - Replace OCV curve with real discharge curve fit
      - Make R_internal depend on temperature/SOC
      - Add hysteresis / RC recovery (sag + slow rebound)
    """
    soc = float(state.soc)

    # 1) OCV curve (placeholder, but reasonable):
    #    At SOC=1.0 -> ~4.2V, at SOC=0.0 -> ~3.0V
    ocv = 3.0 + 1.2 * soc

    # 2) Internal resistance (placeholder constant)
    r_internal = 0.015  # Ohms (cell-level example)

    # 3) Terminal voltage
    v_new = ocv - prev_current * r_internal

    # 4) Clamp to realistic bounds to avoid weird plots
    v_new = float(np.clip(v_new, 2.5, 4.25))

    return v_new


# ======================================================
# INPUT CURRENT PROFILE (synthetic)
# ======================================================
time = np.arange(N) * dt
I = np.zeros(N)

I[20:80]   = 5.0
I[100:150] = 2.5
I[170:220] = -3.0
I[250:280] = 6.0

# ======================================================
# LOG ARRAYS
# ======================================================
SOC_log = np.zeros(N)
V_log = np.zeros(N)

# ======================================================
# INITIAL STATE
# ======================================================
state = VehicleState(
    soc=1.0,
    voltage=4.2,   # reasonable initial guess
)

SOC_log[0] = state.soc
V_log[0] = state.voltage

# ======================================================
# SIMULATION LOOP
# ======================================================
for t in range(1, N):
    prev_I = I[t - 1]

    # --- SOC UPDATE (Coulomb counting) ---
    state.soc -= (prev_I * dt / Q_rated)
    state.soc = float(np.clip(state.soc, 0.0, 1.0))

    # --- VOLTAGE UPDATE (TEMPLATE CALL) ---
    state.voltage = update_voltage_template(prev_I, state)

    # --- LOG ---
    SOC_log[t] = state.soc
    V_log[t] = state.voltage

# ======================================================
# PLOTS
# ======================================================
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(time, I, label="Current [A]")
plt.ylabel("Current (A)")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time, V_log, label="Voltage [V]", color="tab:red")
plt.ylabel("Voltage (V)")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time, SOC_log * 100, label="SOC [%]", color="tab:green")
plt.xlabel("Time (s)")
plt.ylabel("SOC (%)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
