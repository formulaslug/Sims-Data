import numpy as np
import scipy.io
from scipy.optimize import curve_fit, differential_evolution
from scipy.integrate import cumulative_simpson
from scipy.signal import convolve
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator as Lnd_interp
import polars as pl

# --- 1. Cell Parameters (Approximated for a typical Li-ion cell like P30B) ---
# In practice, map these to your specific OCV-SOC curve and characterization data
R0 = 0.00686  # Ohms (Internal resistance)
Q_max = 3.0 # Ah (Nominal capacity)

args = np.array([ 9.97263998e-03,  5.00002331e-03,  1.60732764e-02,  0.01,
        7.31018241e-03,  5.72244145e+02,  1.00000000e+00,  1.00000000e-07,
        2.38962033e+00,  1.29375218e+01, -6.85304383e+01,  1.92513553e+02,
       -2.82364531e+02,  2.07151056e+02, -5.99607550e+01, -1.23234655e-05])

R0_discharge, R0_charge, R1, C1, R2, C2, tau_H, M, a0, a1, a2, a3, a4, a5, a6, K_T = args

def OCV_SOC(V):
    return (a0 + a1 * V + a2 * V**2 + a3 * V**3 + a4 * V**4 + a5 * V**5 + a6 * V**6)

matData = scipy.io.loadmat("../fs-data/FS-4/P30B Cell Data/Molicel_INR18650P30B_measurement.mat", squeeze_me=True, struct_as_record=False)
measurements = matData["measurement"]
all_sims = np.concatenate([measurements.fu.DCC, measurements.fu.CHC, measurements.fu.DCP, measurements.fu.CHP])
aboveFreezingSims = [m for m in all_sims if m.T_surf[0, 0] > 5]
notHighPulse = [m for m in aboveFreezingSims if not ("-40.000C" in m.name or "-30.000C" in m.name)]

lowCurrMeasurements = [m for m in measurements.fu.DCC if "-0.1" in m.name]

def func_SOC(I, t, starting_SOC):
    Q_nominal = 3.0  # Ah capacity of Molicel P30B
    integrated_ah = cumulative_simpson(y=I, x=t, initial=0) / 3600.0
    soc = starting_SOC + (integrated_ah / Q_nominal)
    return np.clip(soc, 0.0, 1.0)

lowVs = np.concatenate([sim.V for sim in lowCurrMeasurements])
lowIs = np.concatenate([sim.I for sim in lowCurrMeasurements])
lowTs = np.concatenate([sim.T_surf[0, :] for sim in lowCurrMeasurements])
lowts = np.concatenate([sim.t for sim in lowCurrMeasurements])
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

energy = -(lowts[:-1] - lowts[1:]).mean() * cumulative_simpson(-lowVs * lowIs)/3.6e6*19*30 # kWh


dfVTC5A = pl.read_csv("../fs-data/FS-3/voltageTableVTC5A.csv")
vtc5a_discharge = dfVTC5A.filter(pl.col("Current") == 0.5)

plt.plot(vtc5a_discharge["Charge"]*vtc5a_discharge["Voltage"])
plt.show()

cumulative_simpson(vtc5a_discharge["Voltage"][:-1], x=vtc5a_discharge["Charge"][:-1])*20*30

SOC_lookup(4.13, 25)


