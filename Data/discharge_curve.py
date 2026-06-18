import numpy as np
import polars as pl
import scipy.io
from scipy.integrate import cumulative_simpson
from scipy.interpolate import LinearNDInterpolator as Lnd_interp
import matplotlib.pyplot as plt

matData = scipy.io.loadmat("../fs-data/FS-4/P30B Cell Data/Molicel_INR18650P30B_measurement.mat", squeeze_me=True, struct_as_record=False)
measurements = matData["measurement"]

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

energy = -(lowts[:-1] - lowts[1:]).mean() * cumulative_simpson(-lowVs * lowIs)/3.6e6*19*30

dfVTC5A = pl.read_csv("../fs-data/FS-3/voltageTableVTC5A.csv")
vtc5a_discharge = dfVTC5A.filter(pl.col("Current") == 0.5)

plt.plot(vtc5a_discharge["Charge"]*vtc5a_discharge["Voltage"])
plt.show()

cumulative_simpson(vtc5a_discharge["Voltage"][:-1], x=vtc5a_discharge["Charge"][:-1])*20*30
