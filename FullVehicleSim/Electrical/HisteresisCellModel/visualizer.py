import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, differential_evolution
from scipy.integrate import cumulative_simpson
from scipy.signal import convolve
import scipy.io
from scipy.interpolate import LinearNDInterpolator as Lnd_interp
from scipy.interpolate import CubicSpline as cs


'''
header = batemoData["__header__"]
version = batemoData["__version__"]
globals = batemoData["__globals__"] ## Nothing
print(f"header = {header}")
print(f"version = {version}")
measurement = batemoData["measurement"]
firstLayerMeta = measurement['meta']
Fu = measurement['fu']

# DCC (Discharge)
# CHC (Charging)
# DCP (Discharge Pulse)
# CHP (Charge Pulse)
PRO (Profile Measurement)

Each Has
    name
    T_amb (Ambient Temperature)
    t (Time Seconds)
    I (Current)
    V (Voltage)
    T_surf (Surface temperature, nominally 1xN but may be 2xN?)
'''

matData = scipy.io.loadmat("../fs-data/FS-4/P30B Cell Data/Molicel_INR18650P30B_measurement.mat", squeeze_me=True, struct_as_record=False)
measurements = matData["measurement"]

DCP = measurements.fu.DCP
DCC = measurements.fu.DCC
CHC = measurements.fu.CHC

# plt.plot(DCP[8].t, DCP[8].I, label = "Current (A)")
plt.plot(DCP[8].t, DCP[8].V, label = "Voltage (V)")
# plt.legend()
plt.ylabel("Voltage (V)")
plt.xlabel("Time (s)")
plt.show()

for i, thing in enumerate(DCC):
    print(i, thing.T_amb, thing.I[100]) 

# 25, -3 is [12]
# 25, 3 is [4]
# OCV for discharge is [10]

c = CHC[4]
d3 = DCC[12]
d30 = DCC[17]
docv = DCC[10]

cQ = cumulative_simpson(c.I, x=c.t)/3600
d3Q = cumulative_simpson(d3.I, x=d3.t)/3600
d30Q = cumulative_simpson(d30.I, x=d30.t)/3600
docvQ = cumulative_simpson(docv.I, x=docv.t)/3600

plt.plot(-1*cQ, c.V[1:], label = "Charge Curve (3A)")
plt.plot(d3Q[-1] - d3Q, d3.V[1:], label = "Discharge Curve (3A)")
plt.plot(d30Q[-1] - d30Q, d30.V[1:], label = "Discharge Curve (30A)")
plt.plot(docvQ[-1] - docvQ, docv.V[1:], label = "OCV (0.3A)")
plt.legend()
plt.xlabel("SOC (Ah)")
plt.ylabel("Voltage (V)")
plt.show()