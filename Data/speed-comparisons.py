import matplotlib.pyplot as plt
import numpy as np
from FSLib.IntegralsAndDerivatives import *
from FSLib.fftTools import *
from FSLib.AnalysisFunctions import *

rpm = "SME_TRQSPD_Speed"
speed = "VDM_GPS_SPEED"

dt = 0.01  # adjust to your logger
time = np.arange(df.height) * dt

y1_values = df[rpm].to_numpy() * 12/41*0.2*2*np.pi/60
y2_values = df[speed].to_numpy()
y3_values= .5 * (y1_values + y2_values)

plt.plot(time, y1_values, label = "RPM Speed")
plt.plot(time, y2_values, label = "GPS Speed")
plt.plot(time, y3_values, label = "Avg Speed")

plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Speed")
plt.show()
