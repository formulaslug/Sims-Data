import numpy as np
import matplotlib.pyplot as plt
from FSLib.IntegralsAndDerivatives import *
from FSLib.fftTools import *
from FSLib.AnalysisFunctions import *
print(df.columns)

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")

time = data["time"]
speed = data["speed"]
drag = data["drag"]


speed_drag = speed * drag

# -------- Graph 1: Speed vs Time --------
plt.figure()
plt.plot(time, speed)
plt.title("Speed vs Time")
plt.xlabel("Time")
plt.ylabel("Speed")
plt.grid()
plt.show()

# -------- Graph 2: Speed × Drag vs Time --------
plt.figure()
plt.plot(time, speed_drag)
plt.title("Speed × Drag Coefficient vs Time")
plt.xlabel("Time")
plt.ylabel("Speed × Drag")
plt.grid()
plt.show()
