import matplotlib.pyplot as plt
import numpy as np

x1_values = rpm
y1_values = time
x2_values = gps
y2_values = time
x3_values = 1/2(rpm+gps)
y3_values= time

plt.plot(x1_values, y1_values)
plt.plot(x2_values, y2_values)
plt.plot(x3_values, y3_values)

plt.show
