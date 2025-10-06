from ramen import Parameters, Magic
import matplotlib.pyplot as plt
from Mech.traction import *
import numpy as np

stiff = []
for i in np.arange(0,10,0.1):
    #curr = getTraction([335,335,554,554], (i,i), 0.15, 40, 80, 40, Parameters, Magic)
    #stiff.append((i, curr[0][1]))
    curr = getCorneringStiffness([335,335,554,554], (i,i), 0.15, 40, 80, 40, Parameters, Magic)
    stiff.append((i, curr[0]))


stiff = np.array(stiff)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(stiff[:, 0], stiff[:, 1], label='Cornering Stiffness', color='blue')
plt.title('Cornering Stiffness vs SA')
plt.xlabel('SA')
plt.ylabel('Cornering Stiffness')
plt.grid()
plt.show()
