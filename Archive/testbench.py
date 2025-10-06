from ramen import Parameters, Magic
from Mech.traction import *
import numpy as np
precision = 0.1
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import json
import pandas as pd
import argparse

def plotPressure(ax, velocity):
    if True:
        points = []
        for SA in np.arange(0, 15 + precision, precision):
            runTire = tire.Tire(1000, 0.15, SA, velocity, 80, 40, Parameters, Magic)
            longForce = runTire.getLateralForce() / 1000

            runTire2 = tire.Tire(1000, 0.15, SA + precision, velocity, 80, 40, Parameters, Magic)
            longForce2 = runTire2.getLateralForce() / 1000
            points.append(( SA,(longForce2-longForce)/precision) )
        #print(len(lgx), len(lgy))
        x = []
        y = []
        for i in points:
            x.append(float(i[0]))
            y.append(float(i[1]))
        #@print("DONE")
        #lgx, lgy  = zip(*points)
        print(len(x))
        print(len(y))

        plt.xlabel('Slip Angle')
        plt.ylabel('Normalized Cornering Stiffness')
        plt.title('Normalized Cornering Stiffness Response to Slip Angle')

        plt.scatter(x, y, s=len(x))
        plt.show()

plotPressure(1,40)


print(getCorneringStiffness([-1000, -1000, -1000, -1000], 5, 0.15, 1.5, 80, 40, Parameters, Magic))
