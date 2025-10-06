# There's definitely a better way to do this but whatever
import sys
import os
sys.path.append('/Users/daniel/Documents/Coding/FormulaSlug/sim-test/VehicleDynamics/TireModel')
print(os.path.dirname(os.getcwd()) + '/TireModel')
import dumpling as tire 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import numpy as np

with open('../TireModel/params.json', 'r') as file:
    params = json.load(file)

precision = 0.1 # Controls step size in slip ratio and slip angle
def debugLongForce3D(velocity, force, params, precision):
    points = []
    for SA in np.arange(0, 11.5 + precision, precision):
        for SR in np.arange(-1, 0.5 + precision/10, precision/10):
            runTire = tire.Tire(force, SR, SA, velocity, params["Constants"], params["Mechanical-Parameters"], params["Magic-Parameters"])
            longForce = runTire.getLongForce()
            points.append((SR, SA, longForce))
    return points


if __name__ == "__main__":
    velocity = 40 #mph
    force = 100*9.8 #kg

    forces = []
    for i in range(50,1000,50):
        forces.append(debugLongForce3D(velocity, i, params, precision))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for force in forces:
        lgx, lgy, lgz = zip(*force)
        ax.scatter(lgx, lgy, lgz, c='b', marker='x')  
    



    #ax.scatter(lgx, lgy, lgz, c='r', marker='o')  
    

    ax.set_xlabel('SR')
    ax.set_ylabel('SA')
    ax.set_zlabel('Force')
    plt.show()
   
    
    
    