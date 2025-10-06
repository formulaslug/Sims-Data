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

precision = 0.01 # Controls step size in slip ratio and slip angle

def getTractionCircleLayer(velocity, mass, params, precision):

    latToLong = {} #Store the max long force for every lateral force

    
    for SA in np.arange(0, 11.5 + precision/1, precision/1):
        for SR in np.arange(0, 1 + precision/10, precision/10):
            latTire = tire.Tire(9.8*mass, SR, SA, velocity, params["Constants"], params["Mechanical-Parameters"], params["Magic-Parameters"])
            latForce = latTire.getLateralForce()/(mass*9.8)

            longTire = tire.Tire(9.8*mass, SR*-1, abs(SA), velocity, params["Constants"], params["Mechanical-Parameters"], params["Magic-Parameters"])
            longForce = longTire.getLongForcePureSlip()/(mass*9.8)

            if latForce in latToLong:
                latToLong[latForce] = max(longForce, latToLong[latForce])
            else:
                latToLong[latForce] = longForce

    return latToLong

def debugLongForce(velocity, mass, params, precision):
    res = {}
    for SR in np.arange(-1, 0 + precision/10, precision/10):
        runTire = tire.Tire(mass, SR, 0, velocity, params["Constants"], params["Mechanical-Parameters"], params["Magic-Parameters"])
        longForce = runTire.getLongForce() 
        res[SR] = longForce
    return res
def debugLatForce(velocity, force, params, precision):
    res = {}
    for SA in np.arange(0, 10 + precision, precision):
        runTire = tire.Tire(force, 0.3, SA, velocity, params["Constants"], params["Mechanical-Parameters"], params["Magic-Parameters"])
        longForce = runTire.getLateralForce() 
        res[SA] = longForce
    return res
def debugLongForce3D(velocity, force, params, precision):
    points = []
    for SA in np.arange(-11.5, 11.5 + precision, precision):
        for SR in np.arange(-1, 0.5 + precision/10, precision/10):
            runTire = tire.Tire(force, SR, SA, velocity, 85, 30, params["Constants"], params["Mechanical-Parameters"], params["Magic-Parameters"])
            longForce = runTire.getLongForce() / force
            points.append((SR, SA, longForce))
    return points
def debugLatForce3D(velocity, force, params, precision):
    points = []
    for SA in np.arange(0, 11.5 + precision, precision):
        for SR in np.arange(0, 0.2 + precision/10, precision/10):
            runTire = tire.Tire(force, SR, SA, velocity, 12, 30, 80, 30, params["Constants"], params["Mechanical-Parameters"], params["Magic-Parameters"])
            latForce = runTire.getLateralForce()/ force
            points.append((SR, SA, latForce))
    return points

def findForce(arr1, arr2):
    #print(len(arr1), len(arr2))
    points_dict = {(round(point[0],2), round(point[1],2)): point[2] for point in arr1}
    #print("DICT", len(points_dict))
    
    z_values = []
    for point in arr2:
        key = (round(point[0],2), round(point[1],2))
        if key in points_dict:
            #print(key)
            z_values.append((points_dict[key], point[2]))
    

    #print(len(z_values))
    #print("END")
    return z_values

if __name__ == "__main__":
    velocity = 40 #mph
    mass = 100*9.8 #kg

    #runTire = tire.Tire(9.8*mass, 0.14, 0, velocity, params["Constants"], params["Mechanical-Parameters"], params["Magic-Parameters"])
    #print(runTire.getLongForce())
    #print(runTire.getLateralForce())




    
   
    # Traction quarter? circle creation
    longf = debugLongForce3D(velocity, mass, params, precision)
    latf = debugLatForce3D(velocity, mass, params, precision)

    points = findForce(longf,latf)

    x, y = zip(*points)  # This unzips the list of tuples into two lists

    # Step 4: Create the scatter plot
    plt.scatter(x, y)

    # Adding labels and title
    plt.xlabel('Lat Force (Newtons)')
    plt.ylabel('Long Force (Newtons)')
    plt.title('GG diagram')

    # Show the plot
    plt.show()


    """
    
    # Draw pure long force plot 
    arr = []
    for SR in np.arange(-0.9, 0.9 + precision/10, precision/10):
        runTire = tire.Tire(500, SR, 0, velocity, 85, 30, params["Constants"], params["Mechanical-Parameters"], params["Magic-Parameters"])
        longForce = runTire.getLongForcePureSlip() / 500
        arr.append((SR, longForce))

    x, y = zip(*arr)
    plt.scatter(x, y, color='blue', label='Data Points')
    plt.grid()
    plt.show()

    # Displays lat force/long force on a plot
    longf = debugLongForce3D(velocity, mass, params, precision)
    #latf = debugLatForce3D(velocity, mass, params, precision)

    
    lgx, lgy, lgz = zip(*longf)
    #ltx, lty, ltz = zip(*latf)
    

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(lgx, lgy, lgz, c='r', marker='o')  
    #ax.scatter(ltx, lty, ltz, c='b', marker='x')  

    ax.set_xlabel('SR')
    ax.set_ylabel('SA')
    ax.set_zlabel('Force')
    plt.show()
    """
   
    
    
    