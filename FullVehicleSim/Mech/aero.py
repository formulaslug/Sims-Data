import numpy as np

def calculateDrag(heading, speed, crossSectionalArea=0.63, dragCoefficient=0.6, airDensity=1.230):
    return  0.5 * airDensity * dragCoefficient * crossSectionalArea * speed**2

def calculateDownForce(heading, speed, parameters):
    return np.asarray([0,0,0,0], dtype=float)
