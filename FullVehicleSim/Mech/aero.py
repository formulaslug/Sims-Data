import numpy as np
from paramLoader import Parameters, Magic

def calcDrag(heading, speed, airDensity=1.230):
    return  0.5 * airDensity * Parameters["dragCoeffAreaCombo"] * speed**2

def calcDownForce(heading, speed, parameters):
    return np.asarray([0,0,0,0], dtype=float)
