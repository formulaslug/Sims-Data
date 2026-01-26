import numpy as np
from paramLoader import Parameters, Magic

def calcDrag(prevWorld):
    return  0.5 * Parameters["airDensity"] * Parameters["dragCoeffAreaCombo"] * prevWorld.speed**2

def calcDownForce(prevWorld):
    return np.asarray([0,0,0,0], dtype=float)
