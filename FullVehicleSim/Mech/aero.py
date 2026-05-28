import numpy as np
from numpy import ndarray
from paramLoader import *

def calcDrag(worldArray:ndarray[np.float64], step:int) -> float:
    return  0.5 * Parameters["airDensity"] * Parameters["dragCoeffAreaCombo"] * worldArray[step-1, varSpeed]**2

def calcDownForce(worldArray:ndarray[np.float64], step:int) -> np.ndarray:
    return np.asarray([0,0,0,0], dtype=float)
