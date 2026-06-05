import numpy as np
from paramLoader import *

def calcDrag(worldArray:np.ndarray, step:int) -> float:
    return  0.5 * Parameters["airDensity"] * Parameters["dragCoeffAreaCombo"] * worldArray[step-1, varSpeed]**2

def calcDownForce(worldArray:np.ndarray, step:int) -> np.ndarray:
    return np.asarray([0,0,0,0], dtype=float)
