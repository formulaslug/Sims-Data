import numpy as np
from numpy.typing import NDArray
from paramLoader import *

def calcDrag(worldArray:NDArray[np.float64], step:int) -> np.float64:
    return  0.5 * Parameters["airDensity"] * Parameters["dragCoeffAreaCombo"] * worldArray[step-1, varSpeed]**2

#TODO: Implement downforce model. Could be lookup table based or physics based.
def calcDownForce(worldArray:NDArray[np.float64], step:int) -> NDArray[np.float64]:
    return np.asarray([0,0,0,0], dtype=np.float64)
