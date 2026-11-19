import numpy as np
from paramLoader import *
from units import P, get, to_column

def calcDrag(worldArray:np.ndarray, step:int) -> float:
    speed = get(worldArray, step-1, varSpeed)
    drag = 0.5 * P["airDensity"] * P["dragCoeffAreaCombo"] * speed**2
    return to_column(drag, varDrag)

def calcDownForce(worldArray:np.ndarray, step:int) -> np.ndarray:
    return np.asarray([0,0,0,0], dtype=float)
