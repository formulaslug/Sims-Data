import numpy as np
from paramLoader import Parameters, Magic
from state import VehicleState

def calcDrag(prevWorld:VehicleState) -> float:
    return  0.5 * Parameters["airDensity"] * Parameters["dragCoeffAreaCombo"] * prevWorld.speed**2

def calcDownForce(prevWorld:VehicleState) -> np.ndarray:
    return np.asarray([0,0,0,0], dtype=float)
