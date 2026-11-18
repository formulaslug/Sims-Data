import numpy as np

from ..paramLoader import *
from .aero import calcDrag
from .braking import calcBrakeForce


def calcRollingResistance(wheelLoads:np.ndarray) -> float:
    # F_rr = C_rr * sum(Fz), N. Load-based so downforce/load transfer is included.
    return Parameters["rollingResistanceCoeff"] * float(np.sum(wheelLoads))

def calcResistiveForces(worldArray:np.ndarray, step:int):
    # Drag + brakes + rolling resistance, N, negative = opposing motion (body x). None of them can push a stopped car.
    if worldArray[step-1, varSpeed] <= 1e-5: # Floating point error
        return 0
    frontBrakeForce, rearBrakeForce = calcBrakeForce(worldArray, step)
    rollingResistance = calcRollingResistance(worldArray[step-1, varFLWheelLoad:varBRWheelLoad+1])
    return -1 * (calcDrag(worldArray, step) + frontBrakeForce + rearBrakeForce + rollingResistance)
