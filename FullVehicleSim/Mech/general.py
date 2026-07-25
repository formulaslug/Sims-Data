import numpy as np
from paramLoader import *
from numpy.typing import NDArray
from Mech.traction import calcCorneringStiffness
from Mech.braking import calcBrakeForce
from Mech.aero import calcDrag
from Mech.steering import calcSlipAngle, calcYawRate
from Mech.tireLoad import calcLoadTransfer

def calcResistiveForces(worldArray:NDArray[np.float64], step:int) -> np.float64:
        if worldArray[step-1, varSpeed] <= 1e-5: # Floating point error
            return np.float64(0)
        else:
            frontBrakeForce, rearBrakeForce = calcBrakeForce(worldArray, step)
            return -1 * (calcDrag(worldArray, step) + frontBrakeForce + rearBrakeForce)
        
def calculateYawRate(worldArray:NDArray[np.float64], step:int, initAcceleration:np.float64, initYawRate:np.float64, timeSinceLastSteer:np.float64) -> np.float64:
        """Calculate the yaw rate of the vehicle at the current state.
        This function computes the yaw rate by calculating tire loads, slip angles,
        cornering stiffness, and then applying the vehicle dynamics equations.
        heading : np.ndarray
            Unit heading vector of the vehicle [x, y] components.
            Initial yaw rate of the vehicle before this time step, in rad/s.
            The velocity vector of the vehicle, in m/s.
            The steering angle of the vehicle, in radians.
            The speed of the vehicle, in m/s.
        Returns
        -------
        float
            The yaw rate of the vehicle, in rad/s.
        Notes
        -----
        Slip ratio is fixed at 0.15.
        """
        tireLoad = calcLoadTransfer(initAcceleration * worldArray[step-1, varHeadingX], initAcceleration * worldArray[step-1, varHeadingY], initYawRate)
        slipAngle = calcSlipAngle(worldArray, step)
        slipRatio = 0.15
        corneringStiffness = calcCorneringStiffness(tireLoad, slipAngle, slipRatio, worldArray[step-1, varSpeed], 80, 40, Parameters, Magic) # Works but unused
        res:np.float64 = calcYawRate(initYawRate, worldArray[step-1, varSpeed], worldArray[step, varSteerAngle], timeSinceLastSteer, corneringStiffness[0], corneringStiffness[1])
        return res
