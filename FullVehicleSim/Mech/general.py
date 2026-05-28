from Mech.traction import calcCorneringStiffness
from paramLoader import *
from Mech.braking import calcBrakeForce
from Mech.aero import calcDrag
from Mech.steering import calcSlipAngle, calcYawRate
from Mech.tireLoad import calcLoadTransfer
import numpy as np
from numpy import ndarray

def calcResistiveForces(worldArray:ndarray[np.float64], step:int):
        if worldArray[step-1, varSpeed] <= 1e-5: # Floating point error
            return 0
        else:
            frontBrakeForce, rearBrakeForce = calcBrakeForce(worldArray, step)
            return -1 * (calcDrag(worldArray, step) + frontBrakeForce + rearBrakeForce)
        
def calculateYawRate(worldArray:ndarray[np.float64], step:int, initAcceleration:float, initYawRate:float, timeSinceLastSteer:float):
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
        res = calcYawRate(initYawRate, worldArray[step-1, varSpeed], worldArray[step, varSteerAngle], timeSinceLastSteer, corneringStiffness[0], corneringStiffness[1])
        return res
