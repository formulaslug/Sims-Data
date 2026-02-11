from Mech.traction import calcCorneringStiffness
from paramLoader import Parameters, Magic
from state import VehicleState
from Mech.braking import calcBrakeForce
from Mech.aero import calcDrag
from Mech.steering import calcSlipAngle, calcYawRate
from Mech.tireLoad import calcLoadTransfer
import numpy as np

def calcResistiveForces(worldPrev:VehicleState, inputs):
        if worldPrev.speed <= 1e-5: # Floating point error
            return 0
        else:
            frontBrakeForce, rearBrakeForce = calcBrakeForce(worldPrev, inputs)
            return -1 * (calcDrag(worldPrev) + frontBrakeForce + rearBrakeForce)
        
def calculateYawRate(prevWorld:VehicleState, steerAngle:float, initAcceleration:float, heading:np.ndarray, initYawRate:float, timeSinceLastSteer:float):
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
        tireLoad = calcLoadTransfer(initAcceleration * heading[0], initAcceleration * heading[1], initYawRate)
        slipAngle = calcSlipAngle(initYawRate, prevWorld.velocity, steerAngle, Parameters)
        slipRatio = 0.15
        corneringStiffness = calcCorneringStiffness(tireLoad, slipAngle, slipRatio, prevWorld.speed, 80, 40, Parameters, Magic) # Works but unused
        res = calcYawRate(initYawRate, prevWorld.speed, steerAngle, timeSinceLastSteer, corneringStiffness[0], corneringStiffness[1], Parameters)
        return res