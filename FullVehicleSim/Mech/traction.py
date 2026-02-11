from Mech import tireState as tire
from paramLoader import Parameters, Magic
from Mech.tireLoad import calcLoadTransfer
from Mech.steering import calcSlipAngle
import numpy as np

def calcTraction(tireLoad:tuple[float,float,float,float], slipAngle:tuple[float,float], slipRatio:float, speed, surfaceTemperature, tirePressure):
    frontLeft = tire.Tire(tireLoad[0] , 0.15, slipAngle[0], speed, 80, 40)
    frontRight = tire.Tire(tireLoad[1] , 0.15, slipAngle[0], speed, 80, 40)
    backLeft = tire.Tire(tireLoad[2] , 0.15, slipAngle[1], speed, 80, 40)
    backRight = tire.Tire(tireLoad[3] , 0.15, slipAngle[1], speed, 80, 40)
    return [(frontLeft.getLongForce(), frontLeft.getLateralForce() * 0.6),
        (frontRight.getLongForce() * 0.6, frontRight.getLateralForce() * 0.6),
        (backLeft.getLongForce() * 0.6, backLeft.getLateralForce() * 0.6),
        (backRight.getLongForce() * 0.6, backRight.getLateralForce() * 0.6)]

def calcCorneringStiffness(tireLoad:tuple[float,float,float,float], slipAngle:tuple[float,float], slipRatio, speed, surfaceTemperature, tirePressure):
    """
    Calculate the cornering stiffness of the vehicle at the current state using a Daniel's patented sketchy derivatives 
    
    :param tireLoad: Description
    :type tireLoad: tuple[float, float, float, float]
    :param slipAngle: Description
    :type slipAngle: tuple[float, float]
    :param slipRatio: Description
    :param speed: Description
    :param surfaceTemperature: Description
    :param tirePressure: Description
    """
    delta = 0.1
    less = calcTraction(tireLoad, tuple(x - delta for x in slipAngle), slipRatio, speed, surfaceTemperature, tirePressure) # type: ignore
    more = calcTraction(tireLoad, tuple(x + delta for x in slipAngle), slipRatio, speed, surfaceTemperature, tirePressure) # type: ignore

    front = ((more[0][1] + more[1][1]) - (less[0][1] + less[1][1])) / (2 * delta)
    rear = ((more[2][1] + more[3][1]) - (less[2][1] + less[3][1])) / (2 * delta)

    return (front, rear)

def maxTraction(initAcceleration:float, heading:np.ndarray, initYawRate:float, velocity:np.ndarray, steerAngle:float, speed:float):
        """Calculate the maximum traction available for the vehicle at the current state.
        This function computes the total traction magnitude by calculating tire loads,
        slip angles, and individual tire tractions, then combining them into a resultant
        traction vector.
        heading : np.ndarray
            Unit heading vector of the vehicle [x, y] components.
            Initial yaw rate of the vehicle before this time step, in rad/s.
            The velocity vector of the vehicle, in m/s.
            The steering angle of the vehicle, in radians.
            The speed of the vehicle, in m/s.
        Returns
        -------
        np.float32
            The magnitude of the maximum available traction force, in Newtons.
        Notes
        -----
        Yaw velocity is currently set to 0 in tire load calculations.
        Slip ratio is fixed at 0.15.
        """
        tireLoad = calcLoadTransfer(Parameters, initAcceleration * heading[0], initAcceleration * heading[1], initYawRate) # yaw velocity is currently set to 0

        slipAngle = calcSlipAngle(initYawRate, velocity, steerAngle, Parameters)
        slipRatio = 0.15
        tireTraction = calcTraction(tireLoad, slipAngle, slipRatio, speed, 80, 40, Parameters, Magic)
        longTraction = 0
        latTraction = 0
        for x, y in tireTraction:
            longTraction += x
            latTraction += y
        return np.sqrt(longTraction**2 + latTraction**2)
    
        #tempTire = tire.Tire(500 , 0.15, 0, self.speed, 80, 40, Parameters, Magic)
        #return  ((tempTire.getLongForce()/500 * self.weight * 0.7477)/1.6547084)/(1.0-(0.247718 * tempTire.getLongForce()/500 / 1.6547084))
