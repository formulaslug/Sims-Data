import numpy as np
from numpy.typing import NDArray
from paramLoader import Parameters, Magic
from Mech import tireState as tire
from Mech.tireLoad import calcLoadTransfer
from Mech.steering import calcSlipAngle

def calcTraction(tireLoad:tuple[np.float64,np.float64,np.float64,np.float64], slipAngle:tuple[np.float64,np.float64], 
                 slipRatio:np.float64, speed:np.float64, surfaceTemperature:np.float64, tirePressure:np.float64) -> NDArray[np.float64]:
    frontLeft = tire.Tire(tireLoad[0] , 0.15, slipAngle[0], speed, 80, 40)
    frontRight = tire.Tire(tireLoad[1] , 0.15, slipAngle[0], speed, 80, 40)
    backLeft = tire.Tire(tireLoad[2] , 0.15, slipAngle[1], speed, 80, 40)
    backRight = tire.Tire(tireLoad[3] , 0.15, slipAngle[1], speed, 80, 40)
    return np.array([[frontLeft.getLongForce(), frontLeft.getLateralForce() * 0.6],
        [frontRight.getLongForce() * 0.6, frontRight.getLateralForce() * 0.6],
        [backLeft.getLongForce() * 0.6, backLeft.getLateralForce() * 0.6],
        [backRight.getLongForce() * 0.6, backRight.getLateralForce() * 0.6]])
def calcCorneringStiffness(tireLoad:tuple[np.float64,np.float64,np.float64,np.float64], slipAngle:tuple[np.float64,np.float64], 
                           slipRatio:np.float64, speed:np.float64, surfaceTemperature:np.float64, tirePressure:np.float64) -> tuple[np.float64, np.float64]:
    """
    Calculate the cornering stiffness of the vehicle at the current state using a Daniel's patented sketchy derivatives 
    
    :param tireLoad: Loads on each tire
    :param slipAngle: Slip Angles (LR? FB?)
    :param slipRatio: Vehicle slip ratio
    :param speed: Vehicle Speed
    :param surfaceTemperature: Tire surface temperature simplified such that all tires have the same pressure.
    :param tirePressure: Tire pressure simplified such that all tires have the same pressure.
    """
    delta = 1 / Parameters["stepsPerSecond"]
    less = calcTraction(tireLoad, tuple(x - delta for x in slipAngle), slipRatio, speed, surfaceTemperature, tirePressure) # type: ignore
    more = calcTraction(tireLoad, tuple(x + delta for x in slipAngle), slipRatio, speed, surfaceTemperature, tirePressure) # type: ignore

    front = ((more[0][1] + more[1][1]) - (less[0][1] + less[1][1])) / (2 * delta)
    rear = ((more[2][1] + more[3][1]) - (less[2][1] + less[3][1])) / (2 * delta)

    return (front, rear)

def maxTraction(initAcceleration:np.float64, heading:np.ndarray, initYawRate:np.float64, velocity:NDArray[np.float64], steerAngle:np.float64, speed:np.float64):
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
    np.float64
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
