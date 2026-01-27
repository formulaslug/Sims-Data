from Mech import tireState as tire
from paramLoader import Parameters, Magic

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
