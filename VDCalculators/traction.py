from Mech import tireState as tire

def getTraction(tireLoad, slipAngle, slipRatio, speed, surfaceTemperature, tirePressure, Parameters, Magic):
    frontLeft = tire.Tire(tireLoad[0] , 0.15, slipAngle[0], speed, 80, 40, Parameters, Magic)
    frontRight = tire.Tire(tireLoad[1] , 0.15, slipAngle[0], speed, 80, 40, Parameters, Magic)
    backLeft = tire.Tire(tireLoad[2] , 0.15, slipAngle[1], speed, 80, 40, Parameters, Magic)
    backRight = tire.Tire(tireLoad[3] , 0.15, slipAngle[1], speed, 80, 40, Parameters, Magic)
    return [(frontLeft.getLongForce(), frontLeft.getLateralForce() * 0.6),
        (frontRight.getLongForce() * 0.6, frontRight.getLateralForce() * 0.6),
        (backLeft.getLongForce() * 0.6, backLeft.getLateralForce() * 0.6),
        (backRight.getLongForce() * 0.6, backRight.getLateralForce() * 0.6)]

def getCorneringStiffness(tireLoad, slipAngle, slipRatio, speed, surfaceTemperature, tirePressure, Parameters, Magic):
    delta = 0.1
    less = getTraction(tireLoad, tuple(x - delta for x in slipAngle), slipRatio, speed, surfaceTemperature, tirePressure, Parameters, Magic)
    more = getTraction(tireLoad, tuple(x + delta for x in slipAngle), slipRatio, speed, surfaceTemperature, tirePressure, Parameters, Magic)

    front = ((more[0][1] + more[1][1]) - (less[0][1] + less[1][1])) / (2 * delta)
    rear = ((more[2][1] + more[3][1]) - (less[2][1] + less[3][1])) / (2 * delta)

    return (front, rear)
