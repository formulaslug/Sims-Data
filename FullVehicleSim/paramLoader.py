import json5
Magic:dict
Parameters:dict
with open('params.json5', 'r') as file:
    params = json5.load(file)
    Magic = params["Magic"]
    Parameters = params["Parameters"]
    del params

varTime = 0
varThrottle = 1
varBrakePressureFront = 2
varBrakePressureRear = 3
varSteerAngle = 4
varPosX = 5
varPosY = 6
varPosZ = 7
varVelX = 8
varVelY = 9
varVelZ = 10
varSpeed = 11
varHeadingX = 12
varHeadingY = 13
varHeadingZ = 14
varYawRate = 15
varFrontBrakeTemperature = 16
varRearBrakeTemperature = 17
varCharge = 18
varDrag = 19
varResistiveForces = 20
varMotorTorque = 21
varMotorForce = 22
varNetForce = 23
varMaxTraction = 24
varWheelRotationsHZ = 25
varMotorRPM = 26
varMotorRotationsHZ = 27
varCurrent = 28
varMaxWheelTorque = 29
varMaxPower = 30
varPower = 31
varVoltage = 32
varFrontBrakeForce = 33
varRearBrakeForce = 34
varFrontBrakeHeating = 35
varRearBrakeHeating = 36
varFrontBrakeCooling = 37
varRearBrakeCooling = 38
varFrontSlipAngle = 39
varRearSlipAngle = 40
print("Parameters loaded...")
