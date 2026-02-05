from Mech import brakepadFrictionModel
from paramLoader import *
import numpy as np
# Docs:
# https://docs.google.com/document/d/1oGsGDnY0DEKWpE3S6481A9yZ0F9qUEwWkSXJwTSz4E4/edit?tab=t.2rmbsj26c7w
# The goal of these functions are to calculate the net force on the brakes, applied reverse to heading

def brakePSI_toNewtons(psi:float) -> float:
    return psi * Parameters["brakeCaliperArea"] * 4.448222 # lb force to Newtons

def calcBrakeForce(worldArray:np.ndarray, step:int) -> tuple[float,float]:
    """
    Calculate the brake force.

    FrictionCoeff(temp) * maxBrakeForce * 4 (for 4 wheels)
    
    :param worldArray: World State Array
    :param step: Current step index
    :return: Brake Force
    """
    frontBrakePSI = worldArray[step, varBrakePressureFront]
    rearBrakePSI = worldArray[step, varBrakePressureRear]
    frontBrakeForce = brakePSI_toNewtons(frontBrakePSI)
    rearBrakeForce = brakePSI_toNewtons(rearBrakePSI)

    # Calculate Brake Force
    frontBrakeForce:float = brakepadFrictionModel.calcFrictionCoeff(worldArray[step-1, varFrontBrakeTemperature]) * frontBrakeForce * 2 * Parameters["brakeDiscRadius"] / Parameters["wheelRadius"]
    rearBrakeForce:float = brakepadFrictionModel.calcFrictionCoeff(worldArray[step-1, varRearBrakeTemperature]) * rearBrakeForce * 2 * Parameters["brakeDiscRadius"] / Parameters["wheelRadius"]
    return frontBrakeForce, rearBrakeForce

def calcBrakeCooling(worldArray:np.ndarray, step:int) -> tuple[float,float]:
    """
    Calculate the cooled brake temperature.
    
    :param prevWorld: World State
    :return: Change in Temperature
    """
    frontBrakeCooling = Parameters["ambientTemperature"] + (worldArray[step-1, varFrontBrakeTemperature] - Parameters["ambientTemperature"]) * np.e ** (-1 / Parameters["stepsPerSecond"]/50.2)
    rearBrakeCooling = Parameters["ambientTemperature"] + (worldArray[step-1, varRearBrakeTemperature] - Parameters["ambientTemperature"]) * np.e ** (-1 / Parameters["stepsPerSecond"]/50.2)
    return frontBrakeCooling, rearBrakeCooling
    #q = (initTemperature - parameters["ambientTemperature"]) * parameters["brakeMass"] * parameters["brakeSpecificHeatCapacity"]
    #change = (q * parameters["brakepadThickness"])/(initTemperature * parameters["brakeThermalConductivity"] * parameters["brakeSurfaceArea"]
    #return initTemperature - change

def calcBrakeHeating(worldArray:np.ndarray, step:int) -> tuple[float,float]:
    # Calculate Brake Force
    frontBrakeForce, rearBrakeForce = calcBrakeForce(worldArray, step)
    # Guess energy increase based on kinetic energy decrease of the vehicle.
    # Assumption is 100% of kinetic energy lost goes into brake heating.
    speedChange = (frontBrakeForce + rearBrakeForce) / Parameters["Mass"] / Parameters["stepsPerSecond"] # momentum impulse
    energyChange = 0.5 * Parameters["Mass"] * (worldArray[step-1, varSpeed] - (worldArray[step-1, varSpeed] - speedChange))
    tempChange = energyChange/(Parameters["brakeMass"] * Parameters["brakeSpecificHeatCapacity"])

    # While this doesn't seem physically intuitive, it is based on the idea that the front and rear brakes share heat based on their contribution to total braking force.
    if frontBrakeForce + rearBrakeForce < 1e-6:
        return 0.0, 0.0
    frontTempChange = frontBrakeForce / (frontBrakeForce + rearBrakeForce) * tempChange
    rearTempChange = rearBrakeForce / (frontBrakeForce + rearBrakeForce) * tempChange
    return frontTempChange, rearTempChange

# def calcBrakeTemp(prevWorld:VehicleState) -> float:
#     """
#     Calculate Brake Temp
    
#     :param prevWorld: World State
#     :return: New Brake Temperature
#     """
#     # Calculate Brake Force
#     brakeForce = calcBrakeForce(prevWorld)
#     # Guess energy increase
#     speedChange = brakeForce / Parameters["Mass"] / Parameters["stepsPerSecond"] # momentum impulse
#     energyChange = 0.5 * Parameters["Mass"] * (prevWorld.speed - (prevWorld.speed - speedChange))
#     # Guess temperature increase
#     brakeTemperature = prevWorld.brakeTemperature + energyChange/(Parameters["brakeMass"] * Parameters["brakeSpecificHeatCapacity"])
#     return brakeTemperature
