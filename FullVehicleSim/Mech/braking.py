from Mech import brakepadFrictionModel
from paramLoader import Parameters, Magic
import numpy as np
from state import VehicleState
# Docs:
# https://docs.google.com/document/d/1oGsGDnY0DEKWpE3S6481A9yZ0F9qUEwWkSXJwTSz4E4/edit?tab=t.2rmbsj26c7w
# The goal of these functions are to calculate the net force on the brakes, applied reverse to heading

def brakePSI_toNewtons(psi:float) -> float:
    return psi * Parameters["brakeCaliperArea"] * 4.448222 # lb force to Newtons

def calcBrakeForce(prevWorld:VehicleState, inputs) -> float:
    """
    Calculate the brake force.

    FrictionCoeff(temp) * maxBrakeForce * 4 (for 4 wheels)
    
    :param prevWorld: World State Previous
    :return: Brake Force
    """
    frontBrakePSI = inputs[1]
    rearBrakePSI = inputs[2]
    frontBrakeForce = brakePSI_toNewtons(frontBrakePSI)
    rearBrakeForce = brakePSI_toNewtons(rearBrakePSI)

    # Calculate Brake Force
    frontBrakeForce = brakepadFrictionModel.calcFrictionCoeff(prevWorld.brakeTemperature) * Parameters["maxBrakeForce"] * 4
    return brakeForce

def calcBrakeCooling(prevWorld:VehicleState, inputs) -> float:
    """
    Calculate the cooled brake temperature.
    
    :param prevWorld: World State
    :return: Change in Temperature
    """
    frontBrakePSI = inputs[1]
    rearBrakePSI = inputs[2]
    return Parameters["ambientTemperature"] + (prevWorld.brakeTemperature - Parameters["ambientTemperature"]) * np.e ** (-1 / Parameters["stepsPerSecond"]/50.2)
    #q = (initTemperature - parameters["ambientTemperature"]) * parameters["brakeMass"] * parameters["brakeSpecificHeatCapacity"]
    #change = (q * parameters["brakepadThickness"])/(initTemperature * parameters["brakeThermalConductivity"] * parameters["brakeSurfaceArea"]
    #return initTemperature - change

def calcBrakeHeating(prevWorld:VehicleState, inputs) -> float:
    # Calculate Brake Force
    brakeForce = calcBrakeForce(prevWorld)
    # Guess energy increase
    speedChange = brakeForce / Parameters["Mass"] / Parameters["stepsPerSecond"] # momentum impulse
    energyChange = 0.5 * Parameters["Mass"] * (prevWorld.speed - (prevWorld.speed - speedChange))
    tempChange = energyChange/(Parameters["brakeMass"] * Parameters["brakeSpecificHeatCapacity"])
    return tempChange

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
