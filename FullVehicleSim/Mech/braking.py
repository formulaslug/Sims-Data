from Mech import brakepadFrictionModel
from paramLoader import Parameters, Magic
import numpy as np
from FullVehicleSim.state import VehicleState
# Docs:
# https://docs.google.com/document/d/1oGsGDnY0DEKWpE3S6481A9yZ0F9qUEwWkSXJwTSz4E4/edit?tab=t.2rmbsj26c7w
# The goal of these functions are to calculate the net force on the brakes, applied reverse to heading

def calcBrakeForce(prevWorld:VehicleState) -> float:
    """
    Calculate the brake force.

    FrictionCoeff(temp) * maxBrakeForce * 4 (for 4 wheels)
    
    :param prevWorld: World State Previous
    :param parameters: Parameter dictionary
    :return: Brake Force
    """
    # Calculate Brake Force
    brakeForce = brakepadFrictionModel.calcFrictionCoeff(prevWorld.brakeTemperature) * Parameters["maxBrakeForce"] * 4
    return brakeForce

def calcBrakeTemp(prevWorld:VehicleState) -> float:
    """
    Calculate Brake Temp
    
    :param prevWorld: World State Previous
    :param parameters: Parameter dictionary
    :return: New Brake Temperature
    """
    # Calculate Brake Force
    brakeForce = calcBrakeForce(prevWorld)
    # Guess energy increase
    speedChange = brakeForce / Parameters["Mass"] / Parameters["stepsPerSecond"] # momentum impulse
    energyChange = 0.5 * Parameters["Mass"] * (prevWorld.speed - (prevWorld.speed - speedChange))
    # Guess temperature increase
    brakeTemperature = prevWorld.brakeTemperature + energyChange/(Parameters["brakeMass"] * Parameters["brakeSpecificHeatCapacity"])
    return brakeTemperature

def calcBrakeCooling(prevWorld:VehicleState) -> float:
    """
    Calculate the cooled brake temperature.
    
    :param previousBrakeTemperature: Description
    :param parameters: Description
    :return: New Brake Temperature
    """
    return Parameters["ambientTemperature"] + (prevWorld.brakeTemperature - Parameters["ambientTemperature"]) * np.e ** (-1 / Parameters["stepsPerSecond"]/50.2)
    #q = (initTemperature - parameters["ambientTemperature"]) * parameters["brakeMass"] * parameters["brakeSpecificHeatCapacity"]
    #change = (q * parameters["brakepadThickness"])/(initTemperature * parameters["brakeThermalConductivity"] * parameters["brakeSurfaceArea"]
    #return initTemperature - change
