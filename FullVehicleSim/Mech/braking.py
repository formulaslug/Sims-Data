from Mech import brakepadFrictionModel
import numpy as np
# Docs:
# https://docs.google.com/document/d/1oGsGDnY0DEKWpE3S6481A9yZ0F9qUEwWkSXJwTSz4E4/edit?tab=t.2rmbsj26c7w
# The goal of this function is to calculate the net force on the brakes, applied reverse to heading
def getBrakeForceAndTemp(prevWorld, parameters):
    """
    Calculate the brake force and updated brake temperature.
    
    :param speed: Vehicle Speed
    :param previousBrakeTemperature: Previous Brake Temperature
    :param parameters: Parameter dictionary
    :return: Tuple of (brakeForce, brakeTemperature)
    """
    # Calculate Brake Force
    brakeForce = brakepadFrictionModel.getFriction(prevWorld.brakeTemperature) * parameters["maxBrakeForce"] * 4
    # Guess energy increase
    speedChange = brakeForce / parameters["Mass"] * parameters["stepSize"] # momentum impulse
    energyChange = 0.5 * parameters["Mass"] * (prevWorld.speed - (prevWorld.speed - speedChange))
    # Guess temperature increase
    brakeTemperature = prevWorld.brakeTemperature + energyChange/(parameters["brakeMass"] * parameters["brakeSpecificHeatCapacity"])
    return brakeForce, brakeTemperature

def calculateBrakeCooling(previousBrakeTemperature, parameters):
    """
    Calculate the cooled brake temperature.
    
    :param previousBrakeTemperature: Description
    :param parameters: Description
    :return: New Brake Temperature
    """
    return parameters["ambientTemperature"] + (previousBrakeTemperature - parameters["ambientTemperature"]) * np.e ** (-1 * parameters["stepSize"]/50.2)
    #q = (initTemperature - parameters["ambientTemperature"]) * parameters["brakeMass"] * parameters["brakeSpecificHeatCapacity"]
    #change = (q * parameters["brakepadThickness"])/(initTemperature * parameters["brakeThermalConductivity"] * parameters["brakeSurfaceArea"]
    #return initTemperature - change
