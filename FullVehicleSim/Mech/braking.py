from Mech import brakepadFrictionModel
import numpy as np
# Docs:
# https://docs.google.com/document/d/1oGsGDnY0DEKWpE3S6481A9yZ0F9qUEwWkSXJwTSz4E4/edit?tab=t.2rmbsj26c7w
# The goal of this function is to calculate the net force on the brakes, applied reverse to heading
def getBrakeForce(speed, initTemperature, stepSize, parameters):
    # Calculate Brake Force
    brakeForce = brakepadFrictionModel.getFriction(initTemperature) * parameters["maxBrakeForce"] * 4
    # Guess energy increase
    speedChange = brakeForce / parameters["Mass"] * stepSize # momentum impulse
    energyChange = 0.5 * parameters["Mass"] * (speed - (speed - speedChange))
    # Guess temperature increase
    brakeTemperature = initTemperature + energyChange/(parameters["brakeMass"] * parameters["brakeSpecificHeatCapacity"])
    return brakeForce, brakeTemperature

def calculateBrakeCooling(initTemperature, stepSize, parameters):
    return parameters["ambientTemperature"] + (initTemperature - parameters["ambientTemperature"]) * np.e ** (-1 * stepSize/50.2)

    #q = (initTemperature - parameters["ambientTemperature"]) * parameters["brakeMass"] * parameters["brakeSpecificHeatCapacity"]
    #change = (q * parameters["brakepadThickness"])/(initTemperature * parameters["brakeThermalConductivity"] * parameters["brakeSurfaceArea"]
    #return initTemperature - change
