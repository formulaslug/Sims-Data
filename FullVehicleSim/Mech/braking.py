from Mech import brakepadFrictionModel
from paramLoader import *
import numpy as np
from numpy.typing import NDArray
import polars as pl
# Docs:
# https://docs.google.com/document/d/1oGsGDnY0DEKWpE3S6481A9yZ0F9qUEwWkSXJwTSz4E4/edit?tab=t.2rmbsj26c7w
# The goal of these functions are to calculate the net force on the brakes, applied reverse to heading

try:
    brakeConvectionData = pl.read_csv("Mech/brakeThermalTransfer.csv")
except Exception as e:
    print("Error loading brake thermal transfer data from Mech/brakeThermalTransfer.csv:", e)
    raise e
brakeAirSpeedData, brakeCoeffData = brakeConvectionData["airSpeed"].to_numpy(), brakeConvectionData["coeff"].to_numpy()

def brakeTransferCoeff(speed:float) -> float:
    # Interpolate the heat transfer coefficient based on air speed
    return np.interp(speed, brakeAirSpeedData, brakeCoeffData) #type: ignore Function says it returns an array but it just returns a scalar

def brakePSI_toNewtons(psi:float) -> float:
    return psi * Parameters["brakeCaliperArea"] * 2 * 4.448222 # lb force to Newtons, 2 for 2 sides of a caliper per disc

def calcBrakeForce(worldArray:NDArray[np.float64], step:int) -> tuple[float,float]:
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
    # Factor of 2 for 2 brakes on front and 2 on rear
    frontBrakeForce:float = brakepadFrictionModel.calcFrictionCoeff(worldArray[step-1, varFrontBrakeTemperature]) * frontBrakeForce * 2 * Parameters["brakeDiscRadius"] / Parameters["wheelRadius"]
    rearBrakeForce:float = brakepadFrictionModel.calcFrictionCoeff(worldArray[step-1, varRearBrakeTemperature]) * rearBrakeForce * 2 * Parameters["brakeDiscRadius"] / Parameters["wheelRadius"]
    return frontBrakeForce, rearBrakeForce

def calcBrakeCooling(worldArray:NDArray[np.float64], step:int) -> tuple[float,float]:
    """
    Calculate the cooled brake temperature.
    
    :param prevWorld: World State
    :return: Change in Temperature
    """
    speed = worldArray[step-1, varSpeed]
    heatTransferCoeff = brakeTransferCoeff(speed)
    brakeThermalMass = Parameters["4130SpecificHeatCapacity"] * Parameters["brakeRotorMass"]
    brakeCoolingArea = Parameters["brakeRotorArea"]
    frontBrakeCoolingEnergy = heatTransferCoeff * brakeCoolingArea *(worldArray[step-1, varFrontBrakeTemperature] - Parameters["ambientTemperature"])
    frontBrakeCooling = frontBrakeCoolingEnergy / brakeThermalMass / Parameters["stepsPerSecond"]
    
    rearBrakeCoolingEnergy = heatTransferCoeff * brakeCoolingArea *(worldArray[step-1, varRearBrakeTemperature] - Parameters["ambientTemperature"])
    rearBrakeCooling = rearBrakeCoolingEnergy / brakeThermalMass / Parameters["stepsPerSecond"]

    return frontBrakeCooling, rearBrakeCooling
    #q = (initTemperature - parameters["ambientTemperature"]) * parameters["brakeMass"] * parameters["brakeSpecificHeatCapacity"]
    #change = (q * parameters["brakepadThickness"])/(initTemperature * parameters["brakeThermalConductivity"] * parameters["brakeSurfaceArea"]
    #return initTemperature - change

def calcBrakeHeating(worldArray:NDArray[np.float64], step:int) -> tuple[float,float]:
    # Calculate Brake Force
    frontBrakeForce, rearBrakeForce = calcBrakeForce(worldArray, step)
    # Guess energy increase based on kinetic energy decrease of the vehicle.
    # Assumption is 100% of kinetic energy lost goes into brake heating.
    speedChange = (frontBrakeForce + rearBrakeForce) / Parameters["Mass"] / Parameters["stepsPerSecond"] # momentum impulse
    energyChange = 0.5 * Parameters["Mass"] * (worldArray[step-1, varSpeed]**2 - ((worldArray[step-1, varSpeed] - speedChange)**2))
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
