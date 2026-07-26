from Mech import brakepadFrictionModel
from paramLoader import *
import numpy as np
from numpy.typing import NDArray
import polars as pl
# Docs:
# https://docs.google.com/document/d/1oGsGDnY0DEKWpE3S6481A9yZ0F9qUEwWkSXJwTSz4E4/edit?tab=t.2rmbsj26c7w
# The goal of these functions are to calculate the net force on the brakes, applied reverse to heading

brakeThermalTransferDataPath = "Mech/brakeThermalTransfer.csv"
try:
    brakeConvectionData = pl.read_csv(brakeThermalTransferDataPath)
except Exception as e:
    print(f"Error loading brake thermal transfer data from {brakeThermalTransferDataPath}:", e)
    raise e
brakeAirSpeedData, brakeCoeffData = brakeConvectionData["airSpeed"].to_numpy(), brakeConvectionData["coeff"].to_numpy()

def brakeTransferCoeff(speed:np.float64) -> np.float64:
    # Interpolate the heat transfer coefficient based on air speed
    return np.interp(speed, brakeAirSpeedData, brakeCoeffData) #type: ignore Function says it returns an array but it just returns a scalar

def brakePSI_toNewtons(psi:np.float64) -> np.float64:
    return psi * Parameters["brakeCaliperArea"] * 2 * 4.448222 # lb force to Newtons, 2 for 2 sides of a caliper per disc

def calcBrakeForce(worldArray:NDArray[np.float64], step:int) -> tuple[np.float64,np.float64]:
    """
    Calculate the brake force.

    FrictionCoeff(temp) * maxBrakeForce * 4 (for 4 wheels)
    
    :param worldArray: World State Array
    :param step: Current step index
    :return: Brake Force
    """
    frontBrakePSI:np.float64 = worldArray[step, varBrakePressureFront]
    rearBrakePSI:np.float64 = worldArray[step, varBrakePressureRear]
    frontBrakeCaliperForce:np.float64 = brakePSI_toNewtons(frontBrakePSI)
    rearBrakeCaliperForce:np.float64 = brakePSI_toNewtons(rearBrakePSI)

    # Calculate Brake Force
    # Factor of 2 for 2 brakes on front and 2 on rear
    frontBrakeForce:np.float64 = brakepadFrictionModel.calcFrictionCoeff(worldArray[step-1, varFrontBrakeTemperature]) * frontBrakeCaliperForce * 2 * Parameters["brakeDiscRadius"] / Parameters["wheelRadius"]
    rearBrakeForce:np.float64 = brakepadFrictionModel.calcFrictionCoeff(worldArray[step-1, varRearBrakeTemperature]) * rearBrakeCaliperForce * 2 * Parameters["brakeDiscRadius"] / Parameters["wheelRadius"]
    return frontBrakeForce, rearBrakeForce

def calcBrakeCooling(worldArray:NDArray[np.float64], step:int) -> tuple[np.float64,np.float64]:
    """
    Calculate the change in brake temperature due to convective cooling. Uses data from step-1.
    This currently neglects conduction to nearby bits of metal like the inner rotor/hub, the brake lines, etc.
    
    :param worldArray: World Array
    :param step: Current Step
    :return: Change in Temperature
    """
    speed:np.float64 = worldArray[step-1, varSpeed]
    heatTransferCoeff:np.float64 = brakeTransferCoeff(speed)
    brakeThermalMass:np.float64 = Parameters["4130SpecificHeatCapacity"] * Parameters["brakeRotorMass"]
    brakeCoolingArea:np.float64 = Parameters["brakeRotorArea"]
    frontBrakeCoolingEnergy:np.float64 = heatTransferCoeff * brakeCoolingArea *(worldArray[step-1, varFrontBrakeTemperature] - Parameters["ambientTemperature"])
    frontBrakeCooling:np.float64 = frontBrakeCoolingEnergy / brakeThermalMass / Parameters["stepsPerSecond"]
    
    rearBrakeCoolingEnergy:np.float64 = heatTransferCoeff * brakeCoolingArea *(worldArray[step-1, varRearBrakeTemperature] - Parameters["ambientTemperature"])
    rearBrakeCooling:np.float64 = rearBrakeCoolingEnergy / brakeThermalMass / Parameters["stepsPerSecond"]

    return frontBrakeCooling, rearBrakeCooling
    #q = (initTemperature - parameters["ambientTemperature"]) * parameters["brakeMass"] * parameters["brakeSpecificHeatCapacity"]
    #change = (q * parameters["brakepadThickness"])/(initTemperature * parameters["brakeThermalConductivity"] * parameters["brakeSurfaceArea"]
    #return initTemperature - change

def calcBrakeHeating(worldArray:NDArray[np.float64], step:int) -> tuple[float,float]:
    """
    Calculate the change in brake temperature due to brake force heating. Uses data from step-1.
    
    :param worldArray: World Array
    :param step: Current Step
    :return: Change in Temperature
    """
    # Calculate Brake Force
    frontBrakeForce, rearBrakeForce = calcBrakeForce(worldArray, step)
    # Guess energy increase based on kinetic energy decrease of the vehicle.
    # Assumption is 100% of kinetic energy lost by the change in speed due to braking goes into brake heating. 
    # In reality, some of it is also going to heating the brake fluid although that is largely conductive. 
    speedChange = (frontBrakeForce + rearBrakeForce) / Parameters["mass"] / Parameters["stepsPerSecond"] # momentum impulse
    energyChange = 0.5 * Parameters["mass"] * (worldArray[step-1, varSpeed]**2 - ((worldArray[step-1, varSpeed] - speedChange)**2))
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
#     speedChange = brakeForce / Parameters["mass"] / Parameters["stepsPerSecond"] # momentum impulse
#     energyChange = 0.5 * Parameters["mass"] * (prevWorld.speed - (prevWorld.speed - speedChange))
#     # Guess temperature increase
#     brakeTemperature = prevWorld.brakeTemperature + energyChange/(Parameters["brakeMass"] * Parameters["brakeSpecificHeatCapacity"])
#     return brakeTemperature
