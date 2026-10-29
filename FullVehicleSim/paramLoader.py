import json5
from typing import Dict, List, Tuple
import polars as pl
import numpy as np

Magic: dict
Parameters: dict
with open('params.json5', 'r') as file:
    params = json5.load(file)
    Magic = params["Magic"] #type: ignore
    Parameters = params["Parameters"] #type: ignore
    del params

savedHisteresisKernel = pl.read_csv("Electrical/HisteresisCellModel/trained_voltage_kernel.csv").to_numpy()
kernelStepSize = Magic["cellModel_KernelStepSize"]
newKernelLen = int(savedHisteresisKernel.shape[0] * Parameters["stepsPerSecond"] * kernelStepSize)
histeresisKernel = np.interp(
    np.linspace(0, savedHisteresisKernel.shape[0] * kernelStepSize, newKernelLen, endpoint=False),
    np.linspace(0, savedHisteresisKernel.shape[0] * kernelStepSize, savedHisteresisKernel.shape[0], endpoint=False),
    savedHisteresisKernel[:, 1]
)
 
## IMPORTANT: Do not name any other variable that starts with "var" in this file, as it will be included in the variable schema.
# Variable definitions - maintain original order for compatibility

varTime = 0
varThrottle = 1
varBrakePressureFront = 2
varBrakePressureRear = 3
varBrakePedalTravel = 4
varSteerAngle = 5
varPosX = 6
varPosY = 7
varPosZ = 8
varVelX = 9
varVelY = 10
varVelZ = 11
varSpeed = 12
varHeadingX = 13
varHeadingY = 14
varHeadingZ = 15
varYawRate = 16
varFrontBrakeTemperature = 17
varRearBrakeTemperature = 18
varCharge = 19
varDrag = 20
varResistiveForces = 21
varMotorTorque = 22
varMotorForce = 23
varNetForce = 24
varMaxTraction = 25
varWheelRotationsHZ = 26
varMotorRPM = 27
varMotorRotationsHZ = 28
varCurrent = 29
varMaxWheelTorque = 30
varMaxPower = 31
varPower = 32
varVoltage = 33
varFrontBrakeForce = 34
varRearBrakeForce = 35
varFrontBrakeHeating = 36
varRearBrakeHeating = 37
varFrontBrakeCooling = 38
varRearBrakeCooling = 39
varFrontSlipAngle = 40
varRearSlipAngle = 41
varMaxMotorTorque = 42
varAcceleration = 43
varWheelRPM = 44

# Automatically generate schema from defined variables
def generate_variable_schema() -> Dict[int, str]:
    """
    Generate a schema mapping variable indices to their names.
    Preserves the order of definition in the file.
    """
    schema = {}
    
    # Get all variables that start with 'var' from the current module
    current_module = globals()
    var_items = [(name, value) for name, value in current_module.items() 
                 if name.startswith('var') and isinstance(value, int)]
    
    # Sort by value to maintain order
    var_items.sort(key=lambda x: x[1])
    
    # Create the schema
    for name, index in var_items:
        # Convert variable name to readable format
        readable_name = name[3].lower() + name[4:]  # Remove 'var' prefix and lowercase first letter
        schema[index] = readable_name
    
    return schema

def get_variable_names() -> List[str]:
    """
    Get ordered list of variable names (without 'var' prefix).
    """
    schema = generate_variable_schema()
    return [schema[i] for i in range(len(schema))]

def get_variable_mapping() -> Dict[str, int]:
    """
    Get mapping from variable names to indices.
    """
    schema = generate_variable_schema()
    return {name: index for index, name in schema.items()}

# Generate the schema on module load
VARIABLE_SCHEMA = generate_variable_schema()
VARIABLE_NAMES = get_variable_names()
VARIABLE_MAPPING = get_variable_mapping()
print("Parameters loaded...")
