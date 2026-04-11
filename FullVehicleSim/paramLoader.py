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
varMaxMotorTorque = 41
varAcceleration = 42
varWheelRPM = 43
varLatVel = 44

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
