import numpy as np
from paramLoader import *

def calcMaxMotorTorque(worldArray:np.ndarray, step:int, resistiveForces:float, maxPower:float, maxTractionTorqueAtWheel:float):
        '''
        Motor Torque at the wheel
        
        minimum(rpm limited torque, power limited torque, perfect traction torque)
        '''
        ## RPM Limited Torque (Motor Controller limits it to ~ this in practice. Maybe something more like 7490ish)
        if worldArray[step-1, varMotorRPM] > 7490:
            return -1 * resistiveForces * Parameters["wheelRadius"]
        if worldArray[step-1, varMotorRotationsHZ] != 0: ## If rolling, torque may be power limited. 
            maxPowerTorque = maxPower / worldArray[step-1, varMotorRotationsHZ] * Parameters["gearRatio"]
        else: ## Avoid divide by 0 error but it's just the same as the max torque that the motor can deliver (180 Nm)
            maxPowerTorque = 180.0 # Nm at 0 rpm
        torque = min(Parameters["maxTorque"], maxPowerTorque, maxTractionTorqueAtWheel/Parameters["gearRatio"])
        return torque

def calcCurrent(power:float, voltage:float) -> float:
        if (power / voltage) > Parameters["tractiveIMax"]:
            return Parameters["tractiveIMax"]
        return power / voltage

def calcMaxWheelTorque(maxMotorTorque):
        '''
        maxMotorTorque * gear rato
        '''
        return maxMotorTorque * Parameters["gearRatio"]

def calcMotorForce(maxWheelTorque:float) -> float:
        return (maxWheelTorque / Parameters["wheelRadius"])

def calcMaxPower(voltage:float) -> float:
        return Parameters["tractiveIMax"] * voltage

def calcVoltage():
        # return 28.0 * lookup(self.charge, self.lastCurrent)
        return 120.0 # Placeholder voltage. Will be a function of SOC, Temp, and Current Histeresis