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
            maxPowerTorque = maxPower / worldArray[step-1, varMotorRotationsHZ]
        else: ## Avoid divide by 0 error but it's just the same as the max torque that the motor can deliver (180 Nm)
            maxPowerTorque = Parameters["maxTorque"] # Nm at 0 rpm
        torque = min(Parameters["maxTorque"], maxPowerTorque, 2*maxTractionTorqueAtWheel/Parameters["gearRatio"])
        return torque

def calcCurrent(power:float, voltage:float) -> float:
        if (power / voltage) > Parameters["tractiveIMax"]:
            return Parameters["tractiveIMax"]
        return power / voltage

def calcMaxWheelTorque(maxMotorTorque):
        '''
        maxMotorTorque * gear ratio
        '''
        return maxMotorTorque * Parameters["gearRatio"]

def calcMotorForce(motorTorque:float) -> float:
        return (motorTorque * Parameters["gearRatio"] / Parameters["wheelRadius"])

def calcMaxPower(voltage:float) -> float:
        return Parameters["tractiveIMax"] * voltage


def calcVoltage(worldArray:np.ndarray, step:int) -> float:

    F = Parameters["FaradaysConstant"]
    R = Parameters["GasConstant"]
    
    V0 = Magic["cellModel_V0"]
    C1 = Magic["cellModel_C1"]
    C2 = Magic["cellModel_C2"]
    C3 = Magic["cellModel_C3"]
    C4 = Magic["cellModel_C4"]

    R0 = Magic["cellModel_R0"]
    bias = Magic["cellModel_bias"]

    if step > newKernelLen:
        histCurr = worldArray[step-(1+newKernelLen):step-1, varCurrent]
    else:
        histCurr = np.zeros(newKernelLen)
        histCurr[-1*step:] = worldArray[:step, varCurrent]

    def ocv_from_soc(soc, T_K=298.15):
        eps = 1e-9
        soc_shift = soc - (0.1 ** 3)

        denom = np.clip(1.0 - soc_shift + C4, eps, None)
        numer = np.clip(C1 * soc_shift + C3, eps, None)

        log_term = np.log(numer / denom)

        return V0 + (C2 * (R * T_K / F) * log_term)
    h = np.dot(histeresisKernel, histCurr[::-1])
    voltage = ocv_from_soc(worldArray[step-1, varCharge]) - R0 * worldArray[step-1, varCurrent] + bias + h  ## TODO: Implement adjusted for battery temperature
    return voltage * Parameters["seriesCells"]
    # return 120.0

# def step(self, current):

# # Update SOC
# self.SOC -= (current * self.dt) / (3600 * self.capacity_Ah)
# self.SOC = np.clip(self.SOC, 0.0, 1.0)

# # -------- Sliding array logic --------
# self.I_hist[:-1] = self.I_hist[1:]   # shift old values
# self.I_hist[-1] = current             # add new current

# # Hysteresis voltage
# V_hyst = self.hyst_gain * np.sum(self.I_hist * self.kernel)

# # Terminal voltage
# voltage = (
#         self.ocv_from_soc(self.SOC)
#         - self.sag(current) * (1 - self.SOC)
#         - V_hyst
# )

# return voltage