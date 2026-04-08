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


def calcVoltage(worldArray:np.ndarray, step:int) -> float:
    delta = 1 / Parameters["stepsPerSecond"]
    capacity_Ah = Parameters["cellCapacity_Ah"]
    soc = worldArray[step-1, varCharge]

    sigma = Parameters["cellModelSigma"]
    hystGain = Parameters["hysteresisGain"]

    # Sliding window: last 10 seconds of current
    I_hist = np.zeros(int(10 * Parameters["stepsPerSecond"]))
    I_hist[:max(0, step - len(I_hist))] = worldArray[max(0, step - len(I_hist)):step, varCurrent]  # Get the current history up to the current step

    # Hysteresis kernel
    t = Parameters["histeresisKernelLength"]
    kernel = np.exp(-(t**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)

    def ocv_from_soc(self, soc):
        return 3.0 + 0.9 * soc + 0.25 * np.exp(-12 * (1 - soc))s

    def sag(self, current):
        return 0.02 * current + 0.004 * (current ** 1.3)
    
    V_hyst = self.hyst_gain * np.sum(self.I_hist * self.kernel)
    # Terminal voltage
    voltage = (
        self.ocv_from_soc(self.SOC)
        - self.sag(current) * (1 - self.SOC)
        - V_hyst
        )

    return voltage
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