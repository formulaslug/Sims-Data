import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from paramLoader import *
import unyt
from units import unit, CHECK_UNITS

def calcMaxMotorTorque(worldArray:np.ndarray, step: int, resistiveForces:unyt.unyt_quantity, maxPower:unyt.unyt_quantity, maxTractionTorqueAtWheel:unyt.unyt_quantity) -> unyt.unyt_quantity:
        '''
        Motor Torque at the wheel
        
        minimum(rpm limited torque, power limited torque, perfect traction torque)
        '''
        ## RPM Limited Torque (Motor Controller limits it to ~ this in practice. Maybe something more like 7490ish)

        if unit(worldArray[step-1, varMotorRPM], "rpm") > unit(7490, "rpm"):
            return (-1 * resistiveForces * unit(Parameters["wheelRadius"], "m")).to("N*m")
        if unit(worldArray[step-1, varMotorRotationsHZ], "Hz") != unit(0.0, "Hz"): # If rolling, torque may be power limited. 
            maxPowerTorque = (maxPower / (2 * np.pi * unit(worldArray[step-1, varMotorRotationsHZ], "Hz")) * Parameters["gearRatio"]).to("N*m")
        else: ## Avoid divide by 0 error but it's just the same as the max torque that the motor can deliver (180 Nm)
            maxPowerTorque = unit(180.0, "N*m") # N*m at 0 rpm
        torque = (min(unit(Parameters["tractiveIMax"], "N*m"), maxPowerTorque, maxTractionTorqueAtWheel/Parameters["gearRatio"])).to("N*m")
        return torque

def calcCurrent(power: unyt.unyt_quantity, voltage: unyt.unyt_quantity) -> unyt.unyt_quantity:
        if (power.to("W") / voltage.to("V")) > unit(Parameters["tractiveIMax"], "A"):
            return unit(Parameters["tractiveIMax"], "A")
        return (power / voltage).to("A")

def calcMaxWheelTorque(maxMotorTorque: unyt.unyt_quantity) -> unyt.unyt_quantity:
        '''
        maxMotorTorque * gear ratio
        '''
        return maxMotorTorque.to("N*m") * Parameters["gearRatio"]

def calcMotorForce(maxWheelTorque: unyt.unyt_quantity) -> unyt.unyt_quantity:
        return (maxWheelTorque.to("N*m") / unit(Parameters["wheelRadius"], "m")).to("N")

def calcMaxPower(voltage: unyt.unyt_quantity) -> unyt.unyt_quantity:
        return (unit(Parameters["tractiveIMax"], "A") * voltage).to("W")

def calcVoltage() -> unyt.unyt_quantity:
        # return 28.0 * lookup(self.charge, self.lastCurrent)
        return unit(120.0, "V") # Placeholder voltage. Will be a function of SOC, Temp, and Current Histeresis


if __name__ == "__main__":
    print(calcMaxPower(calcVoltage()), calcCurrent(unit(10000.0, "W"), calcVoltage()))