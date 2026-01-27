from state import VehicleState
from paramLoader import Parameters, Magic

def maxMotorTorque(worldPrev:VehicleState, resistiveForces:float, maxPower:float, maxTractionTorqueAtWheel:float):
        '''
        Motor Torque at the wheel
        
        minimum(rpm limited torque, power limited torque, perfect traction torque)
        '''
        ## RPM Limited Torque (Motor Controller limits it to ~ this in practice. Maybe something more like 7490ish)
        if worldPrev.motorRPM > 7490:
            return -1 * resistiveForces * Parameters["wheelRadius"]
        if worldPrev.motorRotationsHZ != 0: ## If rolling, torque may be power limited. 
            maxPowerTorque = maxPower / worldPrev.motorRotationsHZ * Parameters["gearRatio"]
        else: ## Avoid divide by 0 error but it's just the same as the max torque that the motor can deliver (180 Nm)
            maxPowerTorque = 180.0 # Nm at 0 rpm
        perfectTractionTorque = Parameters["maxTorque"]
        torque = min(perfectTractionTorque, maxPowerTorque, maxTractionTorqueAtWheel/Parameters["gearRatio"])
        return torque

def current(power, voltage):
        if (power / voltage) > Parameters["tractiveIMax"]:
            return Parameters["tractiveIMax"]
        return power / voltage

def maxWheelTorque(maxMotorTorque):
        '''
        maxMotorTorque * gear rato
        '''
        return maxMotorTorque * Parameters["gearRatio"]

def motorForce(maxWheelTorque):
        return (maxWheelTorque / Parameters["wheelRadius"])

def maxPower(voltage):
        return Parameters["tractiveIMax"] * voltage

def voltage():
        # return 28.0 * lookup(self.charge, self.lastCurrent)
        return 120.0 # Placeholder voltage. Will be a function of SOC, Temp, and Current Histeresis