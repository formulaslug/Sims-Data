import quesadilla as VoltageTools
import numpy as np
def stepElectrical(worldPrev, worldNext, params, inputs):

    worldNext.wheelRPM = worldPrev.speed / params["mechanical"]["wheelCircumferance"] * 60.0

    worldNext.wheelRotationsHz = worldPrev.speed / params["mechanical"]["wheelCircumferance"] * 2.0 * np.pi

    worldNext.rpm = worldNext.wheelRPM * params["mechanical"]["gearRatio"]

    worldNext.motorRotationHz = worldNext.wheelRotationsHz * params["mechanical"]["gearRatio"]

    worldNext.maxPower = params["electrical"]["tractiveIMax"] * worldPrev.voltage

    if worldNext.rpm > 7500:
        worldNext.torque = worldPrev.drag * params["mechanical"]["wheelRadius"]
    else:
        if worldNext.maxPower / params["mechanical"]["maxTorque"] < worldNext.motorRotationHz:
            perfectTractionTorque = params["mechanical"]["maxTorque"] * params["mechanical"]["gearRatio"]
        else:
            perfectTractionTorque = params["mechanical"]["maxTorque"] * params["mechanical"]["gearRatio"]
        worldNext.torque = min(perfectTractionTorque, worldPrev.maxTractionTorqueAtWheel)

    worldNext.motorTorque = worldNext.torque / params["mechanical"]["gearRatio"]
    worldNext.voltage = 28.0 * VoltageTools.lookup(worldPrev.charge, worldPrev.current)

    worldNext.power = worldNext.motorTorque * worldNext.motorRotationHz

    if worldNext.power / worldNext.voltage > params["electrical"]["tractiveIMax"]:
        worldNext.current = params["electrical"]["tractiveIMax"]
    else:
        worldNext.current = worldNext.power / worldNext.voltage

    worldNext.maxTractionTorqueAtWheel = (worldPrev.lbTireTraction.getLongForcePureSlip() + worldPrev.rbTireTraction.getLongForcePureSlip()) * params["mechanical"]["wheelRadius"]

    worldNext.motorForce = worldNext.torque / params["mechanical"]["wheelRadius"]
