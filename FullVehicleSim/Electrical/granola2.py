import numpy as np
import lionCellModel as LionCellModel
# print("LOADING granola2.py")

# print("STEP ELECTRICAL CALLED")
def stepElectrical(worldPrev, worldNext, params, inputs):

    worldNext.wheelRPM = worldPrev.speed / params["mechanical"]["wheelCircumferance"] * 60.0
    worldNext.wheelRotationsHz = worldPrev.speed / params["mechanical"]["wheelCircumferance"] * 2.0 * np.pi
    worldNext.rpm = worldNext.wheelRPM * params["mechanical"]["gearRatio"]
    worldNext.motorRotationHz = worldNext.wheelRotationsHz * params["mechanical"]["gearRatio"]

    worldNext.maxPower = params["electrical"]["tractiveIMax"] * worldPrev.voltage

    if worldNext.rpm > 7500:
        worldNext.torque = worldPrev.drag * params["mechanical"]["wheelRadius"]
    else:
        perfectTractionTorque = params["mechanical"]["maxTorque"] * params["mechanical"]["gearRatio"]
        worldNext.torque = min(perfectTractionTorque, worldPrev.maxTractionTorqueAtWheel)

    worldNext.motorTorque = worldNext.torque / params["mechanical"]["gearRatio"]

    # update current history for Gaussian smoothing 
    # Make sure state.py has update_history(self, value) as a METHOD
    worldNext.update_history(worldPrev.current)

    # voltage via template (writes batt_v_rc/hyst/temp onto worldNext) ----
    print("ECM RUNNING", worldPrev.current)

    worldNext.pack_voltage = LionCellModel.update_pack_voltage_template(
    prev_current=worldPrev.current,
    vehicle_state=worldNext,
    params=params
)


    worldNext.power = worldNext.motorTorque * worldNext.motorRotationHz

    # Use worldNext.voltage property (it should return pack_voltage if present)
    if worldNext.power / worldNext.voltage > params["electrical"]["tractiveIMax"]:
        worldNext.current = params["electrical"]["tractiveIMax"]
    else:
        worldNext.current = worldNext.power / worldNext.voltage

    worldNext.maxTractionTorqueAtWheel = (
        worldPrev.lbTireTraction.getLongForcePureSlip() + worldPrev.rbTireTraction.getLongForcePureSlip()
    ) * params["mechanical"]["wheelRadius"]

    worldNext.motorForce = worldNext.torque / params["mechanical"]["wheelRadius"]

