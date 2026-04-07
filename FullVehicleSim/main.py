import matplotlib.pyplot as plt
import json
import polars as pl
import argparse
import time

from paramLoader import Magic, Parameters
from state import *
from engine import *

if __name__ == "__main__":
    ## Argument Parsing. Should wind up like:
    # python main.py --simulation_parameters path/to/params.json --simulation_controls path/to/controls.csv
    Parser = argparse.ArgumentParser(description='Full Vehicle Simulator')
    Parser.add_argument('--simulation_controls', '-c', type=str, help='Simulation Controls File Path', required=True)

    args = Parser.parse_args()

    simulation_controls_path = args.simulation_controls

    if simulation_controls_path: # If not None or empty
        if simulation_controls_path.endswith('.csv'): ## Check for csv and read as that
            df_controls = pl.read_csv(simulation_controls_path)
        elif simulation_controls_path.endswith('.parquet'): ## Check for parquet and read as that
            df_controls = pl.read_parquet(simulation_controls_path)
        else:
            raise Exception("Unsupported file format for simulation controls. Please use .csv or .parquet files.")
    else:
        raise Exception("Please provide a valid simulation controls file path using --simulation_controls or -c")
    
    ## Double check it has the correct columns
    if df_controls.columns != ['time', 'throttle', 'brakesFront','brakesRear', 'steerAngle']:
        raise Exception("Simulation controls file must contain the following columns: 'time', 'throttle', 'brakesFront', 'brakesRear', 'steerAngle'")
    
    totalSteps = int(Parameters["stepsPerSecond"] * Parameters["simulationDuration"])
    steps = np.arange(0, Parameters["simulationDuration"], 1/Parameters["stepsPerSecond"])

    cols = ["x", "y", "z", "vX", "vY", "vZ", "speed", 
                    "headingX", "headingY", "headingZ", 
                    "yawRate", "frontBrakeTemperature", "rearBrakeTemperature", 
                    "charge", "drag", "resistiveForces", 
                    "motorTorque", "motorForce", "netForce", 
                    "maxTraction", "wheelRotationsHZ", "motorRPM",
                    "motorRotationsHZ", "current", 
                    "maxWheelTorque", "maxPower", "power", 
                    "voltage", 
                    "frontBrakeForce", "rearBrakeForce", 
                    "frontBrakeHeating", "rearBrakeHeating", 
                    "frontBrakeCooling", "rearBrakeCooling",
                    "frontSlipAngle", "rearSlipAngle"]
    
    log = np.zeros((totalSteps + 1, len(cols)))
    worldArray = np.zeros(totalSteps + 1, dtype=VehicleState)

    # Set the inital time to 0 if not already 0
    timeSeries = df_controls['time'] - df_controls['time'][0]

    # This takes the last time step and copies it out to the end of the simulation duration. 
    # This has the effect of holding the last command constant until the end of the simulation duration. 
    if timeSeries[-1] < Parameters["simulationDuration"]:
        df_controls = df_controls.vstack(pl.DataFrame({
            'time': [Parameters["simulationDuration"]],
            'throttle': df_controls["throttle"][-1],
            'brakesFront': df_controls["brakesFront"][-1],
            'brakesRear': df_controls["brakesRear"][-1],
            'steerAngle': df_controls["steerAngle"][-1]}))

    timeSeries = df_controls['time']
        
    # Interpolation to make the command inputs match the simulation time steps
    # Use cubic spline for driver's real inputs
    if Parameters["interpolationMethod"] == "cubic":
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(timeSeries, df_controls.drop('time').to_numpy())
        controlInputs = cs(steps)
    elif Parameters["interpolationMethod"] == "linear":
        controlInputs = np.zeros((len(steps), 4))
        controlInputs[:,0] = np.interp(steps, timeSeries, df_controls['throttle'])
        controlInputs[:,1] = np.interp(steps, timeSeries, df_controls['brakesFront'])
        controlInputs[:,2] = np.interp(steps, timeSeries, df_controls['brakesRear'])
        controlInputs[:,3] = np.interp(steps, timeSeries, df_controls['steerAngle'])
    else:
        raise Exception("Unsupported interpolation method. Please use 'cubic' or 'linear'.")

    worldArray[0] = VehicleState(
                position=np.asarray([0,0,0], dtype=np.float32),
                speed=0,
                heading = np.asarray([1,0,0], dtype=np.float32),
                charge=Parameters["vehicleSOC"],
                yawRate = 0,
                frontBrakeTemperature = Parameters["initialBrakeTemperature"],
                rearBrakeTemperature= Parameters["initialBrakeTemperature"],
                tractiveBatteryTemperature = Parameters["initialBatteryTemperature"]
                )    
    
    timeCol = np.arange(0, Parameters["simulationDuration"] + 1/Parameters["stepsPerSecond"], 1/Parameters["stepsPerSecond"])

    start = time.time()
    for i in range(totalSteps):
        worldArray[i+1], log[i+1] = stepState(worldArray[i], controlInputs[i]) # Step forward!!
        ## This was above the stepState but I moved it down to make it clearer to read.
        # timeRunning += 1/stepsPerSecond
        # timeSinceLastSteer += 1/stepsPerSecond
        # for commamd in timeBasedInputs:
        #     if currInput + 1 < len(timeBasedInputs) and timeBasedInputs[currInput+1][0] < timeRunning:
        #         currInput += 1
        #         if timeBasedInputs[currInput-1][1][2] != timeBasedInputs[currInput][1][2]:
        #             timeSinceLastSteer = 0
        #             initSpeed = max(currVehicle.speed, 5) # Fails below roughly 5ish
        
    print("*****SIMULATION EXECUTATION TIME****", time.time() -start)

    # columns = ['posX', 'posY', 'velX', 'velY', 'speed', 'acceleration',
    #            'headingX', 'headingY', 'yawRate', 'steerAngle', 'throttle',
    #            'brakesFront', 'brakesRear', 'drag', 'resistiveForces', 'motorForce', 'netForce',
    #            'torque', 'motorTorque', 'maxTraction', 'maxTractionTorqueAtWheel',
    #            'cooledBrakeTemperature', 'wheelRPM', 'wheelRotationsHZ',
    #            'rpm', 'motorRotationsHZ', 'charge', 'voltage', 'current',
    #            'power', 'maxPower', 'stepSize', 'timeSinceLastSteer']

    df = pl.DataFrame(log, schema=cols, orient="row")
    # print(f"df shape: {df.shape}")
    # print(f"control inputs shape: {controlInputs.shape}")
    # print(f"timeCol shape: {timeCol.shape}")
    df = df[1:].with_columns(
        pl.Series(timeCol[1:]).alias("time"),
        pl.Series(controlInputs[:,0]).alias("throttle"),
        pl.Series(controlInputs[:,1]).alias("brakesFront"),
        pl.Series(controlInputs[:,2]).alias("brakesRear"),
        pl.Series(controlInputs[:,3]).alias("steerAngle")
    )

    df.write_parquet("simulation_output.parquet")

    t = df['time']
    current = df['current']
    speed = df['speed']
    voltage = df['voltage']
    torque = df['motorTorque']
    yawRate = df['yawRate']
    frontBrakeTemperature = df['frontBrakeTemperature']
    ax1 = plt.subplot(1,4,1)
    ax2 = plt.subplot(1,4,2)
    ax3 = plt.subplot(1,4,3)
    ax4 = plt.subplot(1,4,4)

    ax1.set_title("Current vs Time")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Current (A)")
    ax1.plot(t, current)

    ax2.set_title("Speed vs Time")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Speed (m/s)")
    ax2.plot(t, speed)

    ax3.set_title("Voltage vs Time")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Voltage (V)")
    ax3.plot(t, voltage)

    ax3.set_title("Voltage vs Time")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Voltage (V)")
    ax3.plot(t, voltage)

    ax4.set_title("rvt")
    ax4.plot(t, yawRate)

    #ax4.set_ylim([0, 190])
    #ax4.set_yticks(np.arange(0, 181, 20))

    plt.show()
