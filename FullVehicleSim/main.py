import matplotlib.pyplot as plt
import json
import polars as pl
import argparse
import time

from paramLoader import *
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
    if df_controls.columns != ['time', 'throttle', 'brakePressureFront','brakePressureRear', 'steerAngle']:
        raise Exception("Simulation controls file must contain the following columns: 'time', 'throttle', 'brakePressureFront', 'brakePressureRear', 'steerAngle'")
    
    totalSteps = int(Parameters["stepsPerSecond"] * Parameters["simulationDuration"])
    steps = np.arange(0, Parameters["simulationDuration"], 1/Parameters["stepsPerSecond"])


    ## This is structured so the first row is the initial conditions (inputs don't matter and will just be left to 0), and the
    ## rest are generated as the simulation progresses. This means that a simulation array will always be 1 longer than just the time steps
    ## and duration would indicate. 
    worldArray = np.zeros((totalSteps + 1, len(VARIABLE_NAMES)), dtype=np.float32)

    # Set the inital time to 0 if not already 0. Eg. [1.79, 2.36, 3.13] becomes [0.0, 0.57, 1.34]
    timeSeries = df_controls['time'] - df_controls['time'][0] # Normalize to start at 0

    # This takes the last time step and copies it out to the end of the simulation duration. 
    # This has the effect of holding the last command constant until the end of the simulation duration. 
    if timeSeries[-1] < Parameters["simulationDuration"]:
        df_controls = df_controls.vstack(pl.DataFrame({
            'time': [Parameters["simulationDuration"]],
            'throttle': df_controls["throttle"][-1],
            'brakePressureFront': df_controls["brakePressureFront"][-1],
            'brakePressureRear': df_controls["brakePressureRear"][-1],
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
        controlInputs[:,1] = np.interp(steps, timeSeries, df_controls['brakePressureFront'])
        controlInputs[:,2] = np.interp(steps, timeSeries, df_controls['brakePressureRear'])
        controlInputs[:,3] = np.interp(steps, timeSeries, df_controls['steerAngle'])
    else:
        raise Exception("Unsupported interpolation method. Please use 'cubic' or 'linear'.")

    ## Setup initial conditions. Leaves row 0 with no inputs (don't matter anyway since sim runs from 1 -> end)
    ## Some other initial conditions based on input parameters.
    worldArray[1:, varThrottle] = controlInputs[:,0]
    worldArray[1:, varBrakePressureFront] = controlInputs[:,1]
    worldArray[1:, varBrakePressureRear] = controlInputs[:,2]
    worldArray[1:, varSteerAngle] = controlInputs[:,3]
    worldArray[0,varCharge] = Parameters["vehicleSOC"]
    worldArray[0,varFrontBrakeTemperature] = Parameters["initialBrakeTemperature"]
    worldArray[0,varRearBrakeTemperature] = Parameters["initialBrakeTemperature"]
    worldArray[0, varHeadingX:varHeadingZ+1] = Parameters["initHeading"]
    worldArray[0, varPosX:varPosZ+1] = Parameters["initPosition"]
    worldArray[0, varVelX:varVelZ+1] = Parameters["initVelocity"]
    worldArray[0, varLateralVelocty] = Parameters["InitLateralVelocity"] # velocity in y direction (needed for yaw rate)
    worldArray[0, varYawRate] = Parameters["InitYawRate"]
    worldArray[:, varTime] = np.arange(0, Parameters["simulationDuration"] + 1/Parameters["stepsPerSecond"], 1/Parameters["stepsPerSecond"])

    start = time.time()
    for i in range(1, totalSteps):
        worldArray[i, :] = stepState(worldArray, i) # Step forward!!
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
    # print(VARIABLE_NAMES)

    df = pl.DataFrame(worldArray, schema=VARIABLE_NAMES, orient="row")
    # print(f"df shape: {df.shape}")
    # print(f"control inputs shape: {controlInputs.shape}")
    # print(f"timeCol shape: {timeCol.shape}")

    df.write_parquet("simulation_output.parquet")

    t = df['time']
    current = df['current']
    speed = df['speed']
    voltage = df['voltage']
    torque = df['motorTorque']
    yawRate = df['yawRate']
    frontBrakeTemperature = df['frontBrakeTemperature']
    ax1 = plt.subplot(411)
    ax11 = ax1.twinx()
    ax2 = plt.subplot(412)
    ax22 = ax2.twinx()
    ax3 = plt.subplot(413)
    ax33 = ax3.twinx()
    ax4 = plt.subplot(414)
    ax44 = ax4.twinx()

    ax1.set_title("I (Blue)/V (Orange) vs Time")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Current (A) / Voltage (V)")
    ax1.plot(t, current, label="Current")
    ax11.plot(t, voltage, label="Voltage", color='orange')
    ax1.legend()

    ax2.set_title("Speed vs Time")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Speed (m/s)")
    ax2.plot(t, speed)

    ax3.set_title("Throttle (Blue)/Brakes (Orange) vs Time")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Throttle (0-1)")
    ax33.set_ylabel("Brake Pressure (PSI)")
    ax3.plot(t, df["throttle"], label="Throttle")
    ax33.plot(t, df["brakePressureFront"], color='orange')

    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Yaw Rate (radians)")
    ax4.set_title("yawRate")
    ax4.plot(t, yawRate)

    #ax4.set_ylim([0, 190])
    #ax4.set_yticks(np.arange(0, 181, 20))

    plt.show()
