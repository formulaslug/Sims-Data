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
    if df_controls.columns != ['time', 'throttle', 'brakes', 'steerAngle']:
        raise Exception("Simulation controls file must contain the following columns: 'time', 'throttle', 'brakes', 'steerAngle'")
    
    totalSteps = int(Parameters["stepsPerSecond"] * Parameters["simulationDuration"])
    steps = np.arange(0, Parameters["simulationDuration"], 1/Parameters["stepsPerSecond"])
    worldArray = np.zeros(totalSteps + 1, dtype=VehicleState)

    # Set the inital time to 0 if not already 0
    timeSeries = df_controls['time'] - df_controls['time'][0]

    # This takes the last time step and copies it out to the end of the simulation duration. 
    # This has the effect of holding the last command constant until the end of the simulation duration. 
    if timeSeries[-1] < Parameters["simulationDuration"]:
        df_controls = df_controls.vstack(pl.DataFrame({
            'time': [Parameters["simulationDuration"]],
            'throttle': df_controls["throttle"][-1],
            'brakes': df_controls["brakes"][-1],
            'steerAngle': df_controls["steerAngle"][-1]}))

    timeSeries = df_controls['time']
        
    # Interpolation to make the command inputs match the simulation time steps
    # Use cubic spline for driver's real inputs
    if Parameters["interpolationMethod"] == "cubic":
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(timeSeries, df_controls.drop('time').to_numpy())
        controlInputs = cs(steps)
    elif Parameters["interpolationMethod"] == "linear":
        controlInputs = np.zeros((len(steps), 3))
        controlInputs[:,0] = np.interp(steps, timeSeries, df_controls['throttle'])
        controlInputs[:,1] = np.interp(steps, timeSeries, df_controls['brakes'])
        controlInputs[:,2] = np.interp(steps, timeSeries, df_controls['steerAngle'])
    else:
        raise Exception("Unsupported interpolation method. Please use 'cubic' or 'linear'.")

    worldArray[0] = VehicleState(
                position=np.asarray([0,0,0], dtype=np.float32),
                speed=0,
                heading = np.asarray([1,0,0], dtype=np.float32),
                charge=Parameters["vehicleSOC"],
                yawRate = 0,
                brakeTemperature = Parameters["initialBrakeTemperature"]
                )    

    start = time.time()
    # timeRunning = 0
    # currInput = 0
    # stepCount = 0
    # timeSinceLastSteer = 0
    # initSpeed = 0
    for i in range(totalSteps):

        # timeRunning += 1/stepsPerSecond
        # timeSinceLastSteer += 1/stepsPerSecond
        # for commamd in timeBasedInputs:
        #     if currInput + 1 < len(timeBasedInputs) and timeBasedInputs[currInput+1][0] < timeRunning:
        #         currInput += 1
        #         if timeBasedInputs[currInput-1][1][2] != timeBasedInputs[currInput][1][2]:
        #             timeSinceLastSteer = 0
        #             initSpeed = max(currVehicle.speed, 5) # Fails below roughly 5ish
        worldArray[i+1] = stepState(worldArray[i], controlInputs[i]) # Step forward!!
        
    print("*****SIMULATION EXECUTATION TIME****", time.time() -start)

    columns = ['posX', 'posY', 'velX', 'velY', 'speed', 'acceleration',
               'headingX', 'headingY', 'yawRate', 'steerAngle', 'throttle',
               'brakes', 'drag', 'resistiveForces', 'motorForce', 'netForce',
               'torque', 'motorTorque', 'maxTraction', 'maxTractionTorqueAtWheel',
               'cooledBrakeTemperature', 'wheelRPM', 'wheelRotationsHZ',
               'rpm', 'motorRotationsHZ', 'charge', 'voltage', 'current',
               'power', 'maxPower', 'stepSize', 'timeSinceLastSteer']

    dataRows = []
    timeCol = []
    runningTime = 0

    for state in worldArray:
        timeCol.append(runningTime)
        # dataRows.append(state.logProperties())
        runningTime += 1/Parameters["stepsPerSecond"]

    df = pl.DataFrame(dataRows, schema=columns, orient="row")
    df = df.with_columns(pl.Series("time", timeCol, dtype=pl.Float64))

    time = df['time'].to_list()
    current = df['current'].to_list()
    speed = df['speed'].to_list()
    voltage = df['voltage'].to_list()
    torque = df['motorTorque'].to_list()
    yawRate = df['yawRate'].to_list()
    brakeTemperature = df['cooledBrakeTemperature'].to_list()
    ax1 = plt.subplot(1,4,1)
    ax2 = plt.subplot(1,4,2)
    ax3 = plt.subplot(1,4,3)
    ax4 = plt.subplot(1,4,4)

    ax1.set_title("Current vs Time")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Current (A)")
    ax1.plot(time, current)

    ax2.set_title("Speed vs Time")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Speed (m/s)")
    ax2.plot(time, speed)

    ax3.set_title("Voltage vs Time")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Voltage (V)")
    ax3.plot(time, voltage)

    ax3.set_title("Voltage vs Time")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Voltage (V)")
    ax3.plot(time, voltage)

    ax4.set_title("rvt")
    ax4.plot(time, yawRate)

    #ax4.set_ylim([0, 190])
    #ax4.set_yticks(np.arange(0, 181, 20))

    plt.show()
