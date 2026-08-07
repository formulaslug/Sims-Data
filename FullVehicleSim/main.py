import matplotlib.pyplot as plt
import json
import polars as pl
import argparse
import time
import numpy as np

from paramLoader import *
from engine import *

if __name__ == "__main__":
    Parser = argparse.ArgumentParser(description='Full Vehicle Simulator')
    
   
    Parser.add_argument('--simulation_controls', '-c', type=str, help='Simulation Controls File Path', required=False)
    
    # NEW: Accepts a list of ratios to test 
    Parser.add_argument('--ratios', '-r', nargs='+', type=float, help='List of gear ratios to sweep', default=[3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    #(if you want to test your own ratios, enter them when you are running the program. Ex: "python main.py --ratios 1.5 2.5")
    #if we want to test 3:2, we write the gear ratio as 1.5 since 3/2 = 1.5
    
    args = Parser.parse_args()

    
    if args.simulation_controls:
        if args.simulation_controls.endswith('.csv'):
            df_controls = pl.read_csv(args.simulation_controls)
        elif args.simulation_controls.endswith('.parquet'):
            df_controls = pl.read_parquet(args.simulation_controls)
        else:
            raise Exception("Unsupported file format for simulation controls.")
    else:
        # Default Fix 1: Flat-out 100% throttle straight line
        print("Running Straight line acceleration")
        df_controls = pl.DataFrame({
            'time': [0.0, Parameters["simulationDuration"]],
            'throttle': [1.0, 1.0],
            'brakePressureFront': [0.0, 0.0],
            'brakePressureRear': [0.0, 0.0],
            'steerAngle': [0.0, 0.0]
        })

    # Double check it has the correct columns
    if df_controls.columns != ['time', 'throttle', 'brakePressureFront','brakePressureRear', 'steerAngle']:
        raise Exception("Simulation controls file must contain: 'time', 'throttle', 'brakePressureFront', 'brakePressureRear', 'steerAngle'")
    
    totalSteps = int(Parameters["stepsPerSecond"] * Parameters["simulationDuration"])
    steps = np.arange(0, Parameters["simulationDuration"], 1/Parameters["stepsPerSecond"])

    # Interpolation to match simulation time steps
    timeSeries = df_controls['time'] - df_controls['time'][0]
    if timeSeries[-1] < Parameters["simulationDuration"]:
        df_controls = df_controls.vstack(pl.DataFrame({
            'time': [Parameters["simulationDuration"]],
            'throttle': df_controls["throttle"][-1],
            'brakePressureFront': df_controls["brakePressureFront"][-1],
            'brakePressureRear': df_controls["brakePressureRear"][-1],
            'steerAngle': df_controls["steerAngle"][-1]}))

    timeSeries = df_controls['time']
        
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

    # loop through gear ratios
    gear_ratios = args.ratios
    summary_results = []

    for ratio in gear_ratios:
        print(f"\n---> Running Sim for Gear Ratio: {ratio}")
        
        # Set gear ratio in parameters
        Parameters["gearRatio"] = ratio

        # Allocate world array & initial conditions
        worldArray = np.zeros((totalSteps + 1, len(VARIABLE_NAMES)), dtype=np.float32)
        worldArray[1:, varThrottle] = controlInputs[:,0]
        worldArray[1:, varBrakePressureFront] = controlInputs[:,1]
        worldArray[1:, varBrakePressureRear] = controlInputs[:,2]
        worldArray[1:, varSteerAngle] = controlInputs[:,3]
        worldArray[0, varCharge] = Parameters["vehicleSOC"]
        worldArray[0, varFrontBrakeTemperature] = Parameters["initialBrakeTemperature"]
        worldArray[0, varRearBrakeTemperature] = Parameters["initialBrakeTemperature"]
        worldArray[0, varHeadingX:varHeadingZ+1] = Parameters["initHeading"]
        worldArray[0, varPosX:varPosZ+1] = Parameters["initPosition"]
        worldArray[0, varVelX:varVelZ+1] = Parameters["initVelocity"]
        worldArray[:, varTime] = np.arange(0, Parameters["simulationDuration"] + 1/Parameters["stepsPerSecond"], 1/Parameters["stepsPerSecond"])

        # Execute simulation run
        start = time.time()
        for i in range(1, totalSteps):
            worldArray[i, :] = stepState(worldArray, i)
        
        # Save results
        df = pl.DataFrame(worldArray, schema=VARIABLE_NAMES, orient="row")
        
        # Metric Extraction
        top_speed_mps = df['speed'].max()
        peak_current = df['current'].max()
        
        # Time to ~60mph (26.8 m/s)
        sub_60 = df.filter(pl.col("speed") >= 26.8)
        time_to_60 = sub_60['time'].min() if len(sub_60) > 0 else "N/A"

        summary_results.append({
            "Gear Ratio": ratio,
            "0-60 mph (s)": round(time_to_60, 3) if isinstance(time_to_60, float) else time_to_60,
            "Top Speed (m/s)": round(top_speed_mps, 2),
            "Peak Current (A)": round(peak_current, 1)
        })

    # Output overall table to console
    print("\n" + "="*50)
    print("      POWERTRAIN GEARING SWEEP RESULTS")
    print("="*50)
    print(pl.DataFrame(summary_results))

    # Save last run parquet for visualization
    df.write_parquet("simulation_output.parquet")