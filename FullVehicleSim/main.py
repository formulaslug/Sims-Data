import matplotlib.pyplot as plt
import json
import polars as pl

from state import *
from engine import *

# Sim parameters
stepsPerSecond = 100
simDuration = 20
import time

if __name__ == "__main__":
    currVehicle = VehicleState(
                stepSize = 1/stepsPerSecond,
                position=np.asarray([0,0,0], dtype=np.float32),
                speed=0,
                acceleration=0,
                heading = np.asarray([1,0,0], dtype=np.float32),
                charge=50,
                lastCurrent=0,
                throttle = 0,
                brakes = 0,
                yawRate = 0,
                steerAngle = 0,
                brakeTemperature = 150,
                timeSinceLastSteer = 0,
                initSpeed = 0
                )

    with open('controls.json', 'r') as file:
        timeBasedInputs = json.load(file)
    timeBasedInputs = sorted((float(key), [float(value) for value in values]) for key, values in timeBasedInputs.items())
    # No inputs
    if len(timeBasedInputs) == 0:
        raise Exception("controls.json must contain at least 1 valid input")
    vehicleStates = [currVehicle]


    #timeBasedInputs = {2: [1,0,0], 6.3: [0,1,0.2]}
    start = time.time()
    timeRunning = 0
    currInput = 0
    stepCount = 0
    timeSinceLastSteer = 0
    initSpeed = 0
    for _ in range(simDuration*stepsPerSecond):
        timeRunning += 1/stepsPerSecond
        timeSinceLastSteer += 1/stepsPerSecond
        for commamd in timeBasedInputs:
            if currInput + 1 < len(timeBasedInputs) and timeBasedInputs[currInput+1][0] < timeRunning:
                currInput += 1
                if timeBasedInputs[currInput-1][1][2] != timeBasedInputs[currInput][1][2]:
                    timeSinceLastSteer = 0
                    initSpeed = max(currVehicle.speed, 5) # Fails below roughly 5ish
        currVehicle = stepState(currVehicle, timeBasedInputs[currInput][1], 1/stepsPerSecond, timeSinceLastSteer, initSpeed) # Step forward!!
        vehicleStates.append(currVehicle)
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

    for state in vehicleStates:
        timeCol.append(runningTime)
        dataRows.append(state.logProperties())
        runningTime += 1/stepsPerSecond

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
