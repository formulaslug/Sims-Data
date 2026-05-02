import csv
import json
import time
import math
import copy
import numpy as np

from sim.state import VehicleState
from sim.engine import stepState
from sim.ramen import Parameters, Magic


def cloneVehicle(vehicle):
    return copy.deepcopy(vehicle)


def step(vehicle, inputs, dt, t):
    """Single integration step wrapper."""
    return stepState(vehicle, inputs, dt, t, vehicle.speed)


def readCSV(filename):
    track = []
    with open(filename, "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row:
                continue

            feature = row[0].strip().upper()

            if feature == "STRAIGHT" and len(row) >= 2:
                track.append(("STRAIGHT", float(row[1])))

            elif feature == "CORNER" and len(row) >= 3:
                track.append(("CORNER", float(row[1]), float(row[2])))

    return track

def calculateMaxCornerSpeed(radius, vehicle):
  maxLatAccel = Parameters["MaxAccel"] / Parameters["Mass"]
  return math.sqrt(maxLatAccel * radius) * 0.85

def simulateCorner(track, index, targetExitSpeed, vehicle, config):
    debug = config.get("debug", False)
    radius = abs(track[index][1])
    arcLength = track[index][2]

    requiredAngle = abs(arcLength) / radius
    steerDir = 1 if arcLength > 0 else -1

    maxCornerSpeed = calculateMaxCornerSpeed(abs(radius), vehicle)

    bestTime = float("inf")
    bestVehicle = None
    bestInputs = []

    dt = 1 / config["stepsPerSecond"]

    strategies = ["brake_then_turn", "trail_brake", "coast_then_power"]

    for strategy in strategies:
        for steerScale in np.arange(0.5, 1.5, 0.25):
            currVehicle = cloneVehicle(vehicle)
            angle = 0.0
            t = 0.0
            inputsLog = []

            steerAngle = steerDir * min(
                steerScale / radius, 2.0
            )

            while abs(angle) < 0.95 * requiredAngle and t < config["maxCornerTime"]:
                progress = abs(angle) / requiredAngle

                if strategy == "brake_then_turn":
                    if progress < 0.3:
                        inputs = [0.0, 0.5, steerAngle]
                    elif progress < 0.7:
                        inputs = [0.0, 0.0, steerAngle]
                    else:
                        inputs = [0.4, 0.0, steerAngle]

                elif strategy == "trail_brake":
                    brake = max(0.3 * (1 - progress), 0)
                    throttle = max(0.4 * (progress - 0.5), 0)
                    inputs = [throttle, brake, steerAngle]

                else:
                    inputs = [0.0 if progress < 0.6 else 0.4, 0.0, steerAngle]

                currVehicle = step(currVehicle, inputs, dt, t)

                if currVehicle.speed > maxCornerSpeed:
                    currVehicle.speed = maxCornerSpeed

                angle += currVehicle.yawRate * dt
                t += dt
                inputsLog.append((t, inputs))

                if currVehicle.speed < 1.0:
                    break

            if (
                abs(angle) >= 0.9 * requiredAngle
                and currVehicle.speed <= targetExitSpeed * 1.1
                and t < bestTime
            ):
                if debug:
                    print(
                            f"[CORNER {index}] "
                            f"strategy={strategy} "
                            f"time={t:.2f}s "
                            f"exitSpeed={currentVehicle.speed:.2f} "
                            f"target={targetExitSpeed: .2f} "
                            f"angle={angle:.2f}/{requiredAngle:.2f}"
                    )
                bestTime = t
                bestVehicle = currVehicle
                bestInputs = inputsLog

    if bestVehicle is None:
        currVehicle = cloneVehicle(vehicle)
        angle = 0.0
        t = 0.0
        inputsLog = []

        safeSpeed = min(targetExitSpeed, maxCornerSpeed * 0.7)
        steerAngle = (safeSpeed / abs(radius)) * steerDir

        while abs(angle) < requiredAngle and t < config["maxCornerTime"]:
            if currVehicle.speed > safeSpeed:
                inputs = [0.0, 0.4, steerAngle]
            else:
                inputs = [0.3, 0.0, steerAngle]

            currVehicle = step(currVehicle, inputs, dt, t)
            angle += currVehicle.yawRate * dt
            t += dt
            inputsLog.append((t, inputs))

        return t, currVehicle, inputsLog

    return bestTime, bestVehicle, bestInputs


def simulateStraight(track, index, targetExitSpeed, vehicle, config):
    length = track[index][1]
    dt = 1 / config["stepsPerSecond"]

    bestTime = float("inf")
    bestVehicle = None
    bestInputs = []

    for brakeFrac in np.arange(0.6, 0.95, 0.05):
        currVehicle = cloneVehicle(vehicle)
        dist = 0.0
        t = 0.0
        inputsLog = []

        brakeDist = length * brakeFrac

        while dist < length and t < 100:
            if dist < brakeDist:
                inputs = [1.0, 0.0, 0.0]
            else:
                inputs = [0.0, 1.0, 0.0]

            currVehicle = step(currVehicle, inputs, dt, t)
            dist += currVehicle.speed * dt
            t += dt
            inputsLog.append((t, inputs))

            if currVehicle.speed < 0.1:
                break

        if abs(dist - length) < 1.0 and currVehicle.speed <= targetExitSpeed:
            if t < bestTime:
                bestTime = t
                bestVehicle = currVehicle
                bestInputs = inputsLog

    if bestVehicle is None:
        currVehicle = cloneVehicle(vehicle)
        dist = 0.0
        t = 0.0
        inputsLog = []

        while dist < length and t < 100:
            stoppingDist = (currVehicle.speed ** 2) / (2 * 5)

            if stoppingDist < (length - dist) * 0.8:
                inputs = [0.5, 0.0, 0.0]
            else:
                inputs = [0.0, 0.5, 0.0]

            currVehicle = step(currVehicle, inputs, dt, t)
            dist += currVehicle.speed * dt
            t += dt
            inputsLog.append((t, inputs))

        return t, currVehicle, inputsLog

    return bestTime, bestVehicle, bestInputs

def backwardPass(track, config):
    maxEntry = []
    maxExitSpeed = 100.0

    dummy = VehicleState(
        stepSize=1 / config["stepsPerSecond"],
        position=np.zeros(3),
        speed=20,
        acceleration=0,
        heading=np.array([1, 0, 0]),
        charge=50,
        lastCurrent=0,
        throttle=0,
        brakes=0,
        yawRate=0,
        steerAngle=0,
        brakeTemperature=150,
        timeSinceLastSteer=0,
        initSpeed=20,
    )

    for feature in reversed(track):
        if feature[0] == "CORNER":
            maxSpeed = calculateMaxCornerSpeed(abs(feature[1]), dummy)
            entry = min(maxSpeed, maxExitSpeed)

        elif feature[0] == "STRAIGHT":
            length = feature[1]
            maxAccel = Parameters["MaxAccel"] / Parameters["Mass"]

            entry = min(
                60.0,
                math.sqrt(maxExitSpeed**2 + 2 * 10 * length),
            )
        else:
            entry = maxExitSpeed

        maxEntry.insert(0, entry)
        maxExitSpeed = entry

    return maxEntry


def forwardPass(track, initVehicle, maxEntrySpeeds, config):
    currVehicle = cloneVehicle(initVehicle)
    segmentTimes = []
    allInputs = []

    for i, feature in enumerate(track):
        targetExitSpeed = maxEntrySpeeds[i]

        if feature[0] == "CORNER":
            t, currVehicle, inputs = simulateCorner(
                track, i, targetExitSpeed, currVehicle, config
            )
        else:
            t, currVehicle, inputs = simulateStraight(
                track, i, targetExitSpeed, currVehicle, config
            )

        segmentTimes.append(t)
        allInputs.append(inputs)

    return segmentTimes, allInputs


def main(track, initVehicle, config):
    maxEntrySpeeds = backwardPass(track, config)
    segmentTimes, allInputs = forwardPass(
        track, initVehicle, maxEntrySpeeds, config
    )

    totalTime = sum(segmentTimes)

    formatted = []
    tOffset = 0.0

    for seg in allInputs:
        segOut = []
        for dt, inputs in seg:
            segOut.append([tOffset + dt] + inputs)
        if seg:
            tOffset += seg[-1][0]
        formatted.append(segOut)

    return totalTime, formatted


if __name__ == "__main__":
    start = time.time()

    with open("laptimeSimConfig.json") as f:
        config = json.load(f)

    initVehicle = VehicleState(
        stepSize=1 / config["stepsPerSecond"],
        position=np.zeros(3),
        speed=0.1,
        acceleration=0,
        heading=np.array([1, 0, 0]),
        charge=50,
        lastCurrent=0,
        throttle=0,
        brakes=0,
        yawRate=0,
        steerAngle=0,
        brakeTemperature=150,
        timeSinceLastSteer=0,
        initSpeed=0,
    )

    track = readCSV(config["trackDefinition"])

    lapTime, inputs = main(track, initVehicle, config)

    with open("lapControlInputs.json", "w") as f:
        json.dump(inputs, f, indent=2)

    print("Lap time:", lapTime)
    print("Runtime:", time.time() - start)
