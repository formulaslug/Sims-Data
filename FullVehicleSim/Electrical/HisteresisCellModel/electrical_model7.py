"""Electrical Model 7

This file is a medium-complexity lithium-ion cell model built for interactive
work rather than CLI execution.

Design goals:
- Use all available experiment groups that matter here: CHC, DCP, CHP, and
  DCC when present.
- Keep the model functional and easy to inspect from an editor or notebook.
- Capture the main battery effects without jumping straight to a full single
  particle model.
- Explain each part of the model directly in code.

Why this approach instead of a full SPM:
Single particle models are powerful, but they are substantially more complex
to identify, tune, and maintain. They usually need more electrochemical state
structure than this dataset clearly provides. For the current use case, a
structured equivalent-circuit model with SOC-dependent OCV, ohmic loss, two
polarization branches, and a hysteresis state is a better middle ground.

The model below is built in layers:
1. Stack all experiments into a single training set.
2. Estimate a smooth voltage curve from SOC and temperature.
3. Add current-dependent ohmic loss.
4. Add fast and slow polarization states.
5. Add a hysteresis state that depends on current direction and SOC.
6. Fit the whole structure with nonlinear least squares.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.io
from scipy.integrate import cumulative_simpson
from scipy.optimize import least_squares


# ============================================================
# PATH RESOLUTION AND DATA LOADING
# ============================================================

print("[electrical_model7] attempting to load .mat data (top-level) ...")
matData = scipy.io.loadmat("../fs-data/FS-4/P30B Cell Data/Molicel_INR18650P30B_measurement.mat", squeeze_me=True, struct_as_record=False)
print("[electrical_model7] loaded .mat data (top-level)")

def LoadMeasurementData(matPath: Path | None = None):
    """Load the MAT file and return the `measurement` structure."""
    print("[electrical_model7] LoadMeasurementData: returning measurement struct")
    matData = scipy.io.loadmat("../fs-data/FS-4/P30B Cell Data/Molicel_INR18650P30B_measurement.mat", squeeze_me=True, struct_as_record=False)
    print("[electrical_model7] LoadMeasurementData: loaded file")
    return matData["measurement"]


# ============================================================
# DATA STACKING
# ============================================================


def NormalizeExperimentCollection(experimentCollection) -> List[object]:
    """Return a Python list even if MATLAB stored the group as a scalar object."""

    if isinstance(experimentCollection, np.ndarray):
        items = [item for item in experimentCollection.reshape(-1)]
        print(f"[electrical_model7] NormalizeExperimentCollection: normalized array -> {len(items)} items")
        return items
    return [experimentCollection]


def SafeSurfaceTemperature(experiment) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the two surface temperature channels in a shape-safe way."""

    surfaceTemperature = np.asarray(experiment.T_surf, dtype=float)
    if surfaceTemperature.ndim == 1:
        surfaceOne = surfaceTemperature.reshape(-1)
        surfaceTwo = surfaceTemperature.reshape(-1)
        return surfaceOne, surfaceTwo

    if surfaceTemperature.shape[0] == 1:
        surfaceOne = surfaceTemperature[0, :]
        surfaceTwo = surfaceTemperature[0, :]
        return surfaceOne, surfaceTwo

    surfaceOne = surfaceTemperature[0, :]
    surfaceTwo = surfaceTemperature[1, :]
    return surfaceOne, surfaceTwo


def GuessInitialSoc(groupName: str) -> float:
    """Give each experiment family a conservative starting SOC guess."""
    return 1.0


def BuildExperimentRecord(experiment, groupName: str, experimentIndex: int) -> Dict[str, np.ndarray]:
    """Convert one raw experiment into a compact record used by the fitter."""

    currentSeries = np.asarray(experiment.I, dtype=float).reshape(-1)
    voltageSeries = np.asarray(experiment.V, dtype=float).reshape(-1)
    timeSeries = np.asarray(experiment.t, dtype=float).reshape(-1)
    surfaceOne, surfaceTwo = SafeSurfaceTemperature(experiment)
    temperatureSeries = 0.5 * (surfaceOne + surfaceTwo)
    deltaTimeSeries = np.concatenate([[0.0], np.diff(timeSeries)])
    throughputAh = np.concatenate([[0.0], cumulative_simpson(y=currentSeries, x=timeSeries) / 3600.0])

    record = {
        "groupName": np.array(groupName),
        "experimentIndex": np.array(experimentIndex, dtype=int),
        "initialSocGuess": np.array(GuessInitialSoc(groupName), dtype=float),
        "current": currentSeries,
        "voltage": voltageSeries,
        "time": timeSeries,
        "temperature": temperatureSeries,
        "surfaceOne": surfaceOne,
        "surfaceTwo": surfaceTwo,
        "deltaTime": deltaTimeSeries,
        "throughputAh": throughputAh,
    }
    print(f"[electrical_model7] BuildExperimentRecord: {groupName} idx={experimentIndex} samples={len(currentSeries)}")
    return record


def StackExperimentGroups(measurementData, groupNames: Sequence[str] = ("CHC", "DCC", "DCP", "CHP")) -> List[Dict[str, np.ndarray]]:
    """Stack all requested experiment groups into a flat Python list."""

    stackedRecords: List[Dict[str, np.ndarray]] = []

    total = 0
    for groupName in groupNames:
        print(f"[electrical_model7] StackExperimentGroups: checking group {groupName}")
        if not hasattr(measurementData.fu, groupName):
            print(f"[electrical_model7] StackExperimentGroups: group {groupName} not found, skipping")
            continue

        experimentCollection = NormalizeExperimentCollection(getattr(measurementData.fu, groupName))
        print(f"[electrical_model7] StackExperimentGroups: found {len(experimentCollection)} experiments in {groupName}")
        for experimentIndex, experiment in enumerate(experimentCollection):
            stackedRecords.append(BuildExperimentRecord(experiment, groupName, experimentIndex))
            total += 1

    print(f"[electrical_model7] StackExperimentGroups: stacked total {total} experiments")
    return stackedRecords


# ============================================================
# MODEL EQUATIONS
# ============================================================


def Sigmoid(value):
    """Smoothly map an unconstrained raw value to [0, 1]."""

    return 1.0 / (1.0 + np.exp(-value))


def SoftPlus(value):
    """Smoothly map an unconstrained raw value to a positive number."""

    return np.log1p(np.exp(-np.abs(value))) + np.maximum(value, 0.0)


def Logit(value):
    """Inverse of Sigmoid, with clipping for numerical safety."""

    clippedValue = np.clip(value, 1e-6, 1.0 - 1e-6)
    return np.log(clippedValue / (1.0 - clippedValue))


class FitInterrupted(RuntimeError):
    """Raised when a fit should stop early but return the best result so far."""


class ResidualAttemptTracker:
    """Track residual evaluations, best-so-far state, and stop conditions."""

    def __init__(
        self,
        experimentRecords: Sequence[Dict[str, np.ndarray]],
        *,
        statusEvery: int = 1,
        checkpointPath: Path | str | None = None,
        stopSignalPath: Path | str | None = None,
        wallClockLimitSeconds: float | None = None,
        verbose: bool = True,
    ) -> None:
        self.experimentRecords = experimentRecords
        self.statusEvery = max(1, int(statusEvery))
        self.checkpointPath = Path(checkpointPath) if checkpointPath is not None else None
        self.stopSignalPath = Path(stopSignalPath) if stopSignalPath is not None else None
        self.wallClockLimitSeconds = wallClockLimitSeconds
        self.verbose = verbose
        self.attemptCount = 0
        self.bestAttempt = 0
        self.bestCost = np.inf
        self.bestParameterVector: np.ndarray | None = None
        self.bestResidualVector: np.ndarray | None = None
        self.startTime = time.perf_counter()
        self.parameterNames: List[str] = []

    def _should_stop(self) -> str | None:
        if self.stopSignalPath is not None and self.stopSignalPath.exists():
            return f"stop signal detected at {self.stopSignalPath}"

        if self.wallClockLimitSeconds is not None:
            elapsedSeconds = time.perf_counter() - self.startTime
            if elapsedSeconds >= self.wallClockLimitSeconds:
                return f"wall clock limit reached after {elapsedSeconds:.1f}s"

        return None

    def _writeCheckpoint(self, reason: str) -> None:
        if self.checkpointPath is None or self.bestParameterVector is None:
            return

        self.checkpointPath.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.checkpointPath,
            reason=np.array(reason),
            attemptCount=np.array(self.attemptCount, dtype=int),
            bestAttempt=np.array(self.bestAttempt, dtype=int),
            bestCost=np.array(self.bestCost, dtype=float),
            elapsedSeconds=np.array(time.perf_counter() - self.startTime, dtype=float),
            parameterNames=np.array(self.parameterNames, dtype=object),
            bestParameterVector=self.bestParameterVector,
            bestResidualVector=self.bestResidualVector,
        )

    def evaluate(self, parameterVector: np.ndarray) -> np.ndarray:
        stopReason = self._should_stop()
        if stopReason is not None:
            raise FitInterrupted(stopReason)

        self.attemptCount += 1
        if self.verbose and (self.attemptCount == 1 or self.attemptCount % self.statusEvery == 0):
            if np.isfinite(self.bestCost):
                print(
                    f"[electrical_model7] FitModel7: attempt {self.attemptCount}, best cost so far {self.bestCost:.6f}"
                )
            else:
                print(f"[electrical_model7] FitModel7: attempt {self.attemptCount}, no best yet")

        residualVector = BuildResidualVector(
            parameterVector,
            self.experimentRecords,
            verbose=False,
            attemptNumber=self.attemptCount,
        )
        cost = 0.5 * float(np.dot(residualVector, residualVector))

        if cost < self.bestCost:
            self.bestCost = cost
            self.bestAttempt = self.attemptCount
            self.bestParameterVector = parameterVector.copy()
            self.bestResidualVector = residualVector.copy()
            if self.verbose:
                print(
                    f"[electrical_model7] FitModel7: new best at attempt {self.bestAttempt} with cost {self.bestCost:.6f}"
                )
            self._writeCheckpoint("improved")

        return residualVector


def BuildParameterVector(experimentRecords: Sequence[Dict[str, np.ndarray]]) -> Tuple[np.ndarray, List[str]]:
    """Create an initial parameter vector and the matching parameter names."""
    print(f"[electrical_model7] BuildParameterVector: creating initial vector for {len(experimentRecords)} experiments")
    parameterNames = [
        "capacityRaw",
        "ocv0",
        "ocv1",
        "ocv2",
        "ocv3",
        "ocv4",
        "ocv5",
        "ocvTemp",
        "hyst0",
        "hyst1",
        "hyst2",
        "hystTemp",
        "r0Base",
        "r0Soc1",
        "r0Soc2",
        "r0Temp",
        "rFastRaw",
        "tauFastRaw",
        "rSlowRaw",
        "tauSlowRaw",
        "hystGainRaw",
        "hystTauRaw",
        "hystShapeRaw",
        "hystCurrentScaleRaw",
        "tempVoltageCoeff",
    ]

    startSocNames = [f"startSocRaw_{index}" for index in range(len(experimentRecords))]
    parameterNames.extend(startSocNames)

    parameterVector = np.array(
        [
            Logit(3.0 / 4.0),
            3.05,
            1.25,
            -0.95,
            0.65,
            -0.25,
            0.12,
            0.001,
            0.03,
            -0.02,
            0.01,
            0.0005,
            np.log(np.exp(0.018) - 1.0),
            0.004,
            -0.003,
            0.0006,
            np.log(np.exp(0.006) - 1.0),
            np.log(np.exp(1.5) - 1.0),
            np.log(np.exp(0.0025) - 1.0),
            np.log(np.exp(30.0) - 1.0),
            np.log(np.exp(0.020) - 1.0),
            np.log(np.exp(80.0) - 1.0),
            np.log(np.exp(1.2) - 1.0),
            np.log(np.exp(5.0) - 1.0),
            0.001,
        ]
        + [Logit(record["initialSocGuess"]) for record in experimentRecords],
        dtype=float,
    )

    print(f"[electrical_model7] BuildParameterVector: parameter vector length {parameterVector.size}")
    return parameterVector, parameterNames


def UnpackParameters(parameterVector: np.ndarray, experimentCount: int) -> Dict[str, np.ndarray]:
    """Split a raw parameter vector into transformed model parameters."""
    print("[electrical_model7] UnpackParameters: unpacking parameter vector")
    cursor = 0

    capacityRaw = parameterVector[cursor]
    cursor += 1

    ocvCoefficients = parameterVector[cursor:cursor + 6]
    cursor += 6

    ocvTemperatureCoeff = parameterVector[cursor]
    cursor += 1

    hystCoefficients = parameterVector[cursor:cursor + 4]
    cursor += 4

    resistanceCoefficients = parameterVector[cursor:cursor + 4]
    cursor += 4

    rFastRaw = parameterVector[cursor]
    cursor += 1
    tauFastRaw = parameterVector[cursor]
    cursor += 1
    rSlowRaw = parameterVector[cursor]
    cursor += 1
    tauSlowRaw = parameterVector[cursor]
    cursor += 1
    hystGainRaw = parameterVector[cursor]
    cursor += 1
    hystTauRaw = parameterVector[cursor]
    cursor += 1
    hystShapeRaw = parameterVector[cursor]
    cursor += 1
    hystCurrentScaleRaw = parameterVector[cursor]
    cursor += 1
    tempVoltageCoeff = parameterVector[cursor]
    cursor += 1

    startSocRaw = parameterVector[cursor:cursor + experimentCount]

    return {
        "capacityAh": SoftPlus(capacityRaw) + 0.5,
        "ocvCoefficients": ocvCoefficients,
        "ocvTemperatureCoeff": ocvTemperatureCoeff,
        "hystCoefficients": hystCoefficients,
        "resistanceCoefficients": resistanceCoefficients,
        "rFast": SoftPlus(rFastRaw) + 1e-4,
        "tauFast": SoftPlus(tauFastRaw) + 1e-3,
        "rSlow": SoftPlus(rSlowRaw) + 1e-4,
        "tauSlow": SoftPlus(tauSlowRaw) + 1e-3,
        "hystGain": SoftPlus(hystGainRaw) + 1e-4,
        "hystTau": SoftPlus(hystTauRaw) + 1e-3,
        "hystShape": 1.0 + SoftPlus(hystShapeRaw),
        "hystCurrentScale": SoftPlus(hystCurrentScaleRaw) + 0.05,
        "tempVoltageCoeff": tempVoltageCoeff,
        "startSoc": Sigmoid(startSocRaw),
    }


def EvaluateOcvSurface(socValue: float, temperatureDelta: float, currentDirection: float, parameters: Dict[str, np.ndarray]) -> float:
    """Open-circuit voltage surface.

    The base curve captures the SOC shape, the temperature term nudges the
    entire curve with temperature, and the direction term allows a small,
    smooth charge/discharge split.
    """

    eps = 1e-9
    clippedSoc = np.clip(socValue, eps, 1.0 - eps)
    socBasis = np.array([
        1.0,
        clippedSoc,
        clippedSoc ** 2,
        clippedSoc ** 3,
        np.log(clippedSoc),
        np.log(1.0 - clippedSoc),
    ])

    baseCurve = float(np.dot(parameters["ocvCoefficients"], socBasis))
    temperatureShift = parameters["ocvTemperatureCoeff"] * temperatureDelta

    directionBasis = np.array([1.0, clippedSoc, clippedSoc ** 2, temperatureDelta])
    directionShift = float(np.dot(parameters["hystCoefficients"], directionBasis))

    return baseCurve + temperatureShift + currentDirection * directionShift


def EvaluateResistanceSurface(socValue: float, temperatureDelta: float, parameters: Dict[str, np.ndarray]) -> float:
    """SOC and temperature dependent ohmic resistance."""

    resistanceBasis = np.array([1.0, socValue, socValue ** 2, temperatureDelta])
    rawResistance = float(np.dot(parameters["resistanceCoefficients"], resistanceBasis))
    return SoftPlus(rawResistance) + 1e-5


def EvaluateHysteresisDrive(socValue: float, temperatureDelta: float, currentDirection: float, parameters: Dict[str, np.ndarray]) -> float:
    """Drive term for the hysteresis state.

    This term is not the hysteresis voltage itself. It is the direction-aware
    quantity that the state relaxes toward.
    """

    socGate = 1.0 / (1.0 + np.exp(-(socValue - 0.5) / 0.08))
    socBasis = np.array([1.0, socValue, socValue ** 2, temperatureDelta])
    driveShape = float(np.dot(parameters["hystCoefficients"], socBasis))
    return currentDirection * socGate * driveShape


def SimulateExperiment(
    experimentRecord: Dict[str, np.ndarray],
    parameters: Dict[str, np.ndarray],
    experimentIndex: int,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Run the battery model forward for one experiment.

    The model is causal. Each step updates SOC, the two polarization states,
    and the hysteresis state from the previous sample.
    """

    currentSeries = experimentRecord["current"]
    voltageSeries = experimentRecord["voltage"]
    timeSeries = experimentRecord["time"]
    temperatureSeries = experimentRecord["temperature"]
    sampleCount = currentSeries.shape[0]

    if verbose:
        print(f"[electrical_model7] SimulateExperiment: start exp={experimentIndex} samples={sampleCount}")
    progressInterval = max(1, sampleCount // 5)

    predictedVoltage = np.empty(sampleCount, dtype=float)
    socSeries = np.empty(sampleCount, dtype=float)
    fastPolarization = np.empty(sampleCount, dtype=float)
    slowPolarization = np.empty(sampleCount, dtype=float)
    hysteresisState = np.empty(sampleCount, dtype=float)

    socSeries[0] = float(parameters["startSoc"][experimentIndex])
    fastPolarization[0] = 0.0
    slowPolarization[0] = 0.0
    hysteresisState[0] = 0.0

    for sampleIndex in range(sampleCount):
        if verbose and sampleIndex % progressInterval == 0:
            print(f"[electrical_model7] SimulateExperiment: exp={experimentIndex} progress {sampleIndex}/{sampleCount}")
        if sampleIndex > 0:
            deltaTime = max(float(timeSeries[sampleIndex] - timeSeries[sampleIndex - 1]), 1e-6)
            previousCurrent = float(currentSeries[sampleIndex - 1])

            socSeries[sampleIndex] = np.clip(
                socSeries[sampleIndex - 1] - (previousCurrent * deltaTime) / (3600.0 * parameters["capacityAh"]),
                0.0,
                1.0,
            )

            fastAlpha = np.exp(-deltaTime / parameters["tauFast"])
            slowAlpha = np.exp(-deltaTime / parameters["tauSlow"])
            hystAlpha = np.exp(-deltaTime / parameters["hystTau"])

            fastPolarization[sampleIndex] = (
                fastAlpha * fastPolarization[sampleIndex - 1]
                + parameters["rFast"] * (1.0 - fastAlpha) * previousCurrent
            )
            slowPolarization[sampleIndex] = (
                slowAlpha * slowPolarization[sampleIndex - 1]
                + parameters["rSlow"] * (1.0 - slowAlpha) * previousCurrent
            )

            currentDirection = np.tanh(previousCurrent / parameters["hystCurrentScale"])
            temperatureDelta = float(temperatureSeries[sampleIndex - 1] - 298.15)
            hysteresisDrive = EvaluateHysteresisDrive(
                socValue=socSeries[sampleIndex],
                temperatureDelta=temperatureDelta,
                currentDirection=currentDirection,
                parameters=parameters,
            )
            hysteresisState[sampleIndex] = (
                hystAlpha * hysteresisState[sampleIndex - 1]
                + (1.0 - hystAlpha) * hysteresisDrive
            )
        else:
            currentDirection = np.tanh(float(currentSeries[sampleIndex]) / parameters["hystCurrentScale"])
            socSeries[sampleIndex] = socSeries[0]
            fastPolarization[sampleIndex] = 0.0
            slowPolarization[sampleIndex] = 0.0
            temperatureDelta = float(temperatureSeries[sampleIndex] - 298.15)
            hysteresisState[sampleIndex] = EvaluateHysteresisDrive(
                socValue=socSeries[sampleIndex],
                temperatureDelta=temperatureDelta,
                currentDirection=currentDirection,
                parameters=parameters,
            )

        currentDirection = np.tanh(float(currentSeries[sampleIndex]) / parameters["hystCurrentScale"])
        temperatureDelta = float(temperatureSeries[sampleIndex] - 298.15)
        ocvValue = EvaluateOcvSurface(
            socValue=float(socSeries[sampleIndex]),
            temperatureDelta=temperatureDelta,
            currentDirection=currentDirection,
            parameters=parameters,
        )
        resistanceValue = EvaluateResistanceSurface(
            socValue=float(socSeries[sampleIndex]),
            temperatureDelta=temperatureDelta,
            parameters=parameters,
        )

        predictedVoltage[sampleIndex] = (
            ocvValue
            - float(currentSeries[sampleIndex]) * resistanceValue
            - fastPolarization[sampleIndex]
            - slowPolarization[sampleIndex]
            + parameters["hystGain"] * hysteresisState[sampleIndex]
            + parameters["tempVoltageCoeff"] * temperatureDelta
        )

    return {
        "predictedVoltage": predictedVoltage,
        "soc": socSeries,
        "fastPolarization": fastPolarization,
        "slowPolarization": slowPolarization,
        "hysteresisState": hysteresisState,
        "measuredVoltage": voltageSeries,
    }


def BuildResidualVector(
    parameterVector: np.ndarray,
    experimentRecords: Sequence[Dict[str, np.ndarray]],
    verbose: bool = True,
    attemptNumber: int | None = None,
) -> np.ndarray:
    """Convert the full fit problem into one stacked residual vector."""
    if verbose:
        if attemptNumber is None:
            print(f"[electrical_model7] BuildResidualVector: building residuals for {len(experimentRecords)} experiments")
        else:
            print(
                f"[electrical_model7] BuildResidualVector: attempt {attemptNumber}, building residuals for {len(experimentRecords)} experiments"
            )
    parameters = UnpackParameters(parameterVector, experimentCount=len(experimentRecords))
    residualParts: List[np.ndarray] = []

    for experimentIndex, experimentRecord in enumerate(experimentRecords):
        simulationResult = SimulateExperiment(experimentRecord, parameters, experimentIndex, verbose=verbose)
        residualParts.append(simulationResult["predictedVoltage"] - simulationResult["measuredVoltage"])

    return np.concatenate(residualParts)


def FitModel7(
    experimentRecords: Sequence[Dict[str, np.ndarray]],
    verbose: bool = True,
    statusEvery: int = 1,
    wallClockLimitSeconds: float | None = None,
    checkpointPath: Path | str | None = None,
    stopSignalPath: Path | str | None = None,
    maxNfev: int | None = 20,
):
    """Fit the medium-complexity lithium-ion model to all stacked experiments."""
    print(f"[electrical_model7] FitModel7: starting fit for {len(experimentRecords)} experiments")
    initialParameterVector, parameterNames = BuildParameterVector(experimentRecords)
    print(f"[electrical_model7] FitModel7: initial parameter vector length {initialParameterVector.size}")

    tracker = ResidualAttemptTracker(
        experimentRecords,
        statusEvery=statusEvery,
        checkpointPath=checkpointPath,
        stopSignalPath=stopSignalPath,
        wallClockLimitSeconds=wallClockLimitSeconds,
        verbose=verbose,
    )
    tracker.parameterNames = parameterNames

    fitResult = None
    stoppedEarly = False
    stopReason = None

    try:
        fitResult = least_squares(
            fun=tracker.evaluate,
            x0=initialParameterVector,
            method="trf",
            loss="soft_l1",
            f_scale=0.01,
            max_nfev=maxNfev,
        )
        print("[electrical_model7] FitModel7: least_squares finished")
    except (KeyboardInterrupt, FitInterrupted) as interruption:
        stoppedEarly = True
        stopReason = str(interruption)
        print(f"[electrical_model7] FitModel7: stopped early -> {stopReason}")

    if tracker.bestParameterVector is None:
        tracker.bestParameterVector = initialParameterVector.copy()
        tracker.bestResidualVector = BuildResidualVector(initialParameterVector, experimentRecords, verbose=False)
        tracker.bestCost = 0.5 * float(np.dot(tracker.bestResidualVector, tracker.bestResidualVector))
        tracker.bestAttempt = 0

    bestParameterVector = tracker.bestParameterVector
    bestResidualVector = tracker.bestResidualVector
    if checkpointPath is not None:
        tracker._writeCheckpoint(stopReason or "completed")

    fittedParameters = UnpackParameters(bestParameterVector, experimentCount=len(experimentRecords))
    fittedResults = []
    for experimentIndex, experimentRecord in enumerate(experimentRecords):
        fittedResults.append(SimulateExperiment(experimentRecord, fittedParameters, experimentIndex, verbose=False))

    if verbose:
        residualVector = bestResidualVector
        rmse = float(np.sqrt(np.mean(residualVector ** 2)))
        mae = float(np.mean(np.abs(residualVector)))
        if fitResult is not None:
            print("Fit success:", fitResult.success)
            print("Message:", fitResult.message)
        else:
            print("Fit stopped early before least_squares returned a result object")
        print("Stopped early:", stoppedEarly)
        print("Stop reason:", stopReason)
        print("Attempts:", tracker.attemptCount)
        print("Best attempt:", tracker.bestAttempt)
        print(f"RMSE: {rmse:.6f} V")
        print(f"MAE:  {mae:.6f} V")

    return {
        "fitResult": fitResult,
        "parameterNames": parameterNames,
        "parameterVector": bestParameterVector,
        "fittedParameters": fittedParameters,
        "experimentRecords": list(experimentRecords),
        "fittedResults": fittedResults,
        "attemptCount": tracker.attemptCount,
        "bestAttempt": tracker.bestAttempt,
        "bestCost": tracker.bestCost,
        "stoppedEarly": stoppedEarly,
        "stopReason": stopReason,
    }


# ============================================================
# INTERACTIVE HELPERS
# ============================================================


def BuildDefaultTrainingSet(groupNames: Sequence[str] = ("CHC", "DCC", "DCP", "CHP")) -> List[Dict[str, np.ndarray]]:
    """Load the MAT file and stack all available experiment groups."""
    print(f"[electrical_model7] BuildDefaultTrainingSet: groups={groupNames}")
    measurementData = LoadMeasurementData()
    records = StackExperimentGroups(measurementData, groupNames=groupNames)
    print(f"[electrical_model7] BuildDefaultTrainingSet: total records={len(records)}")
    return records


def PlotExperimentFit(experimentRecord: Dict[str, np.ndarray], simulationResult: Dict[str, np.ndarray], titlePrefix: str = "Model 7"):
    """Plot measured and modeled voltage for one experiment."""

    print(f"[electrical_model7] PlotExperimentFit: plotting experiment {experimentRecord.get('groupName')} idx={experimentRecord.get('experimentIndex')}")

    import matplotlib.pyplot as plt

    timeSeries = experimentRecord["time"]
    measuredVoltage = simulationResult["measuredVoltage"]
    predictedVoltage = simulationResult["predictedVoltage"]
    currentSeries = experimentRecord["current"]

    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(timeSeries, currentSeries, color="tab:blue")
    axes[0].set_ylabel("Current [A]")
    axes[0].grid(True)

    axes[1].plot(timeSeries, measuredVoltage, label="Measured", color="tab:orange")
    axes[1].plot(timeSeries, predictedVoltage, label="Modeled", color="tab:green", alpha=0.85)
    axes[1].set_ylabel("Voltage [V]")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(timeSeries, simulationResult["soc"], color="tab:purple")
    axes[2].set_ylabel("SOC")
    axes[2].set_xlabel("Time [s]")
    axes[2].grid(True)

    groupName = str(experimentRecord["groupName"])
    experimentIndex = int(experimentRecord["experimentIndex"])
    figure.suptitle(f"{titlePrefix}: {groupName}_{experimentIndex}")
    figure.tight_layout()
    return figure, axes


def SummarizeModelFit(fitPackage: Dict[str, object]) -> Dict[str, float]:
    """Return high-level error metrics for the full stacked fit."""

    print("[electrical_model7] SummarizeModelFit: computing summary metrics")
    experimentRecords = fitPackage["experimentRecords"]
    fittedParameters = fitPackage["fittedParameters"]

    residualParts = []
    for experimentIndex, experimentRecord in enumerate(experimentRecords):
        simulationResult = SimulateExperiment(experimentRecord, fittedParameters, experimentIndex)
        residualParts.append(simulationResult["predictedVoltage"] - simulationResult["measuredVoltage"])

    residualVector = np.concatenate(residualParts)
    return {
        "rmse": float(np.sqrt(np.mean(residualVector ** 2))),
        "mae": float(np.mean(np.abs(residualVector))),
        "maxAbsError": float(np.max(np.abs(residualVector))),
    }


# ============================================================
# USAGE EXAMPLE
# ============================================================

# The file intentionally does not auto-run a fit on import.
if __name__ == "__main__":
    print("[electrical_model7] Example run: building default training set...")
    experimentRecords = BuildDefaultTrainingSet()
    print("[electrical_model7] Example run: fitting model to stacked experiments...")
    fitPackage = FitModel7(experimentRecords)
    print("[electrical_model7] Example run: summarizing fit metrics...")
    metrics = SummarizeModelFit(fitPackage)
    print(f"[electrical_model7] Example run: metrics={metrics}")
    firstExperimentResult = fitPackage["fittedResults"][0]
    print("[electrical_model7] Example run: plotting first experiment fit...")
    PlotExperimentFit(experimentRecords[0], firstExperimentResult)


# ============================================================
# COMPACT MODEL EQUATIONS AND PARAMETERS
# ============================================================
# Use this block as the single reference form for testing the model.
#
# State updates for sample k -> k+1:
#   SOC[k+1] = clip(SOC[k] - I[k] * dt / (3600 * C), 0, 1)
#   a_f = exp(-dt / tau_f)
#   a_s = exp(-dt / tau_s)
#   a_h = exp(-dt / tau_h)
#   x_f[k+1] = a_f * x_f[k] + R_f * (1 - a_f) * I[k]
#   x_s[k+1] = a_s * x_s[k] + R_s * (1 - a_s) * I[k]
#   dir[k]   = tanh(I[k] / I_scale)
#   T_delta  = T[k] - 298.15
#   g_soc    = 1 / (1 + exp(-(SOC[k] - 0.5) / 0.08))
#   h_drive  = dir[k] * g_soc * (h0 + h1*SOC[k] + h2*SOC[k]^2 + hT*T_delta)
#   h[k+1]   = a_h * h[k] + (1 - a_h) * h_drive
#
# Voltage model:
#   OCV(SOC, T, dir) =
#       ocv0 + ocv1*SOC + ocv2*SOC^2 + ocv3*SOC^3 + ocv4*ln(SOC) + ocv5*ln(1-SOC)
#       + ocvT * T_delta
#       + dir * (d0 + d1*SOC + d2*SOC^2 + dT*T_delta)
#
#   R0(SOC, T) = softplus(r0 + r1*SOC + r2*SOC^2 + rT*T_delta) + 1e-5
#
#   V_hat[k] = OCV(SOC[k], T_delta[k], dir[k])
#              - I[k] * R0(SOC[k], T_delta[k])
#              - x_f[k] - x_s[k]
#              + h_g * h[k]
#              + vT * T_delta[k]
#
# Parameters (raw vector order before transforms):
#   capacityRaw
#   ocv0, ocv1, ocv2, ocv3, ocv4, ocv5, ocvTemp
#   hyst0, hyst1, hyst2, hystTemp
#   r0Base, r0Soc1, r0Soc2, r0Temp
#   rFastRaw, tauFastRaw, rSlowRaw, tauSlowRaw
#   hystGainRaw, hystTauRaw, hystShapeRaw, hystCurrentScaleRaw
#   tempVoltageCoeff
#   startSocRaw_0 ... startSocRaw_(N-1)
#
# Transforms:
#   C         = softplus(capacityRaw) + 0.5
#   R_f       = softplus(rFastRaw) + 1e-4
#   tau_f     = softplus(tauFastRaw) + 1e-3
#   R_s       = softplus(rSlowRaw) + 1e-4
#   tau_s     = softplus(tauSlowRaw) + 1e-3
#   h_g       = softplus(hystGainRaw) + 1e-4
#   tau_h     = softplus(hystTauRaw) + 1e-3
#   h_shape   = 1 + softplus(hystShapeRaw)
#   I_scale   = softplus(hystCurrentScaleRaw) + 0.05
#   SOC0_i    = sigmoid(startSocRaw_i)
