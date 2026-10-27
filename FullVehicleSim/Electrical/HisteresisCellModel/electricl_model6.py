import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import scipy.io
from scipy.integrate import simpson, cumulative_simpson
import matplotlib.patches as patches
from pysr import PySRRegressor, TemplateExpressionSpec

MAT_PATH = "../fs-data/FS-4/P30B Cell Data/Molicel_INR18650P30B_measurement.mat"



mat = scipy.io.loadmat(MAT_PATH, squeeze_me=True, struct_as_record=False)

'''
header = batemoData["__header__"]
version = batemoData["__version__"]
globals = batemoData["__globals__"] ## Nothing
print(f"header = {header}")
print(f"version = {version}")
measurement = batemoData["measurement"]
firstLayerMeta = measurement['meta']
Fu = measurement['fu']

# DCC (Discharge)
# CHC (Charging)
# DCP (Discharge Pulse)
# CHP (Charge Pulse)
PRO (Profile Measurement)

Each Has
    name
    T_amb (Ambient Temperature)
    t (Time Seconds)
    I (Current)
    V (Voltage)
    T_surf (Surface temperature, nominally 1xN but may be 2xN?)
'''

simulation = mat["measurement"]

fullDCC = pl.DataFrame(schema={
    "I":pl.Float64,
    "V":pl.Float64,
    "t":pl.Float64,
    "T_surf1":pl.Float64,
    "T_surf2":pl.Float64,
    "SOC":pl.Float64,
    "I_Past1":pl.Float64,
    "I_Past2":pl.Float64,
    "I_Past3":pl.Float64,
    "I_Past4":pl.Float64,
})

for dcc in simulation.fu.DCC:
    I = dcc.I
    I_back1 = np.concatenate([np.zeros(1), I[:-1]])
    I_back2 = np.concatenate([np.zeros(3), I[:-3]])
    I_back3 = np.concatenate([np.zeros(7), I[:-7]])
    I_back4 = np.concatenate([np.zeros(20), I[:-20]])
    V = dcc.V
    t = dcc.t
    T_surf = dcc.T_surf
    T_surf1 = T_surf[0, :]
    T_surf2 = T_surf[1, :]
    SOC = 3.0 - cumulative_simpson(y=I, x=t)/3600.0
    SOC = np.concatenate([np.array([3.0]), SOC])
    fullDCC = fullDCC.vstack(pl.DataFrame({
        "I": I,
        "V": V,
        "t": t,
        "T_surf1": T_surf1,
        "T_surf2": T_surf2,
        "SOC": SOC,
        "I_Past1": I_back1,
        "I_Past2": I_back2,
        "I_Past3": I_back3,
        "I_Past4": I_back4,
    }))

# simulation.fu.DCC
# len(simulation.fu.DCC)
# for dcc in simulation.fu.DCC:
#     I = dcc.I
#     V = dcc.V
#     t = dcc.t
#     T_amb = dcc.T_amb
#     T_surf = dcc.T_surf
#     name = dcc.name
#     SOC = 3.0 - cumulative_simpson(y=I, x=t)/3600.0
#     SOC = np.concatenate([np.array([3.0]), SOC])
#     fig = plt.figure()
#     ax1 = fig.add_subplot(111)
#     ax2 = ax1.twinx()
#     ax1.plot(SOC, I, label="Current", color="blue")
#     ax2.plot(SOC, V, label="Voltage", color="orange")
#     ## put patches for legends with current and voltage on the same legend with current blue nad voltage orange
#     current_patch = patches.Patch(color='blue', label='Current [A]')
#     voltage_patch = patches.Patch(color='orange', label='Voltage [V]')
#     plt.legend(handles=[current_patch, voltage_patch])
#     plt.title(f"DCC: {name} at T_amb={T_amb} K")
#     plt.xlabel("SOC")
#     plt.grid(True)
#     plt.show()

# SOC = 3.0 - cumulative_simpson(y=simulation.fu.DCC[0].I, x=simulation.fu.DCC[0].t)/3600.0
# SOC = np.concatenate([np.array([3.0]), SOC])

X = fullDCC.select(pl.col("I"), pl.col("SOC"), pl.col("T_surf1"), pl.col("T_surf2"), pl.col("I_Past1"), pl.col("I_Past2"), pl.col("I_Past3"), pl.col("I_Past4")).to_numpy()
y = fullDCC.select(pl.col("V")).to_numpy()

# "I": I,
# "V": V,
# "t": t,
# "T_surf1": T_surf1,
# "T_surf2": T_surf2,
# "SOC": SOC,

template = TemplateExpressionSpec(
    expressions=["d", "e", "f", "g", "h"],
    variable_names=["x1", "x2", "x3", "x4"],
    parameters={"p1":1, "p2":1, "p3":1, "p4":1, "p5":1},
    combine="d(h(x1, x2) * log((f(x2))/(e(1 - x2)))) + g(x1)",
)

template = TemplateExpressionSpec(
    expressions=["g", "h"],
    variable_names=["x1", "x2", "x3", "x4"],
    parameters={"p1":1, "p2":1, "p3":1, "p4":1, "p5":1},
    combine="h(x1) * g(x2)",
)

model = PySRRegressor(
    maxsize=80,
    niterations=1000,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["exp", "log"],
    batching=True,
    batch_size=200,
)

output = model.fit(X, y)

cos = lambda x: np.cos(x)
sin = lambda x: np.sin(x)
exp = lambda x: np.exp(x)
log = lambda x: np.log(x)

def custom_fun_1(X):
    x0 = X[:, 0]
    x1 = X[:, 1]
    x2 = X[:, 2]
    x3 = X[:, 3]
    return ((cos(x1 * 0.95293) * -0.59082) + cos(cos(sin(exp(x0 / 2.5028) + 0.46012)))) + 2.5555

def custom_fun_2(X):
    x0 = X[:, 0]
    x1 = X[:, 1]
    x2 = X[:, 2]
    x3 = X[:, 3]
    return (cos(cos(exp(x0 + cos(x1 + x0)) + 0.74591)) + 2.5539) + (cos(x1 + -0.062883) * -0.57004)

def custom_fun_3(X):
    x0 = X[:, 0]
    x1 = X[:, 1]
    x2 = X[:, 2]
    x3 = X[:, 3]
    return (log(((exp(x0) + 19.943527) + (x1 * -3.1378086)) * (((x0 + (exp(exp((x2 / exp(x2 + -1.089517)) / exp(x3))) + ((x3 + -1.7805654) / exp(exp(exp(x0)))))) * 0.07967347) + 4.030623)) * 0.81259966) - -0.9257864

def custom_fun_4(X):
    x0 = X[:, 0]
    x1 = X[:, 1]
    x2 = X[:, 2]
    x3 = X[:, 3]
    x4 = X[:, 4]
    x5 = X[:, 5]
    x6 = X[:, 6]
    x7 = X[:, 7]
    return ((x5 + ((((x2 / 1.3202298) + exp(x1)) / (exp(exp(x1) * 0.1189319) * 0.16781318)) + ((((x0 + x3) + exp((x6 + ((x1 - x3) * (exp(x6 - (((x2 - x1) + x3) * -0.033423472)) * 0.010805121))) + 4.2677464)) - exp(x1)) + ((((x7 + x5) + ((x3 * 0.40745232) + (x4 / (exp(x2) + 0.20448817)))) + -169.69832) * log(x1))))) * 0.003010543) + 4.5433464

for dcc in simulation.fu.DCC:
    I = dcc.I
    V = dcc.V
    t = dcc.t
    I_back1 = np.concatenate([np.zeros(1), I[:-1]])
    I_back2 = np.concatenate([np.zeros(3), I[:-3]])
    I_back3 = np.concatenate([np.zeros(7), I[:-7]])
    I_back4 = np.concatenate([np.zeros(20), I[:-20]])
    T_amb = dcc.T_amb
    T_surf = dcc.T_surf
    name = dcc.name
    SOC = 3.0 - cumulative_simpson(y=I, x=t)/3600.0
    SOC = np.concatenate([np.array([3.0]), SOC])
    plt.plot(SOC, V, label="Measured Voltage")
    plt.plot(SOC, custom_fun_4(np.column_stack([I, SOC, T_surf[0, :], T_surf[1, :], I_back1, I_back2, I_back3, I_back4])), label="Model 3")
    plt.legend()
    plt.title(f"DCC: {name} at T_amb={T_amb} C")
    plt.xlabel("SOC")
    plt.ylabel("Voltage [V]")
    plt.grid(True)
    plt.show()

plt.plot(X[:, 1], custom_fun_1(X[:, 1], X[:, 2]), label="Model 1")
plt.plot(X[:, 1], y, label="Measured Voltage")
plt.xlabel("SOC [Ah]")
plt.ylabel("Voltage [V]")
plt.legend()
plt.show()


# ============================================================
# VERSION 2: PARAMETERIZED HYSTERESIS KERNEL
# ============================================================

def buildHysteresisKernel(kernelLength, tauFast, tauSlow, mix=0.5, shape=1.0):
    kernelLength = int(kernelLength)
    if kernelLength <= 0:
        raise ValueError("kernelLength must be positive")

    tauFast = max(float(tauFast), 1e-9)
    tauSlow = max(float(tauSlow), 1e-9)
    mix = float(np.clip(mix, 0.0, 1.0))
    shape = max(float(shape), 1e-9)

    lagIndex = np.arange(kernelLength, dtype=float)
    fastComponent = np.exp(-((lagIndex / tauFast) ** shape))
    slowComponent = np.exp(-((lagIndex / tauSlow) ** shape))
    kernel = (mix * fastComponent) + ((1.0 - mix) * slowComponent)

    kernelSum = float(np.sum(kernel))
    if kernelSum == 0.0:
        return np.ones(kernelLength, dtype=float) / float(kernelLength)

    return kernel / kernelSum


def buildHysteresisWindow(currentSeries, kernelLength):
    currentSeries = np.asarray(currentSeries, dtype=float).reshape(-1)
    kernelLength = int(kernelLength)
    if kernelLength <= 0:
        raise ValueError("kernelLength must be positive")

    paddedSeries = np.pad(currentSeries, (kernelLength - 1, 0), mode="constant")
    windows = np.empty((currentSeries.shape[0], kernelLength), dtype=float)

    for sampleIndex in range(currentSeries.shape[0]):
        window = paddedSeries[sampleIndex:sampleIndex + kernelLength]
        windows[sampleIndex, :] = window[::-1]

    return windows


def buildHysteresisFeature(currentSeries, kernelLength, tauFast, tauSlow, mix=0.5, shape=1.0):
    currentWindows = buildHysteresisWindow(currentSeries, kernelLength)
    kernel = buildHysteresisKernel(kernelLength, tauFast, tauSlow, mix=mix, shape=shape)
    return currentWindows @ kernel


def buildHysteresisTrainingSet(simulationData, kernelLength=64, tauFast=4.0, tauSlow=28.0, mix=0.7, shape=1.15):
    featureRows = []
    targetRows = []

    for dcc in simulationData.fu.DCC:
        currentSeries = np.asarray(dcc.I, dtype=float).reshape(-1)
        voltageSeries = np.asarray(dcc.V, dtype=float).reshape(-1)
        timeSeries = np.asarray(dcc.t, dtype=float).reshape(-1)
        surfaceSeries = np.asarray(dcc.T_surf, dtype=float)

        if surfaceSeries.ndim == 1:
            surfaceOne = surfaceSeries
            surfaceTwo = surfaceSeries
        else:
            surfaceOne = surfaceSeries[0, :]
            surfaceTwo = surfaceSeries[1, :]

        socSeries = 3.0 - cumulative_simpson(y=currentSeries, x=timeSeries) / 3600.0
        socSeries = np.concatenate([np.array([3.0], dtype=float), socSeries])

        hysteresisSeries = buildHysteresisFeature(
            currentSeries=currentSeries,
            kernelLength=kernelLength,
            tauFast=tauFast,
            tauSlow=tauSlow,
            mix=mix,
            shape=shape,
        )

        featureRows.append(np.column_stack([
            currentSeries,
            socSeries,
            surfaceOne,
            surfaceTwo,
            hysteresisSeries,
        ]))
        targetRows.append(voltageSeries.reshape(-1, 1))

    features = np.vstack(featureRows)
    targets = np.vstack(targetRows).ravel()
    return features, targets


def trainHysteresisKernelVersion2(simulationData, kernelLength=128, tauFast=4.0, tauSlow=14.0, mix=0.7, shape=1.35):
    features, targets = buildHysteresisTrainingSet(
        simulationData=simulationData,
        kernelLength=kernelLength,
        tauFast=tauFast,
        tauSlow=tauSlow,
        mix=mix,
        shape=shape,
    )

    modelV2 = PySRRegressor(
        maxsize=30,
        niterations=40,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "log"],
        batching=True,
        batch_size=200,
    )

    fittedModel = modelV2.fit(features, targets)
    return fittedModel, features, targets

model, features, targets = trainHysteresisKernelVersion2(simulation)


# Optional reference entry point for the new version:
# fittedModel, features, targets = trainHysteresisKernelVersion2(simulation)