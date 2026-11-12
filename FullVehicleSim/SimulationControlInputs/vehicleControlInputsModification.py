import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# time,throttle,brakePressureFront,brakePressureRear,steerAngle

df = pl.read_parquet("../fs-data/FS-3/03162026/2_steeper_regen_curve.parquet")
df = df.fill_null(strategy="forward").fill_null(strategy="backward")
df = df.with_columns(
    (pl.col("Time_ms")*0.001 - pl.col("Time_ms").min() * 0.001).alias("time"),
    ((df["TMAIN_DATA_STEERING"]-10.5)/180*np.pi).alias("steerAngle"),
    (df["ETC_STATUS_PEDAL_TRAVEL"]/100.0).alias("throttle"),
    pl.Series(np.clip(df["ETC_STATUS_BRAKE_SENSE_VOLTAGE"]-330, 0, 2640)/2640*2000).alias("brakePressureFront"),
    pl.Series(np.clip(df["ETC_STATUS_BRAKE_SENSE_VOLTAGE"]-330, 0, 2640)/2640*2000).alias("brakePressureRear"),
    (pl.col("ETC_STATUS_BRAKE_PEDAL_TRAVEL") / 106.7).alias("brakePedalTravel")
)

[x for x in df.columns if "SME" in x]

plt.plot(df["time"], (df["TMAIN_DATA_STEERING"]))
plt.show()

plt.plot(df["time"], (df["TMAIN_DATA_STEERING"]-10.5)/180*np.pi)
plt.show()


dfT = df.filter(pl.col("time") < 50).filter(pl.col("time") > 48)
dfT = df
plt.scatter(dfT["VDM_GPS_Latitude"], -1*dfT["VDM_GPS_Longitude"], c=dfT["TMAIN_DATA_STEERING"], cmap="viridis", s=0.5)
plt.colorbar(label="Steering Angle")
plt.axis("scaled")
plt.ylabel("Latitude")
plt.xlabel("Longitude")
plt.show()

plt.plot(df["time"], df["ETC_STATUS_PEDAL_TRAVEL"])
plt.show()

plt.plot(df["time"], df["ETC_STATUS_BRAKE_SENSE_VOLTAGE"])
plt.show()

plt.plot(df["time"], df["brakePressureFront"])
plt.show()

df.select(["time", "throttle", "brakePressureFront","brakePressureRear", "brakePedalTravel", "steerAngle"]).write_csv("FullVehicleSim/SimulationControlInputs/simulationControls_Mar162026_SteeperRegenCurve.csv")

def fun (x, a):
    return a * x

dfHere = df.filter(pl.col("SME_THROTL_TorqueDemand") < 0)
plt.scatter(dfHere["ETC_STATUS_BRAKE_PEDAL_TRAVEL"], dfHere["SME_THROTL_TorqueDemand"], s=0.5)
plt.plot(np.arange(0, 60, 1), fun(np.arange(0, 60, 1), -300))
plt.xlabel("Brake Pedal Travel")
plt.ylabel("Torque Demand")
plt.show()

plt.plot(df["time"], df["ETC_STATUS_BRAKE_PEDAL_TRAVEL"]/100, label="Brake Pedal Travel")
plt.plot(df["time"], df["ETC_STATUS_PEDAL_TRAVEL"]/100, label="Accelerator Pedal Travel")
plt.plot(df["time"], df["SME_THROTL_TorqueDemand"]/30000, label="Torque Demand")
plt.plot(df["time"], (df["ETC_STATUS_PEDAL_TRAVEL"]/100) - (df["ETC_STATUS_BRAKE_PEDAL_TRAVEL"]/100), label="Combined Pedal Travel")
plt.plot(df["time"], df["SME_TEMP_FaultCode"], label="Fault Code")
plt.xlabel("Time (s)")
plt.legend()
plt.show()
