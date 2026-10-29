import polars as pl
import numpy as np
import matplotlib.pyplot as plt


dfComparison = pl.read_parquet("../fs-data/FS-3/03162026/2_steeper_regen_curve.parquet")
dfComparison = dfComparison.fill_null(strategy="forward").fill_null(strategy="backward")
dfComparison = dfComparison.with_columns(
    (pl.col("Time_ms")*0.001 - pl.col("Time_ms").min() * 0.001).alias("time"),
    ((dfComparison["TMAIN_DATA_STEERING"]-10.5)/180*np.pi).alias("steerAngle"),
    (dfComparison["ETC_STATUS_PEDAL_TRAVEL"]/100.0).alias("throttle"),
    pl.Series(np.clip(dfComparison["ETC_STATUS_BRAKE_SENSE_VOLTAGE"]-330, 0, 2640)/2640*2000).alias("brakePressureFront"),
    pl.Series(np.clip(dfComparison["ETC_STATUS_BRAKE_SENSE_VOLTAGE"]-330, 0, 2640)/2640*2000).alias("brakePressureRear")
)

df = pl.read_parquet("FullVehicleSim/simulation_output.parquet")
df = df.filter(pl.col("time") < dfComparison["time"].max())

## Columns
# ['time', 'throttle', 'brakePressureFront', 'brakePressureRear', 'steerAngle', 
#  'posX', 'posY', 'posZ', 'velX', 'velY', 'velZ', 'speed', 'headingX', 'headingY', 
#  'headingZ', 'yawRate', 'frontBrakeTemperature', 'rearBrakeTemperature', 'charge', 
#  'drag', 'resistiveForces', 'motorTorque', 'motorForce', 'netForce', 'maxTraction', 
#  'wheelRotationsHZ', 'motorRPM', 'motorRotationsHZ', 'current', 'maxWheelTorque', 
#  'maxPower', 'power', 'voltage', 'frontBrakeForce', 'rearBrakeForce', 'frontBrakeHeating', 
#  'rearBrakeHeating', 'frontBrakeCooling', 'rearBrakeCooling', 'frontSlipAngle', 
#  'rearSlipAngle', 'maxMotorTorque', 'acceleration', 'wheelRPM']

plt.plot(df["time"], df["motorRPM"], label="Sim")
plt.plot(dfComparison["time"], dfComparison["SME_TRQSPD_Speed"], label="Real")
plt.plot(df["time"], df["motorTorque"]*10, label="Motor Torque")
plt.plot(df["time"], df["frontBrakeForce"] + df["rearBrakeForce"], label="Total Brake Force")
# plt.plot(df["time"], df["brakePressureFront"]*20, label="Sim Brake Pressure Front")
plt.legend()
plt.show()


plt.plot(df["time"], df["voltage"]/30)
plt.plot(df["time"], df["charge"])
plt.plot(df["time"], df["current"], label="Current")
plt.plot(df["time"], df["power"], label="Power")
plt.plot(df["time"], df["motorRPM"], label="Motor RPM")
plt.plot(df["time"], df["motorTorque"], label="Motor Torque")
plt.plot(df["time"], df["throttle"], label="Throttle")
plt.plot(df["time"], df["maxMotorTorque"], label="Max Motor Torque")
plt.plot(df["time"], df["speed"]*2.237, label="Speed")
plt.plot(df["time"], df["frontBrakeForce"], label="Front Brake Force")
plt.plot(df["time"], df["rearBrakeForce"], label="Rear Brake Force")
plt.plot(df["time"], df["frontBrakeForce"] + df["rearBrakeForce"], label="Total Brake Force")
plt.plot(df["time"], df["drag"], label="Drag Force")
plt.legend()
plt.show()