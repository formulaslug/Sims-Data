import polars as pl
import matplotlib.pyplot as plt

df = pl.read_parquet("FullVehicleSim/simulation_output.parquet")
t = df["time"]


plt.plot(t, df["throttle"]*300, label="throttle")
plt.plot(t, df["brakePressureFront"], label="brakesF")
plt.plot(t, df["netForce"], label="netForce")
plt.plot(t, df["motorForce"], label="motorForce")
plt.plot(t, df["motorTorque"], label="motorTorque")
# plt.plot(t, df["motorRPM"], label="motorRPM")
plt.plot(t, df["speed"], label="speed")
plt.legend()
plt.show()

df["speed"].max()