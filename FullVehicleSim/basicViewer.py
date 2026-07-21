import polars as pl
import matplotlib.pyplot as plt

df = pl.read_parquet("FullVehicleSim/simulation_output.parquet")
dfReal = pl.read_parquet("../fs-data/FS-3/03162026/2_steeper_regen_curve.parquet").fill_null(strategy="forward").fill_null(strategy="backward")
t = df["time"]

dfReal["Time_ms"][-1]

[x for x in dfReal.columns if "SME" in x]



df.columns

plt.plot(t, df["posX"])
plt.show()

plt.plot(t, df["throttle"]*300, label="throttle")
plt.plot(t, df["brakePressureFront"], label="brakesF")
plt.plot(t, df["netForce"], label="netForce")
plt.plot(t, df["motorForce"], label="motorForce")
plt.plot(t, df["motorTorque"], label="motorTorque")
# plt.plot(t, df["motorRPM"], label="motorRPM")
plt.plot(t, df["speed"], label="speed")
plt.legend()
plt.show()


plt.plot(dfReal["Time_ms"]/1000, dfReal["SME_TRQSPD_Speed"], label="Real RPM")
plt.plot(t, df["motorRPM"], label="Simulated RPM")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("RPM")
plt.show()
df["speed"].max()