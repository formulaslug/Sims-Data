## This file contains the code originally used to create a plot using gps coordinates to create
## a heat map. Currently has updated functioning graphs for rpm, torque, and braking. 2025-01-20 (last update by Nathaniel)
## Made and commented by Nathaniel Platt
import polars as pl
import matplotlib.pyplot as plt
from Data.FSLib.heatGraph import colored_line
from Data.FSLib.fftTools import *

# df = pl.read_parquet("FS-2/Parquet/2025-03-06-BrakingTests1.parquet")
df = pl.read_parquet("FS-2/Parquet/2025-03-06-BrakingTests1.parquet").vstack(pl.read_parquet("FS-2/Parquet/2025-03-06-Part2.parquet"))

df
# df = pl.read_csv("Temp/2024-12-02-Part1-100Hz.csv",infer_schema_length=0).with_columns(pl.all().cast(pl.Float32, strict=False))
# df1 = pl.read_csv("Temp/2024-12-02-Part2-100Hz.csv",infer_schema_length=0).with_columns(pl.all().cast(pl.Float32, strict=False))
df1 = pl.read_parquet("FS-2/Parquet/2024-12-02-Part2-100Hz.pq")


df.columns

time1 = 1400
# time1 = 1000
time2 = 1650
# time2 = 1900
lat = "VDM_GPS_Latitude"
long = "VDM_GPS_Longitude"
speed = "SME_TRQSPD_Speed"
busCurrent = "SME_TEMP_BusCurrent"
tsCurrent = "TS_Current"
torque = "SME_THROTL_TorqueDemand"
brakes = "Brakes"
df.columns
df[lat]
# short = pl.DataFrame(df.filter(pl.col("Seconds") >= time1).filter(pl.col("Seconds") <= time2)).filter(pl.col("VDM_GPS_Latitude") != 0).filter(pl.col("VDM_GPS_Longitude") != 0)
short = df.filter(pl.col("VDM_GPS_Latitude") != 0).filter(pl.col("VDM_GPS_Longitude") != 0)


# df.drop_nulls().select(lat).mean()

# df.select(lat).filter(pl.col("VDM_GPS_Latitude") != 0)
# df.filter(pl.col("Seconds") == 498.199).select([lat,long])\
# fig = plt.figure()
# fig.add_subplot(1,1,1)
# ax = plt.figure().add_subplot(1,1,1)
# ax.pcolorfast(-1*short[long],-1*short[lat],a)
# ax.plot(-1*short[long],-1*short[lat])
# ax.axis('scaled')
# plt.show()
# df.columns
short
import warnings

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import LineCollection



fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
lines = colored_line(short[lat], short[long], short[busCurrent], ax1, linewidth=1, cmap="plasma")
fig1.colorbar(lines)  # add a color legend
ax1.axis('scaled')
ax1.set_title("Bus Current (A)")
plt.show()

# Create a figure and plot the line on it
fig1 = plt.figure()
ax1 = fig1.add_subplot(1,3,1)
lines = colored_line(short[lat], short[long], short[speed]/7500*109, ax1, linewidth=1, cmap="plasma")
fig1.colorbar(lines)  # add a color legend
ax1.axis('scaled')
ax1.set_title("Speed (mph)")

ax1 = fig1.add_subplot(1,3,2)
lines = colored_line(short[lat], short[long], short[torque]/30000*180, ax1, linewidth=1, cmap="viridis")
fig1.colorbar(lines)  # add a color legend
ax1.axis('scaled')
ax1.set_title("Torque (Nm)")

ax1 = fig1.add_subplot(1,3,3)
lines = colored_line(short[lat], short[long], (short[brakes]-0.1)*2000, ax1, linewidth=1, cmap="inferno")
fig1.colorbar(lines)  # add a color legend
ax1.axis('scaled')
ax1.set_title("Braking (psi)")

plt.show()

fig1 = plt.figure()
ax1 = fig1.add_subplot(1,3,1)
lines = colored_line(short[lat], short[long], short[busCurrent], ax1, linewidth=1, cmap="plasma")
fig1.colorbar(lines)  # add a color legend
ax1.axis('scaled')
ax1.set_title("Motor Controller Current (A)")

ax1 = fig1.add_subplot(1,3,2)
lines = colored_line(short[lat], short[long], short[tsCurrent], ax1, linewidth=1, cmap="viridis")
fig1.colorbar(lines)  # add a color legend
ax1.axis('scaled')
ax1.set_title("Accumulator Current (A)")

ax1 = fig1.add_subplot(1,3,3)
lines = colored_line(short[lat], short[long], (short[brakes]-0.1)*2000, ax1, linewidth=1, cmap="inferno")
fig1.colorbar(lines)  # add a color legend
ax1.axis('scaled')
ax1.set_title("Braking (psi)")

plt.show()

# Filtered and unfiltered brake data with low pass filter 
# front, then rear
plt.plot((df["TELEM_STEERBRAKE_BRAKEF"]/65535*5 - 0.5)*500)
plt.plot(low_pass_filter(((df["TELEM_STEERBRAKE_BRAKEF"]/65535*5 - 0.5)*500).to_numpy(), 0.9))
plt.legend(["Raw","Filtered"])
plt.xlabel("Time (s/100)")
plt.ylabel("Braking (psi)")
plt.title("Front Brake")
plt.show()

plt.plot((df["TELEM_STEERBRAKE_BRAKER"]/65535*5 - 0.5)*500)
plt.plot(low_pass_filter(((df["TELEM_STEERBRAKE_BRAKER"]/65535*5 - 0.5)*500).to_numpy(), 0.9))
plt.legend(["Raw","Filtered"])
plt.xlabel("Time (s/100)")
plt.ylabel("Braking (psi)")
plt.title("Rear Brake")
plt.show()


## Trying different filter amounts
plt.plot(low_pass_filter(((df["TELEM_STEERBRAKE_BRAKEF"]/65535*5 - 0.5)*500).to_numpy(), 0))
plt.plot(low_pass_filter(((df["TELEM_STEERBRAKE_BRAKEF"]/65535*5 - 0.5)*500).to_numpy(), 0.25))
plt.plot(low_pass_filter(((df["TELEM_STEERBRAKE_BRAKEF"]/65535*5 - 0.5)*500).to_numpy(), 0.7))
plt.plot(low_pass_filter(((df["TELEM_STEERBRAKE_BRAKEF"]/65535*5 - 0.5)*500).to_numpy(), 0.9))
plt.legend(["Raw", "0.25", "0.7", "0.9"])
plt.xlabel("Time (s/100)")
plt.ylabel("Braking (psi)")
plt.title("Low pass filters comparison")
plt.show()
df.columns

## Front and Rear together
dfIn = pl.read_parquet("FS-2/Parquet/2025-03-06-BrakingTests1.parquet")
df = dfIn[24700:26000]
df = dfIn
fig = plt.figure()
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)
ax1.plot(low_pass_filter(((df["TELEM_STEERBRAKE_BRAKEF"]/65535*5 - 0.5)*500).to_numpy(), 0.9))
ax1.plot(low_pass_filter(((df["TELEM_STEERBRAKE_BRAKER"]/65535*5 - 0.5)*500).to_numpy(), 0.9))
ax2.plot(df["VDM_GPS_SPEED"].rolling_mean(100))
ax3.plot(df["VDM_X_AXIS_ACCELERATION"]*-1)
ax1.legend(["Front Brake PSI","Rear Brake PSI"])
ax2.legend(["GPS Speed (m/s)"])
ax3.legend([ "X Axis Acceleration (g)"])
ax3.set_xlabel("Time (s/100)")
ax1.set_ylabel("Braking (psi)")
ax2.set_ylabel("Speed (m/s)")
ax3.set_ylabel("Acceleration (g)")
#set figure title
fig.suptitle("Blue Max 3/6/2025")
plt.show()

