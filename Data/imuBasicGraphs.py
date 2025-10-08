## This file was created to generate IMU data but was too simple to function properly.
## Does not work in its current state - Nathaniel 1/11/25

from matplotlib.axes import Axes
from matplotlib.text import Text
from matplotlib.transforms import Bbox
import polars as pl
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

df = pl.read_parquet("./Parquet/2024-12-02-Part1-100Hz.pq")
# df = pl.read_parquet("./Parquet/2024-12-02-Part2-100Hz.pq")

time1 = 100
# time1 = 1530.4
# time1 = 1000
time2 = 9999999999
# time2 = 1710
# time2 = 1900
lat = "VDM_GPS_Latitude"
long = "VDM_GPS_Longitude"

df = pl.DataFrame(df.filter(pl.col("Seconds") >= time1).filter(pl.col("Seconds") <= time2))[::100]

print(df.columns)

fig, ax = plt.subplots(1,1)
fig.set_label("Gyroscope (deg/s)")

ax.plot(df.select("Seconds"), df.select("VDM_Z_AXIS_YAW_RATE"), label="Z Axis")
ax.legend(loc="best")

fig.tight_layout()
plt.show()


