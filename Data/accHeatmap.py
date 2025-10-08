## Code for generating an accumulator heatmap from data from the car. Created by Jack Nystrom.

from matplotlib.axes import Axes
import matplotlib.patheffects as path_effects
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from matplotlib.text import Text
import polars as pl
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

# df = pl.read_parquet("./Parquet/2024-12-02-Part1-100Hz.pq")
# df = pl.read_parquet("./Parquet/2024-12-02-Part2-100Hz.pq")
df = pl.read_parquet("FS-2/Parquet/2024-12-02-Part1-100Hz.pq").with_columns(pl.all().cast(pl.Float32, strict=False))
# df = pl.read_parquet("FS-3/08102025Endurance1_FirstHalf.parquet").vstack(
    # pl.read_parquet("FS-3/08102025Endurance1_SecondHalf.parquet"))


time1 = 1314
time1 = 1300
# time1 = 100
# time1 = 1000
# time2 = 9999999999
time2 = 2296
time2 = 2350
# time2 = 1900
lat = "VDM_GPS_Latitude"
long = "VDM_GPS_Longitude"
sec = "Seconds"
speed = "SME_TRQSPD_Speed"
tempLabels = [f"Seg{i}_TEMP_{j}" for i in range(0, 4) for j in range(0, 7)]
voltageLabels = [f"Seg{i}_VOLT_{j}" for i in range(0, 4) for j in range(0, 7)]


# tempLabels = [f"Seg{i}_TEMP_{j}" for i in range(0, 5) for j in range(0, 6)]
# voltageLabels = [f"Seg{i}_VOLT_{j}" for i in range(0, 5) for j in range(0, 6)]

# Filter to the specified time range
short = pl.DataFrame(df.filter(pl.col(sec) >= time1).filter(pl.col(sec) <= time2))

# replace incorrect 0 rows with None, then interpolate using prev/next rows (maintains correct size)
colsWithIncorrectZeros = tempLabels + voltageLabels + [lat, long]
short = short.with_columns([
    pl.when((pl.col(col) > -0.5) & (pl.col(col) < 0.5))
      .then(None)  # Replace invalid values with None
      .otherwise(pl.col(col))
      .alias(col)
    for col in colsWithIncorrectZeros # this for loop is only to generate the correct list of transformations, not actually iterating the data! That's cool
]).select([
    pl.col(col).interpolate().alias(col)
    for col in short.columns
]) # Interpolates all None (null) values

animRate = 10 # 50 good for larger chunks per frame.
# shortAcc = short.filter(pl.all_horizontal(pl.col(tempLabels) != 0))[::animRate]
# shortGPS = short.filter(pl.all_horizontal(pl.col(tempLabels) != 0)).filter(pl.col(lat) != 0).filter(pl.col(long) != 0)

tempCols = short[::animRate].select(tempLabels)
voltCols = short[::animRate].select(voltageLabels)
secondCol = short[::animRate].select(sec).to_series()

# These df's are the 4x7 square dataframes placed onto the grids/heatmaps
tempDfs: list[pl.DataFrame] = []
for row in tempCols.iter_rows():
    ndarr = np.reshape(row, (4, 7))
    tempDfs.append(pl.DataFrame(ndarr))
voltDfs: list[pl.DataFrame] = []
for row in voltCols.iter_rows():
    ndarr = np.reshape(row, (4, 7))
    voltDfs.append(pl.DataFrame(ndarr))

gs = GridSpec(2, 2)
gs.update(wspace=-0.2, hspace=0.4)
fig = plt.figure()
figTitle = fig.suptitle(f"Acc Heat/Voltage Map, GPS  -  Time: ?s", ha="center", va="top")
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[:, 1])

accTempImg = ax1.imshow(tempDfs[0], cmap="Spectral_r")
ax1.set_xticks(range(0, 7), labels=[f"TEMP{i}" for i in range(0, 7)])
ax1.set_yticks(range(0, 4), labels=[f"Seg{i}" for i in range(0, 4)])
title1 = ax1.set_title(f"Acc Temperature: ?s")
cbar1 = fig.colorbar(accTempImg, ax=ax1, location="right", fraction=0.046, pad=0.04)
accTempImg.set_clim(0, 70)
cbar1.minorticks_on()
cbar1.set_label("Degrees °C")

accVoltImg = ax2.imshow(voltDfs[0], cmap="OrRd")
ax2.set_xticks(range(0, 7), labels=[f"VOLT{i}" for i in range(0, 7)])
ax2.set_yticks(range(0, 4), labels=[f"Seg{i}" for i in range(0, 4)])
title2 = ax2.set_title(f"Acc Voltage: ?s")
cbar2 = fig.colorbar(accVoltImg, ax=ax2, location="right", fraction=0.046, pad=0.04)
accVoltImg.set_clim(2.5, 4) # Volt range
cbar2.minorticks_on()
cbar2.set_label("Volts")

# plt.axis("off")
texts1: list[list[Text]] = []
texts2: list[list[Text]] = []
for i in range(4):
    t = []
    for j in range(7):
        t.append(ax1.text(j, i, tempDfs[0][i, j], ha="center", va="center"))
    texts1.append(t)
for i in range(4):
    t = []
    for j in range(7):
        t.append(ax2.text(j, i, voltDfs[0][i, j], ha="center", va="center"))
    texts2.append(t)

gpsLine, = ax3.plot(short[lat],-1*short[long], linewidth=10, color="gray")
# gpsLine.set_data([], [])
# for art in list(ax3.lines): art.remove()
lc = LineCollection([], antialiaseds=True, path_effects=[path_effects.Stroke(capstyle="round")])
lc.set_cmap("Greens") # RdPu RdYlBu
# gpsLine, = ax3.plot([], [])
ax3.add_collection(lc)
ax3.axis("scaled")
ax3.set_xlabel("Latitude")
ax3.set_ylabel("Longitude")
ax3.tick_params(
    left=False, right=False, labelleft=False, labelbottom=False, bottom=False
)
ax3.set_title("GPS Location")


def animate(n):
    figTitle.set_text(f"Accumulator Heat/Voltage Map, GPS  ({secondCol[n]:.1f}s)")

    tempDf = tempDfs[n]
    accTempImg.set_data(tempDf)
    for i in range(4):
        for j in range(7):
            texts1[i][j].set_text(f"{tempDf[i, j]:.1f}°C")
    title1.set_text(f"Acc Temperature")
    
    voltDf = voltDfs[n]
    accVoltImg.set_data(voltDf)
    for i in range(4):
        for j in range(7):
            texts2[i][j].set_text(f"{voltDf[i, j]:.1f}V")
    title2.set_text(f"Acc Voltage")
    
    numGPSPoints = 1000
    lhs = max(0, n*animRate - numGPSPoints)

    lats = short[lat][lhs:n*animRate]
    longs = -1*short[long][lhs:n*animRate]

    widths = np.linspace(0, 6, len(lats))
    colors = np.linspace(0, 1, len(lats))
    points = np.array([lats, longs]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc.set_linewidth(list(widths))
    lc.set_array(list(colors))
    lc.set_segments(list(segments))

    return [accTempImg, gpsLine, title1, title2, figTitle] + [q for t in texts1 for q in t] + [q for t in texts2 for q in t]

# ugh, blit=False sucks (so much slower), but it's required for animated title :/
anim = animation.FuncAnimation(fig, animate, frames=len(short[::animRate]), interval=1, blit=False, repeat=False)
# fig.tight_layout()
plt.show()

# ---------------

fig, axs = plt.subplots(2,2, squeeze=False)
axs = axs.ravel()
for i in range(0, 4):
    axx: Axes = axs[i]
    axx.set_title(f"Seg{i}")
    axx.set_xlabel(sec)
    axx.set_ylabel("Degrees °C")
    for j in range(0, 7):
        axx.plot(secondCol, tempCols.select(f"Seg{i}_TEMP_{j}"), label=f"Seg{i}_TEMP_{j}")
    axx.legend(loc="best")

fig.tight_layout()
# plt.show()

# ---------------

fig, axs = plt.subplots(2,2, squeeze=False)
axs = axs.ravel()
for i in range(0, 4):
    axx: Axes = axs[i]
    axx.set_title(f"Seg{i}")
    axx.set_xlabel(sec)
    axx.set_ylabel("Volts (?)")
    for j in range(0, 7):
        axx.plot(secondCol, voltCols.select(f"Seg{i}_VOLT_{j}"), label=f"Seg{i}_VOLT_{j}")
    axx.legend(loc="best")

fig.tight_layout()
plt.show()
