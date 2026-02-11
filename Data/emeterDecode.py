from nptdms import TdmsFile
import polars as pl
import matplotlib.pyplot as plt
from Data.FSLib.IntegralsAndDerivatives import *
from Data.FSLib.AnalysisFunctions import *

autoxDaniel2File = "FS-3/compEmeterData/autoxDaniel2.tdms"
autoxDaniel11File = "FS-3/compEmeterData/autoxDaniel11.tdms"
autoxDaniel12File = "FS-3/compEmeterData/autoxDaniel12.tdms"
accel1 = "FS-3/compEmeterData/216_univ-of-calif---santa-cruz-_250620-203307_ ACCEL-EV.tdms"
accel2 = "FS-3/compEmeterData/216_univ-of-calif---santa-cruz-_250620-205609_ ACCEL-EV.tdms"
endur1 = "FS-3/compEmeterData/216_univ-of-calif---santa-cruz-_250621-154731_ ENDUR-EV.tdms"
endur2 = "FS-3/compEmeterData/216_univ-of-calif---santa-cruz-_250621-160530_ ENDUR-EV.tdms"

dfLaptimes = pl.read_csv("FS-3/compLapTimes.csv")
firstHalf = dfLaptimes.filter(pl.col("Lap") < 12)["Time"].sum()
secondHalf = dfLaptimes.filter(pl.col("Lap") > 11)["Time"].sum()

autoxD1_StartTime = 18
autoxD1_EndTime  = 34

autoxD2_StartTime = 13.6
autoxD2_EndTime = 63.17

endur1_StartTime = 28.1
endur1_EndTime = endur1_StartTime + firstHalf

endur2_StartTime = 60.24
endur2_EndTime = endur2_StartTime + secondHalf

def rms(df, col):
    return np.sqrt((df[col].pow(2) / df.height).sum())

def fileTodf(path):
    df = pl.DataFrame()
    file = TdmsFile.read(path)
    groups = file.groups()
    channels = groups[0].channels()
    df = pl.DataFrame()
    for channel in channels:
        colName = channel.name
        # print(channel.properties)
        ser = pl.Series(channel[:]).alias(colName)
        df.insert_column(0, ser)
    # df = df["Violation", "GLV", "Energy", "Current", "Voltage"].with_columns((pl.col(V).mul(pl.col(I))).alias("Power"))
    # print(df.columns)
    df = df["Violation", "GLV", "Energy", "Current", "Voltage"]
    df.insert_column(0, pl.Series(np.arange(df.height)/100).alias("Time"))
    # df = df.with_columns(pl.col(V) * pl.col(I))
    # print(f"Energy {in_place_integrate(df[P].to_numpy())[0,-1]}")
    return df


lv = "GLV"
v = "Violation"
V = "Voltage"
I = "Current"
E = "Energy"
P = "Power"
t = "Time"


dfautoxD2 = fileTodf(autoxDaniel2File).filter(pl.col(t) > autoxD2_StartTime).filter(pl.col(t) < autoxD2_EndTime)
dfautoxD1 = fileTodf(autoxDaniel11File)
dfautoxD1 = dfautoxD1.vstack(fileTodf(autoxDaniel12File).with_columns((pl.col(t) + dfautoxD1.height/100).alias(t))).filter(pl.col(t) > autoxD1_StartTime).filter(pl.col(t) < autoxD1_EndTime)
dfaccel1 = fileTodf(accel1)
dfaccel2 = fileTodf(accel2)
dfendur1 = fileTodf(endur1).filter(pl.col(t) > endur1_StartTime).filter(pl.col(t) < endur1_EndTime)

dfendur1 = dfendur1.with_columns(
    (pl.col(I) * pl.col(V)).alias("Power")
)

dfendur1 = dfendur1.with_columns(
    integrate_with_tCol(dfendur1[P], dfendur1[t]).alias(E) # type: ignore
)

t0 = endur1_StartTime
pos = 0
arr = np.zeros_like(dfendur1[t])
for i, time in enumerate(dfLaptimes.filter(pl.col("Lap") < 12)[t]):
    print(f"pos = {pos}")
    print(f"time = {time}")
    print(f"i = {i}")
    lap = i + 1
    height = dfendur1.filter(pl.col(t) >= t0).filter(pl.col(t) < t0 + time).height
    print(f"height = {height}")
    arr[pos:pos+height] = np.ones_like(arr[pos:pos+height],dtype=np.int64) * lap
    pos+=height
    t0+=time

dfendur1 = dfendur1.with_columns(
    pl.Series(arr).cast(pl.Int64).alias("Lap")
)



dfendur2 = fileTodf(endur2).filter(pl.col(t) > endur2_StartTime).filter(pl.col(t) < endur2_EndTime)

dfendur2 = dfendur2.with_columns(
    (pl.col(I) * pl.col(V)).alias("Power")
)

dfendur2 = dfendur2.with_columns(
    integrate_with_tCol(dfendur2[P], dfendur2[t]).alias(E) # type: ignore
)

t0 = endur2_StartTime
pos = 0
arr = np.zeros_like(dfendur2[t])
for i, time in enumerate(dfLaptimes.filter(pl.col("Lap") > 11)[t]):
    print(f"pos = {pos}")
    print(f"time = {time}")
    print(f"i = {i}")
    lap = i + 12
    height = dfendur2.filter(pl.col(t) >= t0).filter(pl.col(t) < t0 + time).height
    print(f"height = {height}")
    arr[pos:pos+height] = np.ones_like(arr[pos:pos+height],dtype=np.int64) * lap
    pos+=height
    t0+=time

dfendur2 = dfendur2.with_columns(
    pl.Series(arr).cast(pl.Int64).alias("Lap")
)

print(f"autoxD2 - RMS Current = {rms(dfautoxD2, I)} - Mean Current = {dfautoxD2[I].mean()}")
print(f"autoxD1 - RMS Current = {rms(dfautoxD1, I)} - Mean Current = {dfautoxD1[I].mean()}")
print(f"endur1 - RMS Current = {rms(dfendur1, I)} - Mean Current = {dfendur1[I].mean()}")
print(f"endur2 - RMS Current = {rms(dfendur2, I)} - Mean Current = {dfendur2[I].mean()}")

# df.filter(pl.col("Violation") == True)

# plt.plot(dfautoxD2[E])
# plt.plot(dfautoxD2[P])
# plt.show()

firstHalfLapTimes = laptimesNEnergy(dfendur1)
secondHalfLapTimes = laptimesNEnergy(dfendur2, bottom=11)
firstHalfLapTimes[E].sum() + secondHalfLapTimes[E].sum()

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
firstHalfLapTimes = laptimesNEnergy(dfendur1)
secondHalfLapTimes = laptimesNEnergy(dfendur2, bottom=11)
ax1.scatter(firstHalfLapTimes["LapTime"],firstHalfLapTimes[E], label = "Daniel")
ax1.scatter( secondHalfLapTimes["LapTime"], secondHalfLapTimes[E],label = "Brenner")
fig1.suptitle("Comp Energy vs. Time")
ax1.set_xlabel("Lap Time (s)")
ax1.set_ylabel("Energy Used (kWh)")
ax1.legend()
plt.show()

# dfendur1[t][dfendur1.height-1]-dfendur1[t][0]


fig = plt.figure(layout="constrained")
ax1 = fig.add_subplot(221)
ax11 = ax1.twinx()
ax2 = fig.add_subplot(222)
ax22 = ax2.twinx()
ax3 = fig.add_subplot(223)
ax33 = ax3.twinx()
ax4 = fig.add_subplot(224)
# ax4 = fig.add_subplot(111)
ax44 = ax4.twinx()

Vy_min = -5
Vy_max = 135
Iy_min = -20
Iy_max = 800

ax1.plot(dfautoxD1[t], dfautoxD1[V], label = V)
ax11.plot(dfautoxD1[t], dfautoxD1[I], color="orange", label = I)
ax1.set_title("autoxD1")
ax1.set_ylabel("Voltage (V)")
ax11.set_ylabel("Current (A)")
ax1.set_xlabel("Time (s)")
ax1.set_ylim(Vy_min, Vy_max)
ax11.set_ylim(Iy_min, Iy_max)

ax2.plot(dfautoxD2[t], dfautoxD2[V])
ax22.plot(dfautoxD2[t], dfautoxD2[I], color="orange")
ax2.set_title("autoxD2")
ax2.set_ylabel("Voltage (V)")
ax22.set_ylabel("Current (A)")
ax2.set_xlabel("Time (s)")
ax2.set_ylim(Vy_min, Vy_max)
ax22.set_ylim(Iy_min, Iy_max)

ax3.plot(dfendur1[t], dfendur1[V])
ax33.plot(dfendur1[t], dfendur1[I], color="orange")
ax3.set_title("endur pt 1 - Daniel")
ax3.set_ylabel("Voltage (V)")
ax33.set_ylabel("Current (A)")
ax3.set_xlabel("Time (s)")
ax3.set_ylim(Vy_min, Vy_max)
ax33.set_ylim(Iy_min, Iy_max)

segments = []
t0 = endur1_StartTime
for i in dfLaptimes.filter(pl.col("Lap") < 12)["Lap"]:
    segments.append((t0, t0 + dfLaptimes[t][i-1]))
    t0+=dfLaptimes[t][i-1]

# segments = [(dfendur1[t][0] + i * segmentSize, dfendur1[t][0] + (i + 1) * segmentSize) for i in range(numEndurSegments)]

arr = np.zeros_like()
for i,seg in enumerate(segments):
    color = "red"
    if i%2 == 0:
        color = "blue"
    ax3.axvspan(seg[0], seg[1], alpha=0.1, color=color)


ax4.plot(dfendur2[t], dfendur2[V])
# ax4.plot([dfendur2[t][0], dfendur2[t][dfendur2.height-1]], [30*2.65, 30*2.65], color="red")
ax44.plot(dfendur2[t], dfendur2[I], color="orange")
ax4.set_title("endur pt 2 - Brenner")
ax4.set_ylabel("Voltage (V)")
ax44.set_ylabel("Current (A)")
ax4.set_xlabel("Time (s)")
ax4.set_ylim(Vy_min, Vy_max)
ax44.set_ylim(Iy_min, Iy_max)

segments = []
t0 = endur2_StartTime
for i in dfLaptimes.filter(pl.col("Lap") > 11)["Lap"]:
    segments.append((t0, t0 + dfLaptimes[t][i-1]))
    t0+=dfLaptimes[t][i-1]

for i,seg in enumerate(segments):
    color = "red"
    if i%2 == 0:
        color = "blue"
    ax4.axvspan(seg[0], seg[1], alpha=0.1, color=color)

fig.legend()
plt.suptitle("Comp Data")
plt.show()



fig = plt.figure(layout="constrained")
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)


ax1.plot(dfendur1[t],dfendur1[P]/1000, label="Power")

segments = []
t0 = endur1_StartTime
for i in dfLaptimes.filter(pl.col("Lap") < 12)["Lap"]:
    segments.append((t0, t0 + dfLaptimes[t][i-1]))
    t0+=dfLaptimes[t][i-1]

# segments = [(dfendur1[t][0] + i * segmentSize, dfendur1[t][0] + (i + 1) * segmentSize) for i in range(numEndurSegments)]
for i,seg in enumerate(segments):
    color = "red"
    if i%2 == 0:
        color = "blue"
    ax1.axvspan(seg[0], seg[1], alpha=0.1, color=color)


ax2.plot(dfendur2[t], dfendur2[P]/1000, label="Power")

segments = []
t0 = endur2_StartTime
for i in dfLaptimes.filter(pl.col("Lap") < 12)["Lap"]:
    segments.append((t0, t0 + dfLaptimes[t][i-1]))
    t0+=dfLaptimes[t][i-1]

# segments = [(dfendur1[t][0] + i * segmentSize, dfendur1[t][0] + (i + 1) * segmentSize) for i in range(numEndurSegments)]
for i,seg in enumerate(segments):
    color = "red"
    if i%2 == 0:
        color = "blue"
    ax2.axvspan(seg[0], seg[1], alpha=0.1, color=color)

ax1.legend()
ax2.legend()
ax1.set_title("Endurance Part 1 - Daniel")
ax2.set_title("Endurance Part 2 - Brenner")
ax1.set_ylim(0,50)
ax2.set_ylim(0,50)
ax1.set_xlabel("Time(s)")
ax2.set_xlabel("Time(s)")
ax1.set_ylabel("Power(kWh)")
ax2.set_ylabel("Power(kWh)")
fig.suptitle("Comp Power Usage")
plt.show()
