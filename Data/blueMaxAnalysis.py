import polars as pl
import matplotlib.pyplot as plt
from Data.FSLib.IntegralsAndDerivatives import *
from Data.FSLib.fftTools import *
from Data.FSLib.AnalysisFunctions import *

dbcPath = "../fs-3/CANbus.dbc"

# lv = "GLV"
# v = "Violation"
V = "ACC_POWER_PACK_VOLTAGE"
I = "SME_TEMP_BusCurrent"
E = "Energy"
P = "Power"
t = "Time"

smeFaultCode = "SME_TEMP_FaultCode"
smeFaultLevel = "SME_TEMP_FaultLevel"
smeContactor = "SME_TRQSPD_contactor_closed"
busV = "SME_TEMP_DC_Bus_V"
busC = "SME_TEMP_BusCurrent"
bmsFault = "ACC_STATUS_BMS_FAULT"
imdFault = "ACC_STATUS_IMD_FAULT"
pchOn = "ACC_STATUS_PRECHARGING"
pchDone = "ACC_STATUS_PRECHARGE_DONE"
accShutdown = "ACC_STATUS_SHUTDOWN_STATE" 
glv = "ACC_STATUS_GLV_VOLTAGE"

vdmValid = "VDM_GPS_VALID1"
# time = ""
brakeF = "TMAIN_DATA_BRAKES_F"
brakeR = "TMAIN_DATA_BRAKES_R"
frT = "TELEM_FR_SUSTRAVEL"
flT = "TELEM_FL_SUSTRAVEL"
brT = "TELEM_BR_SUSTRAVEL"
blT = "TELEM_BL_SUSTRAVEL"
lat = "VDM_GPS_Latitude"
long = "VDM_GPS_Longitude"
course = "VDM_GPS_TRUE_COURSE"
xA = "xA"
yA = "yA"
zA = "zA"
vA = "vA"
xA_uncorrected = "VDM_X_AXIS_ACCELERATION"
yA_uncorrected = "VDM_Y_AXIS_ACCELERATION"
zA_uncorrected = "VDM_Z_AXIS_ACCELERATION"
vA_uncorrected = "vA_uncorrected"
xG = "VDM_X_AXIS_YAW_RATE"
yG = "VDM_Y_AXIS_YAW_RATE"
zG = "VDM_Z_AXIS_YAW_RATE"
rpm = "SME_TRQSPD_Speed"
speed = "VDM_GPS_SPEED"
tsC = "ACC_POWER_CURRENT"
xA_mps = "IMU_XAxis_Acceleration_mps"
yA_mps = "IMU_YAxis_Acceleration_mps"
zA_mps = "IMU_ZAxis_Acceleration_mps"
speed_mps = "VMD_GPS_Speed_mps"
index = "index"
heFL = "TPERIPH_FL_DATA_WHEELSPEED"
heFR = "TPERIPH_FR_DATA_WHEELSPEED"
heBL = "TPERIPH_BL_DATA_WHEELSPEED"
heBR = "TPERIPH_BR_DATA_WHEELSPEED"

blueMaxGPS_Square = ((-121.7330999, 38.5759097),(-121.7328352, 38.5757670)) ## Tuned! Generally make it bigger than you need probably beacuse GPS is infrequent.


dfautox = readValid("../fs-data/FS-3/08172025/08172025_28autox3&4_45C_40C_~29Cambient_0fans.parquet")
dfautox.insert_column(0, simpleTimeCol(dfautox))

dfautox1 = dfautox.filter(pl.col(t) > 139).filter(pl.col(t) < 230)
dfautox1 = dfautox1.with_columns(
    (pl.col(busC) * pl.col(busV)).alias(P)
)
dfautox1 = dfautox1.with_columns(
    integrate_with_tCol(dfautox1[P], dfautox1[t]).alias(E) #type: ignore
)

dfautox2 = dfautox.filter(pl.col(t) > 809).filter(pl.col(t) < 905)
dfautox2 = dfautox2.with_columns(
    (pl.col(busC) * pl.col(busV)).alias(P)
)
dfautox2 = dfautox2.with_columns(
    integrate_with_tCol(dfautox2[P], dfautox2[t]).alias(E) #type: ignore
)

dfendur1P1 = readValid("../fs-data/FS-3/08102025/08102025Endurance1_FirstHalf.parquet")
dfendur1P1.insert_column(0, timeCol(dfendur1P1))
dfendur1P1 = dfendur1P1.filter(pl.col(t) > 276.4).filter(pl.col(t) < 1087)
dfendur1P1.insert_column(1, lapSegmentation(dfendur1P1, blueMaxGPS_Square))
dfendur1P1 = dfendur1P1.with_columns(
    (pl.col(busC) * pl.col(busV)).alias(P)
)
dfendur1P1 = dfendur1P1.with_columns(
    integrate_with_tCol(dfendur1P1[P], dfendur1P1[t]).alias(E), #type: ignore
    dfendur1P1.select([f"ACC_SEG{i}_TEMPS_CELL{j}" for i in range(5) for j in range(6)]).max_horizontal().alias("MaxTemp")
)



dfendur1P2 = readValid("../fs-data/FS-3/08102025/08102025Endurance1_SecondHalf.parquet")
dfendur1P2.insert_column(0, timeCol(dfendur1P2))
dfendur1P2 = dfendur1P2.filter(pl.col(t) > 72).filter(pl.col(t) < 870.3)
dfendur1P2.insert_column(1, lapSegmentation(dfendur1P2, blueMaxGPS_Square))
dfendur1P2 = dfendur1P2.with_columns(
    (pl.col(busC) * pl.col(busV)).alias(P)
)
dfendur1P2 = dfendur1P2.with_columns(
    integrate_with_tCol(dfendur1P2[P], dfendur1P2[t]).alias(E), #type: ignore
    dfendur1P2.select([f"ACC_SEG{i}_TEMPS_CELL{j}" for i in range(5) for j in range(6)]).max_horizontal().alias("MaxTemp")
)

dfRegenTest = read("../fs-data/FS-3/11222025/11222025_6_RegenTest1.parquet")
dfRegenTest.insert_column(0, simpleTimeCol(dfRegenTest))

basicView(dfRegenTest)
plt.plot(dfRegenTest["SME_CURRLIM_ChargeCurrentLim"])
plt.show()

dfRegenTest2 = read("../fs-data/FS-3/11222025/11222025_12.parquet")
dfRegenTest2.insert_column(0, simpleTimeCol(dfRegenTest2))
basicView(dfRegenTest2)

plt.plot(dfRegenTest2[heBL], label=heBL)
plt.plot(dfRegenTest2[heFL], label=heFL)
plt.plot(dfRegenTest2[heBR], label=heBR)
plt.plot(dfRegenTest2[heFR], label=heFR)
plt.show()

plt.plot(dfRegenTest2["SME_CURRLIM_ChargeCurrentLim"])
plt.show()

fig1 = plt.figure(layout="constrained")
ax1 = fig1.add_subplot(221)
ax11 = ax1.twinx()
ax2 = fig1.add_subplot(222)
ax22 = ax2.twinx()
ax3 = fig1.add_subplot(223)
ax33 = ax3.twinx()
ax4 = fig1.add_subplot(224)
ax44 = ax4.twinx()

Vy_min = -5
Vy_max = 135
Iy_min = -20
Iy_max = 800

ax1.plot(dfautox1[t], dfautox1[V])
ax11.plot(dfautox1[t], dfautox1[I], color="orange")
ax1.set_title("autox1 08172025")
ax1.set_ylim(Vy_min, Vy_max)
ax11.set_ylim(Iy_min, Iy_max)
ax1.set_ylabel("Voltage/Temperature")
ax11.set_ylabel("Current")
ax1.set_xlabel("Time (Sec)")

ax2.plot(dfautox2[t], dfautox2[V])
ax22.plot(dfautox2[t], dfautox2[I], color="orange")
ax2.set_title("autox2 08172025")
ax2.set_ylim(Vy_min, Vy_max)
ax22.set_ylim(Iy_min, Iy_max)
ax2.set_ylabel("Voltage")
ax22.set_ylabel("Current")
ax2.set_xlabel("Time (Sec)")


ax3.plot(dfendur1P1[t], dfendur1P1[V], label="V")
ax3.plot(dfendur1P1[t], dfendur1P1["MaxTemp"], color="purple", label="maxTemp")
ax33.plot(dfendur1P1[t], dfendur1P1[I], color="orange",label = I)
ax3.set_title("endur1P1 08102025")
ax3.set_ylim(Vy_min, Vy_max)
ax33.set_ylim(Iy_min, Iy_max)
segments = createSegments(dfendur1P1)
for i,seg in enumerate(segments):
    color = "red"
    if i%2 == 0:
        color = "blue"
    ax3.axvspan(seg[0], seg[1], alpha=0.1, color=color)
ax3.set_ylabel("Voltage / Temp")
ax33.set_ylabel("Current")
ax3.set_xlabel("Time (Sec)")

ax4.plot(dfendur1P2[t], dfendur1P2[V])
# ax4.plot([dfendur1P2[t][0], dfendur1P2[t][dfendur1P2.height-1]], [30*2.65, 30*2.65], color="red")
ax4.plot(dfendur1P2[t], dfendur1P2["MaxTemp"], color="purple")
ax44.plot(dfendur1P2[t], dfendur1P2[I], color="orange")
ax4.set_title("endur1P2 08102025")
ax4.set_ylim(Vy_min, Vy_max)
ax44.set_ylim(Iy_min, Iy_max)
segments = createSegments(dfendur1P2)
for i,seg in enumerate(segments):
    color = "red"
    if i%2 == 0:
        color = "blue"
    ax4.axvspan(seg[0], seg[1], alpha=0.1, color=color)
ax4.set_ylabel("Voltage")
ax44.set_ylabel("Current")
ax4.set_xlabel("Time (Sec)")

fig1.legend()
fig1.suptitle("FS-3 Bluemax Data")
fig1.show()



fig2 = plt.figure(layout="constrained")
ax1 = fig2.add_subplot(221)
ax2 = fig2.add_subplot(222)
ax3 = fig2.add_subplot(223)
ax4 = fig2.add_subplot(224)

segments = createSegments(dfendur1P1)
for i,seg in enumerate(segments):
    color = "red"
    if i%2 == 0:
        color = "blue"
    ax1.axvspan(seg[0], seg[1], alpha=0.1, color=color)
    ax3.axvspan(seg[0], seg[1], alpha=0.1, color=color)

segments = createSegments(dfendur1P2)
for i,seg in enumerate(segments):
    color = "red"
    if i%2 == 0:
        color = "blue"
    ax2.axvspan(seg[0], seg[1], alpha=0.1, color=color)
    ax4.axvspan(seg[0], seg[1], alpha=0.1, color=color)

ax1.plot(dfendur1P1[t], dfendur1P1[zG], label = "Angular Velocity (Yaw)")
ax1.set_title("Bluemax Endurance P1 Angular Velocity (z axis)")
ax1.set_xlabel("Time")
ax1.set_ylabel("Angular Velocity (deg/sec)")
ax2.plot(dfendur1P2[t], dfendur1P2[zG])
ax2.set_title("Bluemax Endurance P2 Angular Velocity (z axis)")
ax2.set_xlabel("Time")
ax2.set_ylabel("Angular Velocity (deg/sec)")
ax3.plot(dfendur1P1[t], in_place_derive(dfendur1P1[zG]), color="orange", label = "Angular Acceleration (Yaw)")
ax3.set_title("Bluemax Endurance P1 Angular Acceleration (z axis)")
ax3.set_xlabel("Time")
ax3.set_ylabel("Angular Acceleration (deg/sec^2)")
ax4.plot(dfendur1P2[t], in_place_derive(dfendur1P2[zG]), color="orange")
ax4.set_title("Bluemax Endurance P2 Angular Acceleration (z axis)")
ax4.set_xlabel("Time")
ax4.set_ylabel("Angular Acceleration (deg/sec^2)")

fig2.legend()
fig2.suptitle("FS-3 Bluemax Data - Yaw Rate/Acceleration")
fig2.show()

## GGV
bigdf = dfendur1P1.vstack(dfendur1P2)

plt.scatter(bigdf[yA_uncorrected], bigdf[xA_uncorrected], c=bigdf[speed], cmap='viridis', alpha = 0.2, s=0.3)
plt.xlabel("Lateral Acceleration (Gs)")
plt.ylabel("Longitudinal Acceleration (Gs)")
plt.colorbar()
plt.suptitle("GGV")
plt.xlim((-2,2))
plt.ylim((-2,2))
plt.show()

## Speed Estimation Checking. Might illustrate slip.

plt.plot(bigdf[rpm]*11/40*0.2*2*np.pi/60/0.44704)
plt.plot(bigdf[speed])
plt.legend(["rpm convesion","speed vdm"])
plt.show()

## Determine Values for Mechanical Resistance for the car
dfa = readValid("../fs-data/FS-3/08102025/08102025RollingResistanceTestP1.parquet")
dfb = readValid("../fs-data/FS-3/08102025/08102025RollingResistanceTestP2.parquet")
dfc = readValid("../fs-data/FS-3/08102025/08102025RollingResistanceTestP3.parquet")
dfd = readValid("../fs-data/FS-3/08102025/08102025RollingResistanceTestP4.parquet")
dfa.insert_column(0, timeCol(dfa))
dfb.insert_column(0, timeCol(dfb))
dfc.insert_column(0, timeCol(dfc))
dfd.insert_column(0, timeCol(dfd))

## Manually Determined ranges of segments for conducting mechanical resistance calculation

dragTrainingDFs = [
    dfa.filter(pl.col(t) > 117).filter(pl.col(t) < 149),
    dfa.filter(pl.col(t) > 177).filter(pl.col(t) < 197),
    dfb.filter(pl.col(t) > 25).filter(pl.col(t) < 41),
    dfb.filter(pl.col(t) > 54).filter(pl.col(t) < 72),
    dfb.filter(pl.col(t) > 99.5).filter(pl.col(t) < 114.2),
    dfc.filter(pl.col(t) > 9.985).filter(pl.col(t) < 17),
    dfc.filter(pl.col(t) > 112).filter(pl.col(t) < 133)
]

dragTrainingdf = pl.concat([dfa.with_columns(pl.col(t) - pl.col(t).min()) for dfa in dragTrainingDFs])

# def resistanceCurveFun (x, coeffRollingResistance, dragCoeff):
#     carMass = 221.4# kg
#     carNormalForce = 9.805*carMass # N
#     airDensity = 1.23 # kg / m^3
#     def drag(speed):
#         return 0.5*airDensity*dragCoeff*(speed**2)
    
#     outList = []

#     dfs = []
#     pos = 1
#     while (True):
#         try:
#             ind = x["Time"].to_list().index(0, pos)
#             dfs.append(x[pos-1:ind])
#             pos = ind+1
#             continue
#         except:
#             dfs.append(x[pos-1:])
#             break

#     for df in dfs:
#         arr = np.zeros(df.height)
#         time = df[t] - df[t].min() # s
#         speed = df[rpm]*11/40*0.2*2*np.pi/60 # m/s
#         arr[0] = speed[0]
#         for i in range(1, df.height):
#             dt = time[i] - time[i-1]
#             force = carNormalForce*coeffRollingResistance + drag(arr[i-1])
#             accel = force/(carMass + 22.68)
#             arr[i] = arr[i-1] - (dt * accel)
#         outList.append(speed.to_numpy() - arr)
#     print(np.concatenate(outList))
#     return np.concatenate(outList)
#     # return outList
        

# args = curve_fit(resistanceCurveFun, dragTrainingdf, np.zeros(sum([df.height for df in dragTrainingDFs])), p0=[0.1, 0.1])

# args[0]

# arr = np.array([CellInterpolatorVTC5A(5.75, q)] for q in np.arange())

# basicView(read("FS-3/08102025/08102025FirstTurnOn.parquet"), tFun=simpleTimeCol)


# plt.plot(dfendur1P1[t], low_pass_filter(dfendur1P1["TMAIN_DATA_BRAKES_F"]*10, 0.95), label = "Main FB")
# plt.plot(dfendur1P1[t], (dfendur1P1["ETC_STATUS_BRAKE_SENSE_VOLTAGE"]/1000), label="ETC Brake")
# plt.legend()
# plt.show()


# df = dfautox1
# curr = df[busC]
# charge = integrate_with_Scipy_tCol(df[busC], df[t])
# window = np.zeros(1678*2 + 1)
# window[1678:] = np.ones_like(window[1678:])*60/5035
# chargeLast20Sec = np.convolve(curr, window, "same")


# plt.plot(charge, label="Charge")
# plt.plot(curr, label="Current")
# plt.plot(chargeLast20Sec, label="Charge in last 20 sec")
# plt.legend()
# plt.show()


## Laptime vs Energy Graph

# dfLapEnergy = pl.concat([laptimesNEnergy(dfendur1P1), laptimesNEnergy(dfendur1P2)])
# dfLapEnergy

# plt.scatter(dfLapEnergy["LapTime"], dfLapEnergy[E])
# plt.xlabel("BlueMax LapTime (s)")
# plt.ylabel("Energy (kWh)")
# plt.show()

# laptimesNEnergy(dfendur1P1)
# laptimesNEnergy(dfendur1P2)

# dfa = readValid("FS-3/08172025/08172025_27autox2&45C_35C_~28Cambient_100fans.parquet")
# dfa.insert_column(0, timeCol(dfa))

# Comparison of the different time methods
# plt.plot(dfa[t])
# plt.plot(np.arange(0, dfa.height * 60/5035, 60/5035))
# plt.plot((dfa["VDM_UTC_TIME_SECONDS"] - dfa["VDM_UTC_TIME_SECONDS"].min()).cast(pl.Int32)*60)
# plt.legend(["timeCol Estimation", "Raw 60/5035", "VDM Minutes * 60"])
# plt.xlabel("timePoint")
# plt.ylabel("Seconds")
# plt.show()

# Used to determine the 60/5035 thingy
# dfautox1.filter(pl.col("VDM_UTC_TIME_SECONDS") > 41).filter(pl.col("VDM_UTC_TIME_SECONDS") < 47).height / 5

# basicView(readValid("FS-3/08172025/08172025_27autox2&45C_35C_~28Cambient_100fans.parquet"))
# basicView(readValid("FS-3/08172025/08172025_22_6LapsAndWeirdCurrData.parquet"))
# basicView(readValid("FS-3/08172025/08172025_28autox3&4_45C_40C_~29Cambient_0fans.parquet"), tFun=simpleTimeCol)
# basicView(readValid("FS-3/08172025/08172025_26autox1.parquet"))
# basicView(readValid("FS-3/08172025/08172025_20_Endurance1P1.parquet"))

## Stuff to debug weird MC current logging
# FS-3/08172025/08172025_22_6LapsAndWeirdCurrData.parquet
# FS-3/08172025/08172025_26autox1.parquet

# dfa = readValid("FS-3/08102025/08102025Endurance1_FirstHalf.parquet")
# dfa = readValid("FS-3/08172025/08172025_22_6LapsAndWeirdCurrData.parquet")
# dfa.insert_column(0,timeCol(dfa))
# dfa = dfa.filter(pl.col("Time")>260)
# dfa.insert_column(1, lapSegmentation(dfa, blueMaxGPS_Square))

# plt.plot(dfa[t],dfa["SME_TEMP_DC_Bus_V"], label="BUS V")
# plt.plot(dfa[t],dfa["SME_TEMP_FaultCode"], label="Fault Code")
# plt.plot(dfa[t],dfa["SME_TEMP_ControllerTemperature"], label = "Controller Temp")
# plt.plot(dfa[t],dfa["SME_THROTL_MBB_Alive"], label = "mbb alive")
# plt.plot(dfa[t],dfa["SME_TRQSPD_Speed"], label="rpm")
# plt.plot(dfa[t],dfa["SME_TEMP_BusCurrent"], label="current")
# plt.plot(dfa[t],dfa["ACC_STATUS_PRECHARGE_DONE"]*1000, label="prechargeDone")
# plt.legend()
# plt.show()



## GPS Lap View

# printLaptimes(dfa)
# dfaGPSFiltered = dfa.filter(pl.col(lat) != 0).filter(pl.col(long) != 0)

# plt.plot(dfa[t], lapSegmentation(dfa, blueMaxGPS_Square))
# plt.xlabel("Time (s)")
# plt.ylabel("Lap")
# plt.show()

# laps = dfa["Lap"].max()
# colors = [(i/laps, 1 - i/laps, i/laps) for i in range(laps)]
# for lap in range(laps):
#     dfLap = dfa.filter(pl.col("Lap") == lap + 1)
#     plt.plot(dfLap[long], dfLap[lat], c=colors[lap])
# plt.scatter(dfa[long], dfa[lat], c=dfa["Lap"], s=1)
# plt.axis("scaled")
# plt.show()

# df = readCorrectedFSDAQ("FS-3/08172025raw/08172025fsdaq/08172025_20.fsdaq", dbcPath)
# basicView(df.filter(pl.col("VDM_GPS_VALID1") == 1))


# def voltageLookup

# Fit a few cubic functions to charge used in the last 20 sec, current draw, # of discharges, and SOC
# def voltagePredictionFunction(x, a1, a2, a3, b1, b2, b3, c1, c2, c3):
#     # a1-3 are linear, quad, and cubic of current draw
#     # b1-3 are ... for charge used in last 20 sec
#     # c1-3 are ... for number of discharges

#     # Get Current draw
#     # Calculate Charge Used in last 20 sec
#     # Get number of discharges
#     # Get a voltage from linear interpolation
#     df = x
#     curr = df[busC]
#     # charge = integrate_with_Scipy_tCol(df[busC], df[t])
#     # 20 sec range is ~1678
#     # Window gets flipped so bits at the end face backward and are now dependent on past charge usage.
#     window = np.zeros(1678*2 + 1) # ~20 sec to either side of the middle.
#     window[1678:] = np.ones_like(window[1678:])*60/5035 # Effectively performing an integral of the ~20 sec before your current location. Off the beginning is just 0 which works.
#     chargeLast20Sec = np.convolve(curr, window, "same") # In coulombs. Ah is /3600. mAh is /3.6
    
#     ## TODO: The above should be reimplemented to work based on a time range, rather than a set of indexes because that varies over time/throught. Repeated integrals may be time consuming.
#     # May be a good idea to make a column which is ((t[n+1] - t[n]) * I[n]) and do an integral of that over the right time range. Would save some computation.

#     ## TODO: Train on data where # of discharges is well documented to record the effect of that for this model. For now it will be my guess of about 10.

#     discharges = 10

#     NearestNDInterpolator

#     return