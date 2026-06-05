import polars as pl
import matplotlib.pyplot as plt
from Data.FSLib.IntegralsAndDerivatives import *
from Data.FSLib.fftTools import *
from Data.FSLib.AnalysisFunctions import *

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

file1 = "../fs-data/FS-3/11222025/11222025_18.parquet"
file2 = "../fs-data/FS-3/11222025/11222025_19.parquet"
file3 = "../fs-data/FS-3/11222025/11222025_20.parquet"
file4 = "../fs-data/FS-3/11222025/11222025_21.parquet"
file5 = "../fs-data/FS-3/11222025/11222025_22.parquet"
file6 = "../fs-data/FS-3/11222025/11222025_23.parquet"

df = read(file1).vstack(read(file2)).vstack(read(file3)).vstack(read(file4)).vstack(read(file5)).vstack(read(file6))
df.insert_column(0, simpleTimeCol(df))


basicView(df.filter(pl.col(t) > 1200).filter(pl.col(t) < 1350), cellVoltages=False, tempsInsteadOfVoltages=False, faults=True)
mcErrorView(df)

dfNoRegen = read(file1).vstack(read(file2)).vstack(read(file3)).vstack(read(file4))
dfNoRegen = dfNoRegen.insert_column(0, simpleTimeCol(dfNoRegen))
dfWeakRegen = read(file5)
dfWeakRegen = dfWeakRegen.insert_column(0, simpleTimeCol(dfWeakRegen))
dfStrongRegen = read(file6)
dfStrongRegen = dfStrongRegen.insert_column(0, simpleTimeCol(dfStrongRegen))

fig = plt.figure(layout="constrained")
ax = fig.add_subplot(111)
ax.scatter(dfNoRegen[pedalTravel], dfNoRegen[torqueDemand], s=1, label="No Regen")
ax.scatter(dfWeakRegen[pedalTravel], dfWeakRegen[torqueDemand], s=1, label="Weak Regen")
ax.scatter(dfStrongRegen[pedalTravel], dfStrongRegen[torqueDemand], s=1, label="Strong Regen")
ax.legend()
ax.set_xlabel("Pedal Travel (%)")
ax.set_ylabel("Torque Demand (Nm)")
plt.suptitle("Regen Effect on Torque Demand vs Pedal Travel")
plt.show()

fig = plt.figure(layout="constrained")
ax = fig.add_subplot(111)
ax.scatter(dfWeakRegen[pedalTravel], dfWeakRegen[busC], s=1, label="Weak Regen", alpha=0.2)
ax.scatter(dfStrongRegen[pedalTravel], dfStrongRegen[busC], s=1, label="Strong Regen", alpha=0.2)
ax.legend()
ax.set_xlabel("Pedal Travel (%)")
ax.set_ylabel("Bus Current (A)")
plt.suptitle("Regen Effect on Bus Current vs Pedal Travel")
plt.show()

file7 = "../fs-data/FS-3/11222025/11222025_11.parquet"
file8 = "../fs-data/FS-3/11222025/11222025_12.parquet"
file9 = "../fs-data/FS-3/11222025/11222025_13.parquet"
file10 = "../fs-data/FS-3/11222025/11222025_14.parquet"

dfEarlyRegen = read(file7).vstack(read(file8)).vstack(read(file9)).vstack(read(file10))
dfEarlyRegen = dfEarlyRegen.insert_column(0, simpleTimeCol(dfEarlyRegen))

basicView(dfEarlyRegen, title="Early Regen Testing", cellVoltages=False, tempsInsteadOfVoltages=True, faults=True)

fig = plt.figure(layout="constrained")
ax = fig.add_subplot(111)
ax.scatter(dfEarlyRegen[pedalTravel], dfEarlyRegen[busC], s=1, label="Early Regen", alpha=0.2)
ax.legend()
ax.set_xlabel("Pedal Travel (%)")
ax.set_ylabel("Bus Current (A)")
plt.suptitle("Regen Effect on Bus Current vs Pedal Travel")
plt.show()

dfWeakNeg = dfWeakRegen.filter(pl.col(busC) < 0)
dfWeakPos = dfWeakRegen.filter(pl.col(busC) > 0)

plt.plot(dfWeakNeg[t], in_place_integrate(dfWeakNeg[busC]*dfWeakNeg[busV], dt=60/5035))
plt.plot(dfWeakPos[t], in_place_integrate(dfWeakPos[busC]*dfWeakPos[busV], dt=60/5035))
plt.show()

np.min(in_place_integrate(dfWeakNeg[busC]*dfWeakNeg[busV], dt=60/5035))/np.max(in_place_integrate(dfWeakPos[busC]*dfWeakPos[busV], dt=60/5035))*-1

fig = plt.figure(layout="constrained")
ax = fig.add_subplot(111)
# ax.plot(df[t], df["ETC_STATUS_HE1"]/3300, label="HE1")
# ax.plot(df[t], df["ETC_STATUS_HE2"]/3300, label="HE2")

# ax.plot(df[t], df["ETC_STATUS_HE1"]/3300 - df["ETC_STATUS_HE2"]/3300, label="diffHE")
ax.plot(df[t], df["ACC_STATUS_PRECHARGE_DONE"].cast(pl.Int32)*60, label="PCH Done")
# ax.plot(df[t], df[rpm])
ax.plot(df[t], df[busV], label="Bus V")
ax.plot(df[t], df["ACC_STATUS_SHUTDOWN_STATE"].cast(pl.Int32)*50, label="Shutdown State")
ax.plot(df[t], df["ACC_STATUS_GLV_VOLTAGE"]/1000, label="GLV V")
ax.plot(df[t], df["ETC_STATUS_RTD"].cast(pl.Int32)*40, label="RTD")

ax.set_xlabel("Time (s)")
# ax.set_ylabel("Hall Effect Travel")
# plt.suptitle("Hall Effect Sensor Travel during BlueMax Testing")
ax.legend()
plt.show()

## Graphs to make

# 1. Wheel speed vs traction control
# 2. Regen Bit swapping a lot + Regen Mapping
# 3. Regen Power usage
# 4. Potential MC error due to ACC Board not reading shutdown properly
# 5. Tray Temp Sensors
# 6. ACC Weird Temp data
# 7. ACC Cell Temp Distribution
# 8. ACC Cell Voltage Distribution

dfa = df.filter(pl.col(t) < 825).filter(pl.col(t) > 815)

fig = plt.figure(layout="constrained")
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.plot(dfa[t], dfa[heFL], label="FL Wheel Speed")
ax1.plot(dfa[t], dfa[heFR], label="FR Wheel Speed")
ax1.plot(dfa[t], dfa[heBL], label="BL Wheel Speed")
ax1.plot(dfa[t], dfa[heBR], label="BR Wheel Speed")

ax1.set_ylabel("Wheel Speed (rpm)")

ax2.plot(dfa[t], dfa[pedalTravel]/dfa[torqueDemand], label="Traction Control Ratio")
ax2.set_xlabel("Time (s)")
ax1.legend()
fig.show()

fig = plt.figure(layout="constrained")
ax = fig.add_subplot(111)
ax.plot(df[t], df[busC]*df[busV], label="Regen Power")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Power (W)")
plt.suptitle("Regen Power during BlueMax Testing")
fig.show()

fig = plt.figure(layout="constrained")
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.plot(dfWeakRegen[t], in_place_integrate(dfWeakRegen[busC]*dfWeakRegen[busV], dt=60/5035)/3600000, label="Weak Regen Energy")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Energy (kWh)")
ax2.plot(dfStrongRegen[t], in_place_integrate(dfStrongRegen[busC]*dfStrongRegen[busV], dt=60/5035)/3600000, label="Strong Regen Energy")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Energy (kWh)")
plt.suptitle("Regen Energy during BlueMax Testing")
fig.show()