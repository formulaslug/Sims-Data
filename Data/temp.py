import polars as pl
import matplotlib.pyplot as plt
import cantools.database as db

from Data.DataDecoding_N_CorrectionScripts.dataDecodingFunctions import *
from Data.FSLib.AnalysisFunctions import *
from Data.FSLib.IntegralsAndDerivatives import *
from scipy.interpolate import CubicSpline

dbcPath = "../fs-3/CANbus.dbc"
dbc = db.load_file(dbcPath)

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
frT = "TPERIPH_FR_DATA_SUSTRAVEL"
flT = "TPERIPH_FL_DATA_SUSTRAVEL"
brT = "TPERIPH_BR_DATA_SUSTRAVEL"
blT = "TPERIPH_BL_DATA_SUSTRAVEL"
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

pedalTravel = "ETC_STATUS_PEDAL_TRAVEL"
etcImplausibility = "ETC_STATUS_IMPLAUSIBILITY"
etcRTDButton = "ETC_STATUS_RTD_BUTTON"
etcBrakeVoltage = "ETC_STATUS_BRAKE_SENSE_VOLTAGE"

df = read("C:/Projects/FormulaSlug/fs-data/FS-3/10112025/firstDriveMCError30.parquet")
df = df.with_columns(
    df["timestamp"].alias("Time")
)

df = read("C:/Projects/FormulaSlug/fs-data/FS-3/10112025/firstDriveMCError30-filled-null.parquet")
df = df.with_columns(
    simpleTimeCol(df)
)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(df[t], df[frT], label=frT, c="blue")
ax.plot(df[t], df[flT], label=flT, c="red")
ax.plot(df[t], df[brT], label=brT, c="orange")
ax.plot(df[t], df[blT], label=blT, c="cyan")
ax.set_title("Suspension Travel during First Drive with MC Fault")
ax.set_xlabel("Time")
ax.set_ylabel("Suspension Travel (mm)")
ax.legend()
plt.show()


dfNullless = df.drop_nulls(subset=[frT, flT, brT, blT])

cs = CubicSpline(dfNullless[t], dfNullless[frT])

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(dfNullless[t], cs(dfNullless[t]), label=frT, s=0.5)
ax.scatter(dfNullless[t], in_place_derive(cs(dfNullless[t])), label=f"Derived {frT}", s=0.5)
ax.legend()
plt.show()


df = pl.read_parquet("C:/Projects/FormulaSlug/fs-data/FS-3/10112025/firstDriveMCError30-filled-null.parquet")

df.select([a for a in df.columns if a.startswith("ACC") and a[9] == "T"]).mean_horizontal()


fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(df[t], df[frT], label=frT, c="blue")
ax.set_title("Suspension Travel during First Drive with MC Fault")
ax.set_xlabel("Time")
ax.set_ylabel("Suspension Travel (mm)")
ax.legend()
plt.show()


heFR = "TELEM_FR_WHEELSPEED"
heFL = "TELEM_FL_WHEELSPEED"
heBR = "TELEM_BR_WHEELSPEED"
heBL = "TELEM_BL_WHEELSPEED"

df = read("C:/Projects/FormulaSlug/fs-data/FS-2/Parquet/2025-03-06-BrakingTests1.parquet")
df = df.with_columns(
    simpleTimeCol(df)
)


fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(df[t], df[heFR], label=heFR, c="orange")
ax.plot(df[t], df[heFL], label=heFL, c="red")
ax.plot(df[t], df[heBR], label=heBR, c="blue")
ax.plot(df[t], df[heBL], label=heBL, c="cyan")
ax.set_title("HE Sensors")
ax.set_xlabel("Time")
ax.set_ylabel("idk lol, 0-32767")
ax.legend()
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)
ax1 = ax.twinx()

ax.plot(df[t], df[heFR], label="frontRight", c="red")
ax.plot(df[t], df.select([heBL, heBR]).mean_horizontal(), label="meanBack", c="blue")
ax1.plot(df[t],df.select([heBL, heBR]).mean_horizontal() / df[heFR], label="ratio", c="green")
# ax.plot(df[t], df[rpm]*11/41, label="RPM", c="orange") ## Cursed until we have proper wheel speed calibration
ax.set_title("HE Sensors")
ax.set_xlabel("Time")
ax.set_ylabel("idk lol, 0-32767")
ax.legend()
ax1.set_ylim(0,2)
plt.show()