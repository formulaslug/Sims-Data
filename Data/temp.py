import polars as pl
import matplotlib.pyplot as plt
import cantools.database as db

from DataDecoding_N_CorrectionScripts.dataDecodingFunctions import *
from AnalysisFunctions import *

dbcPath = "../fs-3/CANbus.dbc"
dbc = db.load_file(dbcPath)
df1 = readCorrectedFSDAQ("E:/fsdaq/1.fsdaq", dbcPath)
df2 = readCorrectedFSDAQ("E:/fsdaq/2.fsdaq", dbcPath)
df3 = readCorrectedFSDAQ("E:/fsdaq/3.fsdaq", dbcPath)
df1 = applyDBC_ScaleAndOffset(dbc, readFSDAQ("E:/fsdaq/1.fsdaq"))
df2 = applyDBC_ScaleAndOffset(dbc, readFSDAQ("E:/fsdaq/2.fsdaq"))
df3 = applyDBC_ScaleAndOffset(dbc, readFSDAQ("E:/fsdaq/3.fsdaq"))
df4 = applyDBC_ScaleAndOffset(dbc, readFSDAQ("E:/fsdaq/4.fsdaq"))
df5 = applyDBC_ScaleAndOffset(dbc, readFSDAQ("E:/fsdaq/5.fsdaq"))
df6 = applyDBC_ScaleAndOffset(dbc, readFSDAQ("E:/fsdaq/6.fsdaq"))

df1 = pl.DataFrame(df1)
df2 = pl.DataFrame(df2)
df3 = pl.DataFrame(df3)
df1 = df1.insert_column(0, simpleTimeCol(df1))
df2 = df2.insert_column(0, simpleTimeCol(df2))
df3 = df3.insert_column(0, simpleTimeCol(df3))
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

df = readFSDAQ("E:/fsdaq/2.fsdaq")
df[speed]

fig1 = plt.figure()
ax1 = fig1.add_subplot(211)
ax2 = fig1.add_subplot(212)
ax1.plot(df1[t], df1[busC])
ax2.plot(df2[t], df2[busC])
fig1.show()

fig2 = plt.figure()
ax1 = fig2.add_subplot(111)
ax1.plot(df3[t], df3[busC])
fig2.show()

basicView(df3)

[i for i in df6.columns if i.startswith("SME")]

df = readValid("FS-3/08102025/08102025Debug22.parquet")

plt.plot(df6["ETC_STATUS_BRAKE_SENSE_VOLTAGE"])
plt.plot(df6["ETC_STATUS_BRAKELIGHT"]*50000)
plt.plot(df6["SME_THROTL_MBB_Alive"])
plt.plot(df4["SME_THROTL_MBB_Alive"])
plt.plot(df5["SME_THROTL_MBB_Alive"])
plt.plot(df3["SME_THROTL_MBB_Alive"])
plt.plot(df["SME_THROTL_MBB_Alive"])
plt.show()
