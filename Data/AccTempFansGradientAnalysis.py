import polars as pl
import matplotlib.pyplot as plt
import cantools.database as db

# from DataDecoding_N_CorrectionScripts.dataDecodingFunctions import *
from Data.FSLib.AnalysisFunctions import *
from Data.FSLib.IntegralsAndDerivatives import *
from Data.FSLib.fftTools import *

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
tm = "TempMean"

df100 = readValid("../fs-data/FS-3/08172025/08172025_27autox2&45C_35C_~28Cambient_100fans.parquet")
df0 = readValid("../fs-data/FS-3/08172025/08172025_28autox3&4_45C_40C_~29Cambient_0fans.parquet")

segs = np.array([[i for i in df100.columns if i.startswith(f"ACC_SEG{j}_TEMPS")] for j in range(5)]).flatten()

# df100.select(segs).cast(pl.Float32).mean_horizontal().alias("TempMean")

df100 = df100.with_columns(
    df100.select(segs).cast(pl.Float32).mean_horizontal().alias("TempMean"),
    simpleTimeCol(df100)
)

df0 = df0.with_columns(
    df0.select(segs).cast(pl.Float32).mean_horizontal().alias("TempMean"),
    simpleTimeCol(df0)
)

standardStepSize = 60/5035

df100F = df100[tm][6788:]
df0F = df0[tm][69330+370+6788:]

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.plot(df100F[:20000], label = "100%")
ax1.plot(df0F[:20000], label = "0%")

avger = 200

ax2.plot(low_pass_filter(in_place_derive(np.convolve(df100F.to_numpy(), [1/avger for _ in range(avger)], "valid"), standardStepSize)[:20000], 0.5), label = "∆T 100%")
ax2.plot(low_pass_filter(in_place_derive(np.convolve(df0F.to_numpy(), [1/avger for _ in range(avger)], "valid"), standardStepSize)[:20000], 0.5), label = "∆T 0%")

ax1.legend()
ax2.legend()

fig.show()

plt.plot(df100F.rolling_mean(21))
plt.show()