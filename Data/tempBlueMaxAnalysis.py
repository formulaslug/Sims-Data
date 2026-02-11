import polars as pl
import matplotlib.pyplot as plt
from Data.integralsAndDerivatives import *
from Data.fftTools import *
from Data.AnalysisFunctions import *

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
file5 = "../fs-data/FS-3/11222025/11222025_23.parquet"

df = read(file1).vstack(read(file2)).vstack(read(file3)).vstack(read(file4)).vstack(read(file5))
df.insert_column(0, simpleTimeCol(df))


basicView(df, cellVoltages=False, tempsInsteadOfVoltages=True)