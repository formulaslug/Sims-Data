import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from heatGraph import colored_line 

df = pl.read_parquet("FS-3/08102025Endurance1_FirstHalf.parquet").vstack(
    pl.read_parquet("FS-3/08102025Endurance1_SecondHalf.parquet"))

df = pl.read_parquet("FS-3/08102025Debug1.parquet")
for i in range(2,25):
    df = df.vstack(pl.read_parquet(f"FS-3/08102025Debug{i}.parquet"))

[i if i[0] == "A" else "" for i in df.columns]

df.columns

smeFaultCode = "SME_TEMP_FaultCode"
smeFaultLevel = "SME_TEMP_FaultLevel"
smeContactor = "SME_TRQSPD_contactor_closed"
busV = "SME_TEMP_Bus_V"
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

df = df.filter(pl.col(vdmValid) == 1)

fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
colored_line(df[lat], df[long], df[speed], ax1, linewidth=1, cmap="plasma")

plt.plot(df[brakeF])
plt.plot(df[brakeR])

# df.filter(pl.col("ACC_SEG0_VOLTS_CELL0") != 0.0)["ACC_SEG0_VOLTS_CELL0"].min()
fig = plt.figure()
ax = fig.add_subplot(111)
ax2 = ax.twinx()
ax.plot((df["ACC_SEG0_VOLTS_CELL0"]/100 + 2)*30)
ax2.plot(df[tsC]*0.01, color='orange')
plt.show()


for i in range(5):
    for j in range(6):
        plt.plot(df[f"ACC_SEG{i}_VOLTS_CELL{j}"]/100 + 2)

for i in range(5):
    for j in range(6):
        plt.plot(df[f"ACC_SEG{i}_TEMPS_CELL{j}"])

plt.plot(df["ACC_SEG0_VOLTS_CELL0"]/100 + 2)
plt.plot(df["ACC_SEG0_VOLTS_CELL1"]/100 + 2)
plt.plot(df["ACC_SEG0_VOLTS_CELL2"]/100 + 2)
plt.plot(df["ACC_SEG0_VOLTS_CELL3"]/100 + 2)
plt.plot(df["ACC_SEG0_VOLTS_CELL4"]/100 + 2)
plt.plot(df["ACC_SEG0_VOLTS_CELL5"]/100 + 2)

plt.plot(df[xA_uncorrected])
plt.plot(df[yA_uncorrected])
plt.plot(df[zA_uncorrected])
plt.plot(df[speed])

plt.plot(df.filter(pl.col(glv) != 0)[glv]/10)
plt.plot(df[bmsFault])
plt.plot(df[imdFault])
plt.plot(df[pchOn])
plt.plot(df[pchDone])
plt.plot(df[accShutdown])
plt.plot(df[glv]/10)
plt.plot(df[smeContactor])
plt.plot(df[smeFaultCode])
plt.plot(df[smeFaultLevel])
plt.legend(["bmsFault", "imdFault", "pchOn", "pchDone", "accShutdown", "glv", "smeContactor", "smeFaultCode", "smeFaultLevel"])

plt.show()