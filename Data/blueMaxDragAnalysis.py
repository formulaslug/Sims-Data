import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from FSLib.IntegralsAndDerivatives import *
from FSLib.fftTools import *
from FSLib.AnalysisFunctions import *
from scipy.optimize import curve_fit

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

dfa = readValid("../fs-data/FS-3/08102025/08102025RollingResistanceTestP1.parquet")
dfb = readValid("../fs-data/FS-3/08102025/08102025RollingResistanceTestP2.parquet")
dfc = readValid("../fs-data/FS-3/08102025/08102025RollingResistanceTestP3.parquet")
dfd = readValid("../fs-data/FS-3/08102025/08102025RollingResistanceTestP4.parquet")
dfa.insert_column(0, timeCol(dfa))
dfb.insert_column(0, timeCol(dfb))
dfc.insert_column(0, timeCol(dfc))
dfd.insert_column(0, timeCol(dfd))


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


def resistanceCurveFun (x, coeffRollingResistance, dragCoeff):
    carMass = 221.4# kg
    carNormalForce = 9.805*carMass # N
    airDensity = 1.23 # kg / m^3
    def drag(speed):
        return 0.5*airDensity*dragCoeff*(speed**2) #add frontal area
    
    outList = []

    dfs = []
    pos = 1
    while (True):
        try:
            ind = x["Time"].to_list().index(0, pos)
            dfs.append(x[pos-1:ind])
            pos = ind+1
            continue
        except:
            dfs.append(x[pos-1:])
            break

    for df in dfs:
        arr = np.zeros(df.height)
        time_s = df[t] - df[t].min() # s
        speed = (df[rpm]*12/41*0.2*2*np.pi/60)# m/s
        arr[0] = speed[0]
        for i in range(1, df.height):
            dt = time_s[i] - time_s[i-1]
            force = carNormalForce*coeffRollingResistance + drag(arr[i-1])
            accel = force/(carMass + 22.68)
            arr[i] = arr[i-1] - (dt * accel)
        outList.append(arr)
        #print(np.concatenate(outList))
    return np.concatenate(outList)

if dragTrainingDFs:
    # Use the concatenated dataframe
    times = dragTrainingdf[t].to_numpy()
    speeds = dragTrainingdf[rpm].to_numpy() * 12/41*0.2*2*np.pi/60

    popt, pcov = curve_fit(resistanceCurveFun, dragTrainingdf, speeds, p0=[0.01, 0.3])

    coeffRollingResistance, dragCoeff = popt
    print("Rolling Resistance Coefficient:", coeffRollingResistance)
    print("Drag Coefficient * Frontal Area:", dragCoeff)
else:
    print("No data loaded for drag training. Please populate dragTrainingDFs with dataframes.")
