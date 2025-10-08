import polars as pl
import matplotlib.pyplot as plt
import numpy as np


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

def readValid (a):
    return pl.read_parquet(a).filter(pl.col("VDM_GPS_VALID1") == 1)

def read (a):
    return pl.read_parquet(a)

def timeCol(df, verbose = False):
    min = "VDM_UTC_TIME_SECONDS"
    timeArr = np.zeros(df.height)
    lastMinuteUpdate = df[min][0]
    lastUpdate = 0
    if verbose: print(f"df height = {df.height} - firstMinute = {lastMinuteUpdate}")
    initialChunk = df.filter(pl.col(min) == lastMinuteUpdate)
    if verbose: print(f"initial chunk size = {initialChunk.height}")
    if verbose: print(f"setting timeArr[0:{initialChunk.height}] to {np.arange(start=0,stop=initialChunk.height/100,step=0.01)}")
    timeArr[:initialChunk.height] = np.arange(start=0,stop=initialChunk.height*60/5035,step=60/5035)
    lastUpdate = initialChunk.height
    lastMinuteUpdate = initialChunk[min][lastUpdate-1] + 1
    pos = lastUpdate
    if verbose: print(f"lastUpdate = {lastUpdate} - lastMinuteUpdate = {lastMinuteUpdate} - pos = {pos}")
    if verbose: print(f"Border looks like [{df[min][pos-2]}, {df[min][pos-1]}, {df[min][pos]}, {df[min][pos+1]}]")
    while pos < df.height:
        if df[min][pos] != lastMinuteUpdate:
            if verbose: print(f"pos = {pos} - minute = {df[min][pos]}")
            counter = pos - lastUpdate
            stepSize = 60 / counter
            newTimeChunk = np.arange(stepSize,60+stepSize,stepSize) + timeArr[lastUpdate-1]
            timeArr[lastUpdate:lastUpdate+counter] = newTimeChunk[0:counter]
            lastUpdate+=counter
            lastMinuteUpdate+=1
        pos+=1
    counter = pos - lastUpdate
    stepSize = 60*(counter/5035) / counter
    newTimeChunk = np.arange(stepSize,60*(counter/5035)+stepSize,stepSize) + timeArr[lastUpdate-1]
    timeArr[lastUpdate:lastUpdate+counter] = newTimeChunk[0:len(timeArr[lastUpdate:lastUpdate+counter])]
    return pl.Series(timeArr).alias("Time")

def simpleTimeCol (df, verbose=False):
    stepSize = 60/5035
    return pl.Series(np.arange(0,df.height*stepSize, stepSize)).alias("Time")

def basicView (df, title="", tFun=timeCol, scatterGPS=False, scaled=False, cellVoltages=False, verbose=False):
    '''
    Loads a basic view of a given run. Built for FS-3 Data generated and collected by the team.

    Parameters
    ----------
    df
        The Dataframe to base the time graph on. Should have valid GPS data or the graphs will be blank.
    title 
        Title at the top of the graph.
    tFun 
        Time function to be used to generate a time column
    scatterGPS
        Whether to scatter plot the GPS instead of line plot
    scaled
        Whether to scale the GPS plot to have the same vertical and horizontal scale. Useful for small plots
    cellVoltages
        Whether to plot the cell voltages on the plot with ACC and MC voltages
    verbose
        Whether to print debug messages while generating the graph
    '''
    if not ("Time" in df.columns):
        df.insert_column(0, tFun(df, verbose))

    fig = plt.figure(layout="constrained")
    ax1 = fig.add_subplot(321)
    ax2 = fig.add_subplot(322)
    ax3 = fig.add_subplot(323)
    ax4 = fig.add_subplot(324)
    ax5 = fig.add_subplot(325)
    ax6 = fig.add_subplot(326)
    for i in range(5):
        for j in range(6):
            ax1.plot(df[t],df[f"ACC_SEG{i}_VOLTS_CELL{j}"])
    ax1.set_title("Acc Seg Voltages")
    ax2.plot(df[t],df[V], color="cyan", label="Total As Reported by Acc")
    b = sum([sum([df[f"ACC_SEG{i}_VOLTS_CELL{j}"] for j in range(6)]) for i in range (5)])
    ax2.plot(df[t],b, color="pink", label="Sum of Cells")
    ax2.plot(df[t], df[busV], color="purple", label="MC Voltage")
    ax22 = ax2.twinx()
    if cellVoltages:
        ax22 = ax2.twinx()
        for i in range(5):
            a = [df[f"ACC_SEG{i}_VOLTS_CELL{j}"] for j in range(6)]
            ax22.plot(sum(a), label=f"ACC_SEG{i}")
        ax22.legend()
        ax22.set_ylabel("Voltage (V)")
    ax2.set_title("Voltages")
    ax2.legend()
    # ax3.plot(df[t],df[tsC], label="Accumulator Current")
    ax3.plot(df[t],df[busC], label = "Motor Controller Current")
    ax3.legend()
    dfGPSFiltered = df.filter(pl.col(lat) != 0).filter(pl.col(long) != 0)
    if scatterGPS:
        ax4.scatter(dfGPSFiltered[long],dfGPSFiltered[lat], s=0.5)
    else:
        ax4.plot(dfGPSFiltered[long],dfGPSFiltered[lat])
    if scaled:
        ax4.axis("scaled")
    ax5.plot(df[t], (df["ETC_STATUS_BRAKE_SENSE_VOLTAGE"]-330)*25/33, label="Braking (psi)")
    ax5.plot(df[t], df["SME_THROTL_TorqueDemand"]/32767*180, label="Torque Demand (N)", color="orange")
    ax5.plot(df[t], df[speed], color = "goldenrod", label="speed mph")
    ax5.plot(df[t], df[rpm]*11/40*0.2032*2*np.pi/60/0.44704, color = "green", label="rpm speed")
    ax5.plot(df[t], df["VDM_Y_AXIS_ACCELERATION"]*100, label = "yaxis accel (cGs)", color="crimson")
    ax5.set_title("Speed + Braking")
    ax6.plot(df[t],df[xA_uncorrected])
    ax6.set_title("Acceleration (X)")

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Voltage (V)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Voltage (V)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Current (A)")
    ax4.set_xlabel("Longitude (deg)")
    ax4.set_ylabel("Latitude (deg)")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Voltage (V)")
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("Acceleration (Gs)")

    plt.suptitle(title)
    plt.show()

def speedGraph(df, tFun=timeCol):
    if not ("Time" in df.columns):
        df.insert_column(0, tFun(df, False))
    fig3 = plt.figure()
    ax1 = fig3.add_subplot(111)
    ax1.plot(df[t], (df["ETC_STATUS_BRAKE_SENSE_VOLTAGE"]-330)*25/33, label="Braking (psi)")
    ax1.plot(df[t], df["SME_THROTL_TorqueDemand"]/32767*180, label="Torque Demand (N)", color="orange")
    ax1.plot(df[t], df[speed], color = "goldenrod", label="speed mph")
    ax1.plot(df[t], df[rpm]*11/40*0.2032*2*np.pi/60/0.44704, color = "green", label="rpm speed")
    ax1.plot(df[t], df["VDM_Y_AXIS_ACCELERATION"]*100, label = "yaxis accel (cGs)", color="crimson")
    ax1.legend()
    fig3.show()

def lapSegmentation(df, square, verbose=False, superVerbose=False):
    lat = "VDM_GPS_Latitude"
    long = "VDM_GPS_Longitude"
    longs = [square[0][0], square[1][0]]
    lats = [square[0][1], square[1][1]]
    longMin = min(longs)
    longMax = max(longs)
    latMin = min(lats)
    latMax = max(lats)
    if verbose:
        print(f"longMin = {longMin}; longMax = {longMax}")
        print(f"latMin = {latMin}; latMax = {latMax}")
        print(f"long0 = {df[long][0]}; lat0 = {df[lat][0]}")
    def inBox(longitude, latitude):
        if superVerbose:
            print(f"long = {longitude}; lat = {latitude}")
        if ((latitude > latMin) and (latitude < latMax) and (longitude > longMin) and (longitude < longMax)):
            if superVerbose:
                print("inBox!")
            return True
        return False
    arr = np.zeros(df.height, dtype=np.int16)
    lap = 1
    lastTimeInBox = df["Time"][0] + 10.0

    ## Increment Lap when you enter the GPS box as long as:
        # It's been more than 5 sec since you last incremented the lap
        # Your RPM > 100
        # You weren't in the box in the previous time step
    for i in range(df.height):
        latitude = df[lat][i]
        longitude = df[long][i]
        inTheBox = inBox(longitude, latitude)
        
        doNotIncrementConditions = (df["SME_TRQSPD_Speed"][i] < 100) or ((df["Time"][i] - lastTimeInBox) < 5)
        if inTheBox and not doNotIncrementConditions:
            lap+=1
        
        arr[i] = lap

        if inTheBox:
            lastTimeInBox = df["Time"][i]
    return pl.Series(arr).alias("Lap")
        
def laptimesNEnergy(df, verbose=False, emeter=False, bottom=0):
    outT = []
    outE = []
    for i in range(bottom,df["Lap"].max()):
        lap = df.filter(pl.col("Lap") == (i+1))
        t0 = lap[t][0]
        t1 = lap[t][lap.height-1]
        e0 = lap[E][0]
        e1 = lap[E][lap.height-1]
        if verbose:
            print(f"Length of lap {i+1} is {round(t1-t0, 3)} Sec. Energy is {round((e1-e0)/3.6e6, 5)} kWh")
        outT.append(t1-t0)
        outE.append((e1-e0)/3.6e6) # Conversion of J to kWh
    return pl.DataFrame([pl.Series(outT).alias("LapTime"), pl.Series(outE).alias(E)])

def createSegments(df):
    segmentList = []
    for i in range(1,df["Lap"].max()+1,1):
        dfLap = df.filter(pl.col("Lap") == i)
        if i == df["Lap"].max():
            segmentList.append((dfLap[t][0], dfLap[t][dfLap.height-1]))
        else:
            dfLapUp1 = df.filter(pl.col("Lap") == (i+1))
            segmentList.append((dfLap[t][0], dfLapUp1[t][0]))
    return segmentList