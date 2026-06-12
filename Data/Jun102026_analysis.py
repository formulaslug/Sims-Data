from Data.FSLib.AnalysisFunctions import read
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

def basicViewFS4 (df:pl.DataFrame, title:str="", scatterGPS=False, verbose=False):
    '''
    Loads a basic view of a given run. Built for FS-4 Data generated and collected by the team.

    Parameters
    ----------
    df:pl.DataFrame
        The Dataframe to base the time graph on. Should have valid GPS data or the graphs will be blank.
    title:str
        Title at the top of the graph.
    scatterGPS
        Whether to scatter plot the GPS instead of line plot
    verbose
        Whether to print debug messages while generating the graph
    '''

    motorRPM = "SME_TRQSPD_Speed"
    currLimC = "SME_CURRLIM_ChargeCurrentLim"
    currLimD = "SME_CURRLIM_DischargeCurrentLim"
    torqueDemand = "SME_THROTL_TorqueDemand"

    packVoltage = "BATT_POWER_PACK_VOLTAGE"
    packCurrent = "BATT_POWER_CURRENT"
    smeVoltage = "SME_TEMP_DC_Bus_V"
    smeCurrent = "SME_TEMP_BusCurrent"

    frontBrakePressure = "VCU_BRAKE_PRESSSURE_FRONT"
    rearBrakePressure = "VCU_BRAKE_PRESSSURE_REAR"

    wheelSpeedFL = "TPERIPH_FL_DATA_WHEELSPEED"
    wheelSpeedFR = "TPERIPH_FR_DATA_WHEELSPEED"
    wheelSpeedBL = "TPERIPH_BL_DATA_WHEELSPEED"
    wheelSpeedBR = "TPERIPH_BR_DATA_WHEELSPEED"

    VX = "VCU_VN_BODY_VX"
    VY = "VCU_VN_BODY_VY"
    VZ = "VCU_VN_BODY_VZ"

    df = df.with_columns(
        pl.Series("V", np.linalg.norm(df[VX, VY, VZ].to_numpy())) # type:ignore
    )

    lat = "VCU_VN_LAT"
    long = "VCU_VN_LON"

    volts = [f"BATT_MOD{i}_VOLTS_CELL{j}" for i in range(5) for j in range(6)]
    temps = [f"BATT_MOD{i}_TEMPS_CELL{j}" for i in range(5) for j in range(6)]

    t = "Time"

    ETC_implausabilities = [
        "VCU_ETC_IMPLAUS_APPS_OUT_OF_RANGE",
        "VCU_ETC_IMPLAUS_BPPS_OUT_OF_RANGE",
        "VCU_ETC_IMPLAUS_APPS_DEVIATION",
        "VCU_ETC_IMPLAUS_BSE_OUT_OF_RANGE",
        "VCU_ETC_IMPLAUS_BRAKE_AND_ACCEL"
    ]

    df = df.with_columns(
        pl.Series(np.convolve(df["BATT_STATUS_BMS_FAULT"].to_numpy(), [1, -1], 'same')).alias("BMS_FaultStart"),
        pl.Series(np.convolve(df["BATT_STATUS_IMD_FAULT"].to_numpy(), [1, -1], 'same')).alias("IMD_FaultStart"),
        pl.Series(np.convolve(df[ETC_implausabilities].max_horizontal(), [1, -1], 'same')).alias("ETC_ImplausibilityStart"),
        pl.Series(np.convolve(df["VCU_BSPD_FAULT"], [1, -1], 'same')).alias("BSPD_FaultStart")
        )

    if not (t in df.columns):
        df.insert_column(0, (df["Time_ms"]/1000).alias(t))

    fig1 = plt.figure(layout="constrained")
    ax1 = fig1.add_subplot(321)
    ax2 = fig1.add_subplot(322)
    ax3 = fig1.add_subplot(323)
    ax4 = fig1.add_subplot(324)
    ax5 = fig1.add_subplot(325)
    ax6 = fig1.add_subplot(326)
    for temp in temps:
        ax1.plot(df[t],df[temp])
    ax1.set_title(f"Tractive Battery Temperatures")
    ax1.vlines(df.filter(pl.col("BMS_FaultStart") == 1)[t].to_numpy(), ymin=ax1.get_ylim()[0], ymax=ax1.get_ylim()[1], colors="red", label="BMS Fault")
    ax1.vlines(df.filter(pl.col("IMD_FaultStart") == 1)[t].to_numpy(), ymin=ax1.get_ylim()[0], ymax=ax1.get_ylim()[1], colors="orange", label="IMD Fault")
    ax1.vlines(df.filter(pl.col("ETC_ImplausibilityStart") == 1)[t].to_numpy(), ymin=ax1.get_ylim()[0], ymax=ax1.get_ylim()[1], colors="blue", label="ETC Implausibility")
    ax1.vlines(df.filter(pl.col("BSPD_FaultStart") == 1)[t].to_numpy(), ymin=ax1.get_ylim()[0], ymax=ax1.get_ylim()[1], colors="green", label="BSPD Fault")
    ax1.legend()

    for volt in volts:
        ax2.plot(df[t],df[volt])
    ax2.set_title(f"Tractive Battery Voltages")
    ax2.vlines(df.filter(pl.col("BMS_FaultStart") == 1)[t].to_numpy(), ymin=ax2.get_ylim()[0], ymax=ax2.get_ylim()[1], colors="red", label="BMS Fault")
    ax2.vlines(df.filter(pl.col("IMD_FaultStart") == 1)[t].to_numpy(), ymin=ax2.get_ylim()[0], ymax=ax2.get_ylim()[1], colors="orange", label="IMD Fault")
    ax2.vlines(df.filter(pl.col("ETC_ImplausibilityStart") == 1)[t].to_numpy(), ymin=ax2.get_ylim()[0], ymax=ax2.get_ylim()[1], colors="blue", label="ETC Implausibility")
    ax2.vlines(df.filter(pl.col("BSPD_FaultStart") == 1)[t].to_numpy(), ymin=ax1.get_ylim()[0], ymax=ax1.get_ylim()[1], colors="green", label="BSPD Fault")
    ax2.legend()

    ax3.plot(df[t],df[packVoltage], label="Battery Voltage")
    ax3.plot(df[t], df[smeVoltage], label="MC Voltage")
    ax3.set_ylabel("Voltage (V)")

    ax33 = ax3.twinx()
    ax33.plot(df[t],df[packCurrent], color="goldenrod", label = "Battery Current")
    ax33.plot(df[t],df[smeCurrent], color="blue", label = "MC Current")
    ax3.legend(patches=[patches.Patch(color="blue", label="MC Voltage"), patches.Patch(color="orange", label="Battery Voltage"), patches.Patch(color="goldenrod", label="Battery Current"), patches.Patch(color="blue", label="MC Current")], loc="upper left")

    dfGPSFiltered = df.filter(pl.col(lat) != 0).filter(pl.col(long) != 0)
    if scatterGPS:
        ax4.scatter(dfGPSFiltered[long],dfGPSFiltered[lat], s=0.5)
    else:
        ax4.plot(dfGPSFiltered[long],dfGPSFiltered[lat])
    ax4.axis("scaled")

    ax5.plot(df[t], df[frontBrakePressure], label="Front Braking (psi)")
    ax5.plot(df[t], df[rearBrakePressure], label="Rear Braking (psi)")
    ax5.plot(df[t], df["SME_THROTL_TorqueDemand"]/32767*180, label="Torque Demand (N)")
    ax5.plot(df[t], df["V"], color = "goldenrod", label="body speed m/s")
    ax5.plot(df[t], df[motorRPM]*12/41*0.2032*2*np.pi/60, color = "green", label="motor speed m/s")
    ax5.set_title("Speed + Braking")
    ax5.legend()

    ax6.plot(df[t], df[wheelSpeedFL]*0.44704, label = "FL")
    ax6.plot(df[t], df[wheelSpeedFR]*0.44704, label = "FR")
    ax6.plot(df[t], df[wheelSpeedBL]*0.44704, label = "BL")
    ax6.plot(df[t], df[wheelSpeedBR]*0.44704, label = "BR")
    ax6.set_title("Wheel Speeds")

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Temperature (C)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Voltage (V)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Voltage (V)")
    ax4.set_xlabel("Longitude (deg)")
    ax4.set_ylabel("Latitude (deg)")
    ax5.set_xlabel("Time (s)")
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("Speed (m/s)")

    plt.suptitle(title)
    plt.show()

# df1 = read("FS-4/Jun102026/162207.parquet") # Car was wiggled a bit and wheels spun by hand on jacks
df2 = read("FS-4/Jun102026/180316.parquet") # 400A/500A current limit. Only really 400A brief driving
# df3 = read("FS-4/Jun102026/072050.parquet") # 5 hrs, basically nothing
# df4 = read("FS-4/Jun102026/152519.parquet") # 45 min, basically nothing
df5 = read("FS-4/Jun102026/181830.parquet") # Tiny bits of driving. 0.7V voltage drop. 500/600/650 A limit.
df6 = read("FS-4/Jun102026/171744.parquet") # 45 min. 200A. 0.2V voltage drop
# df7 = read("FS-4/Jun102026/145454.parquet") # 35 min. Basically nothing
# df8 = read("FS-4/Jun102026/070347.parquet") # 8 min. Basically nothing

# dfs = [df1, df2, df3, df4, df5, df6, df7, df8]

t = "Time"

# for i, df in enumerate(dfs):
    # plt.scatter(np.arange(df.height), df[t], label = ("df" + str(i)), s=0.5)
# plt.xlabel("row")
# plt.ylabel("time (s)")
# plt.legend()
# plt.show()

plt.plot(df6.filter(pl.col(t) < 260).filter(pl.col(t)>240)[t])
plt.plot(np.arange(0, df6[t].max(), df6[t].max()/(df6.height + 1))[:df6.height]) #type:ignore
plt.show()

df6.filter(pl.col(t) < 260).filter(pl.col(t)>240)

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as fft
from statsmodels.nonparametric.smoothers_lowess import lowess

df = pl.read_parquet("../fs-data/FS-3/03162026/2_steeper_regen_curve.parquet").fill_null(strategy="forward").fill_null(strategy="backward")
df = df2

[x for x in df.columns if "VDM" in x]


FR_base_value = df["TPERIPH_FR_DATA_SUSTRAVEL"][:100].mean()
FL_base_value = df["TPERIPH_FL_DATA_SUSTRAVEL"][:100].mean()
BR_base_value = df["TPERIPH_BR_DATA_SUSTRAVEL"][:100].mean()
BL_base_value = df["TPERIPH_BL_DATA_SUSTRAVEL"][:100].mean()

t = df["Time_ms"]/1000.0
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(t, df["TPERIPH_FR_DATA_SUSTRAVEL"] - FR_base_value, label="TPERIPH_FR_DATA_SUSTRAVEL")
ax.plot(t, df["TPERIPH_FL_DATA_SUSTRAVEL"] - FL_base_value, label="TPERIPH_FL_DATA_SUSTRAVEL")
ax.plot(t, df["TPERIPH_BR_DATA_SUSTRAVEL"] - BR_base_value, label="TPERIPH_BR_DATA_SUSTRAVEL")
ax.plot(t, df["TPERIPH_BL_DATA_SUSTRAVEL"] - BL_base_value, label="TPERIPH_BL_DATA_SUSTRAVEL")
ax.plot(t, df["VDM_Z_AXIS_YAW_RATE"]/2, label="VDM_Z_AXIS_YAW_RATE")
ax.plot(t, df["SME_TRQSPD_Speed"]/1000, label="SME_TRQSPD_Speed")
ax.legend()
ax.set_xlabel("Time [s]")
ax.set_ylabel("TPERIPH Data")
ax.set_title("Suspension Damping View")
ax.grid(True)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(t, -1*df["TMAIN_DATA_STEERING"], label="TMAIN_DATA_STEERING")
ax.plot(t, df["VDM_Z_AXIS_YAW_RATE"], label="VDM_Z_AXIS_YAW_RATE")
ax.legend()
ax.set_xlabel("Time [s]")
ax.set_ylabel("Steering and Yaw Rate")
ax.set_title("Steering and Yaw Rate")
ax.grid(True)
plt.show()

df = df.filter(pl.col("SME_TRQSPD_Speed") > 5000)

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
ax1.hist(df["TPERIPH_FR_DATA_SUSTRAVEL"], bins=401)
ax1.set_title("FR Suspension Travel")
ax1.set_xlabel("Suspension Travel (mm)")
ax2.hist(df["TPERIPH_FL_DATA_SUSTRAVEL"], bins=401)
ax2.set_title("FL Suspension Travel")
ax2.set_xlabel("Suspension Travel (mm)")
ax3.hist(df["TPERIPH_BR_DATA_SUSTRAVEL"], bins=401)
ax3.set_title("BR Suspension Travel")
ax3.set_xlabel("Suspension Travel (mm)")
ax4.hist(df["TPERIPH_BL_DATA_SUSTRAVEL"], bins=401)
ax4.set_title("BL Suspension Travel")
ax4.set_xlabel("Suspension Travel (mm)")
fig.show()

suspension_FR_fft = fft.fft((df["TPERIPH_FR_DATA_SUSTRAVEL"] - FR_base_value).to_numpy())
suspension_FL_fft = fft.fft((df["TPERIPH_FL_DATA_SUSTRAVEL"] - FL_base_value).to_numpy())
suspension_BR_fft = fft.fft((df["TPERIPH_BR_DATA_SUSTRAVEL"] - BR_base_value).to_numpy())
suspension_BL_fft = fft.fft((df["TPERIPH_BL_DATA_SUSTRAVEL"] - BL_base_value).to_numpy())
suspension_FR_freq = fft.fftfreq(len(suspension_FR_fft), d=0.01)
suspension_FL_freq = fft.fftfreq(len(suspension_FL_fft), d=0.01)
suspension_BR_freq = fft.fftfreq(len(suspension_BR_fft), d=0.01)
suspension_BL_freq = fft.fftfreq(len(suspension_BL_fft), d=0.01)

def rollingMean(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1.scatter(suspension_FR_freq, np.abs(suspension_FR_fft), label="TPERIPH_FR_DATA_SUSTRAVEL", s=0.5)
ax1.set_title("FR Suspension FFT")
ax1.set_xlabel("Frequency [Hz]")
ax1.set_ylabel("Magnitude")
ax1.set_yscale("log")
ax1.grid(True)
ax2 = fig.add_subplot(222)
ax2.scatter(suspension_FL_freq, np.abs(suspension_FL_fft), label="TPERIPH_FL_DATA_SUSTRAVEL", s=0.5)
ax2.set_title("FL Suspension FFT")
ax2.set_xlabel("Frequency [Hz]")
ax2.set_ylabel("Magnitude")
ax2.set_yscale("log")
ax2.grid(True)
ax3 = fig.add_subplot(223)
ax3.scatter(suspension_BR_freq, np.abs(suspension_BR_fft), label="TPERIPH_BR_DATA_SUSTRAVEL", s=0.5)
ax3.set_title("BR Suspension FFT")
ax3.set_xlabel("Frequency [Hz]")
ax3.set_ylabel("Magnitude")
ax3.set_yscale("log")
ax3.grid(True)
ax4 = fig.add_subplot(224)
ax4.scatter(suspension_BL_freq, np.abs(suspension_BL_fft), label="TPERIPH_BL_DATA_SUSTRAVEL", s=0.5)
ax4.set_title("BL Suspension FFT")
ax4.set_xlabel("Frequency [Hz]")
ax4.set_ylabel("Magnitude")
ax4.set_yscale("log")
ax4.grid(True)
plt.tight_layout()
plt.show()


fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1.scatter(suspension_FR_freq, rollingMean(np.abs(suspension_FR_fft), 401), label="TPERIPH_FR_DATA_SUSTRAVEL", s=0.5)
ax1.set_title("FR Suspension FFT")
ax1.set_xlabel("Frequency [Hz]")
ax1.set_ylabel("Magnitude")
ax1.set_yscale("log")
ax1.grid(True)
ax2 = fig.add_subplot(222)
ax2.scatter(suspension_FL_freq, rollingMean(np.abs(suspension_FL_fft), 401), label="TPERIPH_FL_DATA_SUSTRAVEL", s=0.5)
ax2.set_title("FL Suspension FFT")
ax2.set_xlabel("Frequency [Hz]")
ax2.set_ylabel("Magnitude")
ax2.set_yscale("log")
ax2.grid(True)
ax3 = fig.add_subplot(223)
ax3.scatter(suspension_BR_freq, rollingMean(np.abs(suspension_BR_fft), 401), label="TPERIPH_BR_DATA_SUSTRAVEL", s=0.5)
ax3.set_title("BR Suspension FFT")
ax3.set_xlabel("Frequency [Hz]")
ax3.set_ylabel("Magnitude")
ax3.set_yscale("log")
ax3.grid(True)
ax4 = fig.add_subplot(224)
ax4.scatter(suspension_BL_freq, rollingMean(np.abs(suspension_BL_fft), 401), label="TPERIPH_BL_DATA_SUSTRAVEL", s=0.5)
ax4.set_title("BL Suspension FFT")
ax4.set_xlabel("Frequency [Hz]")
ax4.set_ylabel("Magnitude")
ax4.set_yscale("log")
ax4.grid(True)
plt.tight_layout()
plt.show()

yawRate_fft = fft.fft(df["VDM_Z_AXIS_YAW_RATE"].to_numpy())
yawRate_freq = fft.fftfreq(len(yawRate_fft), d=0.01)
pitchRate_fft = fft.fft(df["VDM_Y_AXIS_YAW_RATE"].to_numpy())
pitchRate_freq = fft.fftfreq(len(pitchRate_fft), d=0.01)
rollRate_fft = fft.fft(df["VDM_X_AXIS_YAW_RATE"].to_numpy())
rollRate_freq = fft.fftfreq(len(rollRate_fft), d=0.01)


meanSegments = 101
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(yawRate_freq, np.convolve(np.ones(meanSegments)*1/meanSegments, np.abs(yawRate_fft), mode='same'), label="Yaw", s=0.5)
ax.scatter(pitchRate_freq, np.convolve(np.ones(meanSegments)*1/meanSegments, np.abs(pitchRate_fft), mode='same'), label="Pitch", s=0.5)
ax.scatter(rollRate_freq, np.convolve(np.ones(meanSegments)*1/meanSegments, np.abs(rollRate_fft), mode='same'), label="Roll", s=0.5)
ax.legend()
ax.set_title("Yaw Rate FFT")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Magnitude")
ax.set_yscale("log")
ax.grid(True)
plt.show()

## heave based on z acceleration
heave_fft = fft.fft(df["VDM_Z_AXIS_ACCELERATION"].to_numpy())
heave_freq = fft.fftfreq(len(heave_fft), d=0.01)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(heave_freq, np.convolve(np.ones(meanSegments)*1/meanSegments, np.abs(heave_fft), mode='same'), label="Heave", s=0.5)
ax.legend()
ax.set_title("Heave FFT")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Magnitude")
ax.set_yscale("log")
ax.grid(True)
plt.show()