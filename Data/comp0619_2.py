import polars as pl
import numpy as np
import matplotlib.pyplot as plt

from Data.FSLib.AnalysisFunctions import read

df = read("FS-4/Comp/brake.parquet")

VCU_ACCEL_PEDAL_TRAVEL = "VCU_ACCEL_PEDAL_TRAVEL"
SME_THROTL_TorqueDemand = "SME_THROTL_TorqueDemand"
VCU_ETC_IMPLAUS_APPS_OUT_OF_RANGE = "VCU_ETC_IMPLAUS_APPS_OUT_OF_RANGE"
VCU_ETC_IMPLAUS_BPPS_OUT_OF_RANGE = "VCU_ETC_IMPLAUS_BPPS_OUT_OF_RANGE"
VCU_ETC_IMPLAUS_APPS_DEVIATION = "VCU_ETC_IMPLAUS_APPS_DEVIATION"
VCU_ETC_IMPLAUS_BSE_OUT_OF_RANGE = "VCU_ETC_IMPLAUS_BSE_OUT_OF_RANGE"
VCU_ETC_IMPLAUS_BRAKE_AND_ACCEL = "VCU_ETC_IMPLAUS_BRAKE_AND_ACCEL"

t = df["Time_ms"]/1000

plt.plot(t, df[VCU_ACCEL_PEDAL_TRAVEL], label="Accel Pedal Travel")
plt.plot(t, df[SME_THROTL_TorqueDemand], label="Torque Demand")
plt.legend()
plt.show()

plt.plot(t, df[VCU_ETC_IMPLAUS_APPS_OUT_OF_RANGE], label="APPS Out of Range")
plt.plot(t, df[VCU_ETC_IMPLAUS_BPPS_OUT_OF_RANGE], label="BPPS Out of Range")
plt.plot(t, df[VCU_ETC_IMPLAUS_APPS_DEVIATION], label="APPS Deviation")
plt.plot(t, df[VCU_ETC_IMPLAUS_BSE_OUT_OF_RANGE], label="BSE Out of Range")
plt.plot(t, df[VCU_ETC_IMPLAUS_BRAKE_AND_ACCEL], label="Brake and Accel Implausible")
plt.legend()
plt.show()

plt.plot(t, df["VCU_ACCEL_PEDAL_TRAVEL"], label="Accel Pedal Travel")
plt.plot(t, df["VCU_BRAKE_PEDAL_TRAVEL"], label="Brake Pedal Travel")
plt.legend()
plt.show()