import polars as pl
import numpy as np
import matplotlib.pyplot as plt

from Data.FSLib.AnalysisFunctions import read

paths = ["FS-4/Comp/rain_test/035625.parquet",
         "FS-4/Comp/rain_test/035117.parquet",
         "FS-4/Comp/rain_test/041704.parquet",
         "FS-4/Comp/rain_test/041155.parquet"
         ]
shutdown_final = "BATT_STATUS_SHUTDOWN_FINAL"
imd_fault = "BATT_STATUS_IMD_FAULT"
bus_voltage = "SME_TEMP_DC_Bus_V"

dfs = [read(path) for path in paths]

for df in dfs:
    t = df["Time_ms"]/1000
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax1 = fig.add_subplot(212)
    ax.plot(t, df[shutdown_final], label = "shutdown final")
    ax.plot(t, df[imd_fault], label = "IMD fault")
    ax.legend()
    ax1.plot(t, df[bus_voltage])
    fig.show()