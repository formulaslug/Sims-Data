import polars as pl
import numpy as np
import matplotlib.pyplot as plt

from Data.FSLib.AnalysisFunctions import read, simpleTimeCol

path = "FS-3/08172025/08172025_26autox1.parquet"

df = read(path, fs2or3=True)
df = df.insert_column(0, simpleTimeCol(df))

t = df["Time"]

[x for x in df.columns if "PEDAL" in x]

plt.plot(t, df["SME_TRQSPD_Speed"], label = "RPM")
plt.plot(t, df["SME_TRQSPD_Torque"], label = "SME_TRQSPD_Torque")
plt.plot(t, df["ETC_STATUS_PEDAL_TRAVEL"], label = "ETC_STATUS_PEDAL_TRAVEL")
plt.legend()
plt.show()