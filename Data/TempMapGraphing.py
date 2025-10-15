import polars as pl
import matplotlib.pyplot as plt
import numpy as np

from Data.AnalysisFunctions import *

df = read("C:/Projects/FormulaSlug/fs-data/FS-3/08102025/08102025Endurance1_FirstHalf.parquet")
df = df.insert_column(0, simpleTimeCol(df))
df.shape
t = "Time"
seg0 = [i for i in df.columns if i.startswith("ACC_SEG0_TEMPS")]

s0c0 = "ACC_SEG0_TEMPS_CELL0"

dftt = df.filter(pl.col("ACC_SEG0_TEMPS_CELL0") != 0)[seg0]
nptt = dftt.to_numpy()
nptt

# plt.plot(dftt[s0c0])
plt.imshow(nptt.T, aspect=5000)
plt.title("Seg0")
plt.xlabel("Time (s)")
plt.yticks([0,1,2,3,4,5],[f"Cell{i}" for i in range(6)])
plt.colorbar()
plt.show()
