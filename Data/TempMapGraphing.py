import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from slice_viewer import SliceViewer

from Data.AnalysisFunctions import *

df = read("C:/Projects/FormulaSlug/fs-data/FS-3/08102025/08102025Endurance1_SecondHalf.parquet")
df = df.insert_column(0, simpleTimeCol(df))
df.shape
t = "Time"
s0c0 = "ACC_SEG0_TEMPS_CELL0"
s0t0 = "ACC_SEG0_VOLTS_CELL0"

seg0 = [i for i in df.columns if i.startswith("ACC_SEG0_TEMPS")]
segs = [[i for i in df.columns if i.startswith(f"ACC_SEG{j}_VOLTS")] for j in range(5)]
segs

dftt = df.filter(pl.col(s0c0) != 0)[seg0]
nptts = np.array([(df.filter(pl.col(s0t0) != 0)[seg]).to_numpy().T for seg in segs])
nptt = dftt.to_numpy()

# npttsList = [[nptts[i,:3,:]]+[nptts[i,3:,:]] for i in range (5)]
npttsList = [[nptts[i,:3,:]]+[np.flip(nptts[i,3:,:], 0)] for i in range (5)]
nptts1 = np.array([item for sublist in npttsList for item in sublist])

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
x = SliceViewer(nptts1)
x.show()

# plt.plot(dftt[s0c0])
plt.imshow(nptt.T, aspect=5000)
plt.title("Seg0")
plt.xlabel("Time (s)")
plt.yticks([0,1,2,3,4,5],[f"Cell{i}" for i in range(6)])
plt.colorbar()
plt.show()