import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from Data.FSLib.slice_viewer import SliceViewer

from Data.FSLib.AnalysisFunctions import *

df = read("../fs-data/FS-3/08102025/08102025Endurance1_SecondHalf.parquet")

file1 = "../fs-data/FS-3/11222025/11222025_18.parquet"
file2 = "../fs-data/FS-3/11222025/11222025_19.parquet"
file3 = "../fs-data/FS-3/11222025/11222025_20.parquet"
file4 = "../fs-data/FS-3/11222025/11222025_21.parquet"
file5 = "../fs-data/FS-3/11222025/11222025_22.parquet"
file6 = "../fs-data/FS-3/11222025/11222025_23.parquet"

df = read(file1).vstack(read(file2)).vstack(read(file3)).vstack(read(file4)).vstack(read(file5)).vstack(read(file6))

df = df.insert_column(0, simpleTimeCol(df))
df.shape
t = "Time"
# s0c0 = "ACC_SEG0_TEMPS_CELL0"
# s0t0 = "ACC_SEG0_VOLTS_CELL0"

seg0 = [i for i in df.columns if i.startswith("ACC_SEG0_TEMPS")]
segs = [[i for i in df.columns if i.startswith(f"ACC_SEG{j}_TEMPS")] for j in range(5)]
segs

dftt = df.filter(pl.col(seg0[0]) != 0)[seg0]
nptts = np.array([(df.filter(pl.col(seg[0]) != 0)[seg]).to_numpy().T for seg in segs])
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