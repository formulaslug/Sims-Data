import polars as pl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

selected_columns = ["FZ", "SR", "SA", "V", "FX", "NFX", "NFY"]
df1 = pl.read_csv("revisedLatSlipData.csv", separator=",") 
#df1 = pl.read_csv("inputData.dat", separator="	", skip_rows=1, skip_rows_after_header=1)
df2 = pl.read_csv("slipData.csv", separator=",")
df2 = df2.with_columns(pl.lit(0.0).alias("SR"))

df = pl.concat([df1[selected_columns], df2[selected_columns]], how="vertical")
#df = df.filter((pl.col("SA") > -0.1) & (pl.col("SA") < 0.1))
df = df.filter((pl.col("SR") > -1.05) & (pl.col("SR") < 1.05))

df = df.with_columns([
    pl.col("SA").abs().alias("SA"),
    pl.col("SR").abs().alias("SR"),
    pl.col("NFY").abs().alias("NFY")
])


df.write_csv('lateralDataSet.csv')


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df["SR"], df["SA"], df["NFY"], c='r', marker='o') 


ax.set_xlabel('SR')
ax.set_ylabel('SA')
ax.set_zlabel('NFY')
plt.show()
"""

fig = plt.figure()
plt.scatter(df["SA"], df["NFY"], s = 1, color="red")
plt.show()
"""