import polars as pl
import matplotlib.pyplot as plt

#train = pl.read_csv("lateralDataSet.csv", separator="	", skip_rows=1, skip_rows_after_header=1) 
train = pl.read_csv("Data/lateralDataSet.csv") 

train = train.with_columns(pl.col("SR").round(2))
train = train.with_columns(pl.col("SA").round(2))

avg = train.group_by(["SA", "SR"]).agg(pl.col("NFY").mean().alias("NFY"), pl.col("FZ").mean().alias("FZ"), pl.col("V").mean().alias("V"))

avg.write_csv("Data/lateralDataSetAvg.csv")


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(avg["SR"], avg["SA"], avg["NFY"], s=1, color="orange")

ax.set_xlabel('SR')
ax.set_ylabel('SA')
ax.set_zlabel('NFY')

plt.show()
