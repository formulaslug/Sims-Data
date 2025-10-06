# Documentation https://docs.google.com/document/d/1oGsGDnY0DEKWpE3S6481A9yZ0F9qUEwWkSXJwTSz4E4/edit?pli=1&tab=t.93l7bg7vvy8c
import polars as pl

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

train = pl.read_csv("slipAngleData.dat", separator="	", skip_rows=1, skip_rows_after_header=1) #pl.read_csv("train-slipangle.csv", separator=",")") < 10))

train = train.with_columns([
    pl.col("SA").abs().alias("SA"),
    pl.col("SR").abs().alias("SR"),
    pl.col("NFY").abs().alias("NFY")
])



train = train.filter((pl.col("SR") > -1.005) & (pl.col("SR") < 0.25)) # Step 1

train = train.filter((pl.col("NFY") < 3)) # Step 2

train = train.filter((pl.col("SA") > -11.5) & (pl.col("SA") < 11.5)) # Step 3


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(train["SR"], train["SR"], train["NFY"], c='blue', marker='o')


train.write_csv('revisedLatSlipData.csv')

plt.show()

#train = train.with_columns(train.select(pl.col("NFY").abs().alias("NFY")))

#mean_nfx = train.group_by("SA").agg(pl.col("NFY").mean().alias("mean_NFY"))

#train_with_mean = train.join(mean_nfx, on="SA")

#threshold = 0.05 # arbitrary filter value
#train = train_with_mean.filter(pl.col("NFX") >= (pl.col("mean_NFX") * threshold))


#print("WRITE")
