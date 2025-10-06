import polars as pl

import matplotlib.pyplot as plt


train = pl.read_csv("unifiedTraining.csv", separator=",")

train = train.filter((pl.col("SA") < 0.1) & (pl.col("SA") > -0.1))
train = train.filter((pl.col("SR") >= -1) & (pl.col("SR") <= 1))
train = train.filter( ((pl.col("SR") < -0.8) & (pl.col("NFX") > 0.2)) | (pl.col("SR") > -0.8) )

#train = train.filter( ((pl.col("SR") < -0.8) & (pl.col("NFX") > 0.25)) | (pl.col("SR") > -0.8) )

# Spline options
train = train.filter( ((pl.col("SR") < -0.03) | (pl.col("SR") > 0.03)))

# Add slip ratio data on the other side of sr when sr = 1 not just 01
train = train.vstack(
    train.filter(pl.col("SR") < -0.5).with_columns(
        pl.col("SR") * -1, 
        pl.col("NFX") * -1
    )
)

#print("WRITE")
train.write_csv('pureLongSlipDataSpline.csv') 
