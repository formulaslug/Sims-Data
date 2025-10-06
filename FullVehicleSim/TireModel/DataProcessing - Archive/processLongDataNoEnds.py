import polars as pl

import matplotlib.pyplot as plt


train = pl.read_csv("unifiedTraining.csv", separator=",")

train = train.filter((pl.col("SR") >= -1) & (pl.col("SR") <= 1))
train = train.filter((pl.col("SR") > -0.8) & (pl.col("SR") < 0.8))
train = train.filter((pl.col("SR") > -0.8) & (pl.col("SR") < 0.8))
train = train.filter((pl.col("SR") < -0.05) | (pl.col("SR") > 0.05))
#print("WRITE")

train.write_csv('pureLongSlipDataNoEnds.csv') 
