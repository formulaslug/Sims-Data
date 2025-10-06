import matplotlib.pyplot as plt
import polars as pl

train = pl.read_csv("combinedLateralData.csv", separator=",")

train = train.filter((pl.col("SR") > 0.01) | (pl.col("SR") < -0.01))
train = train.filter((pl.col("NFY") > 0.5))


train.write_csv('isolatedCombinedLateralData.csv') 