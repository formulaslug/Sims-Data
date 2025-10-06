import matplotlib.pyplot as plt
import polars as pl

train = pl.read_csv("combinedLateralData.csv", separator=",")

train = train.filter((pl.col("SR") < 0.01) & (pl.col("SR") > -0.01))

train.write_csv('pureLateralData.csv') 