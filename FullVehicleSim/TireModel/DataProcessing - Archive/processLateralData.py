import matplotlib.pyplot as plt
import polars as pl

train = pl.read_csv("unifiedTraining.csv", separator=",")

train = train.filter((pl.col("NFY") >= -4.5) & (pl.col("NFY") <= 4.5))

train = train.with_columns(
    pl.when(pl.col("SR") < -0.8).then(0).otherwise(pl.col("SR")).alias("SR"),
    pl.when(pl.col("SA") < 0).then(pl.col("SA").abs()).otherwise(pl.col("SA")).alias("SA"),
    pl.when(pl.col("SA") < 0).then(pl.col("NFY").abs()).otherwise(pl.col("NFY")).alias("NFY")
)

train = train.filter( (pl.col("SA") > 0.05) | (pl.col("NFY") >= -1) & (pl.col("NFY") <= 1))

train.write_csv('combinedLateralData.csv') 