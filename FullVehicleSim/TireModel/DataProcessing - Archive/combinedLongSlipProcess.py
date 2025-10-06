import polars as pl

import matplotlib.pyplot as plt


train = pl.read_csv("unifiedTraining.csv", separator=",")

### PURE PROCESSING IS SIMILAR other than sa stuff and threshold for nfx has gone down to 0.1 for rejection
train = train.filter((pl.col("SR") >= -1) & (pl.col("SR") <= 1))
train = train.filter( ((pl.col("SR") < -0.8) & (pl.col("NFX") > 0.1)) | (pl.col("SR") > -0.8) )


# Spline options
#train = train.filter( ((pl.col("SR") < -0.03) | (pl.col("SR") > 0.03)))

train = train.vstack(
    train.filter(pl.col("SR") < -0.5).with_columns(
        pl.col("SR") * -1, 
        pl.col("NFX") * -1
    )
)

### COMBINED PROCESSING
# Make all data positive
train = train.with_columns(
    pl.when(pl.col("SA") < 0)
    .then(-pl.col("SA"))
    .otherwise(pl.col("SA"))
    .alias("SA")
)

train.write_csv('combinedLongSlipData.csv') 
