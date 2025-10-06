import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

data = pl.read_csv("Data/R20-6.csv", separator=",")

data = data.with_columns([
    pl.col("NFY").cast(pl.Float64, strict=False),
    pl.col("NFX").cast(pl.Float64, strict=False),
    pl.col("SA").cast(pl.Float64, strict=False),
    pl.col("SR").cast(pl.Float64, strict=False)
])

# Move SR data that's broken
data = data.with_columns([
    pl.when((pl.col("SR") <= -0.95) & (pl.col("SR") >= -1.05))
    .then(pl.col("SR") + 1)
    .otherwise(pl.col("SR"))
    .alias("SR")
])


# Absolute value SA/NFY
filtered_data = data.with_columns([
    pl.col("NFY").abs().alias("NFY"),
    pl.col("SA").abs().alias("SA")
])

# Filter out SR/SA data that's weird at 0
filtered_data = filtered_data.filter(
    pl.when((pl.col("SR") >= -0.05) & (pl.col("SR") <= 0.05))
    .then((pl.col("NFX") >= -2) & (pl.col("NFX") <= 2))
    .otherwise(True)
)
filtered_data = filtered_data.filter(
    pl.when((pl.col("SA") <= 0.05))
    .then(pl.col("NFY") <= 2)
    .otherwise(True)
)

# Filter out initial impossible data
filtered_data = filtered_data.filter(
    (pl.col("NFY") <= 6) & (pl.col("NFX") <= 6) & (pl.col("NFX") >= -6)
)

# Filter out SR impossible data
filtered_data = filtered_data.filter(
    (pl.col("SR") >= -1) & (pl.col("SR") <= 1)
)

# Remove datapoints where NFX < 0.05
filtered_data = filtered_data.filter(pl.col("NFX").abs() >= 0.1)

# Remove datapoints where NFY < 0.05
filtered_data = filtered_data.filter(pl.col("NFY").abs() >= 0.1)

# Remove datapoints within radius 0.1 of SR=0, SA=0
filtered_data = filtered_data.filter(
    (pl.col("SR")**2 + pl.col("SA")**2).sqrt() >= 0.1
)

# If not within radius 1 of SR=0, SA=0, reject points with NFX or NFY < 0.25
filtered_data = filtered_data.filter(
    pl.when((pl.col("SR")**2 + pl.col("SA")**2).sqrt() <= 1.0)
    .then(True)
    .otherwise((pl.col("NFX").abs() >= 0.15) & (pl.col("NFY").abs() >= 0.15))
)


plt.figure(figsize=(10, 6))
plt.scatter(filtered_data["SR"], filtered_data["NFX"], alpha=0.6)
plt.xlabel("SR")
plt.ylabel("NFX")
plt.grid(True, alpha=0.6)


filtered_data.write_csv("R20-6.csv")

plt.show()
