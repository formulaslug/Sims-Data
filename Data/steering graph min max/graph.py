import polars as pl
import matplotlib.pyplot as plt

# Load the parquet file using polars
df = pl.read_parquet("/Users/aanyajain/Desktop/FS-2/Parquet/2025-03-06-Part2.parquet")

col = "TELEM_STEERBRAKE_STEER"

# Filter outliers using polars
df_clean = df.filter(
    (pl.col(col) > 31500) & (pl.col(col) < 34000)
)

# Get min and max using polars
print("Min:", df_clean[col].min())
print("Max:", df_clean[col].max())

# Smooth using polars rolling mean
smoothed = df_clean[col].rolling_mean(window_size=1000)

# Plot
plt.figure(figsize=(14, 6))
plt.plot(smoothed.to_numpy(), color='steelblue', linewidth=1.5)
plt.title("Steering Angle - 2025-03-06-Part2 (Cleaned)")
plt.xlabel("Sample")
plt.ylabel("Steering")
plt.tight_layout()
plt.savefig("/Users/aanyajain/Desktop/steering_graph.png", dpi=150)
plt.show()

print("Graph saved to Desktop!")
