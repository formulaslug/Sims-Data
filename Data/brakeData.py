import matplotlib.pyplot as plt
import polars as pl

df = pl.read_parquet("FS-2/Parquet/2025-03-06-BrakingTests1.parquet")
r = df.select("TELEM_STEERBRAKE_BRAKER") / 32767 * 5
f = df.select("TELEM_STEERBRAKE_BRAKEF") / 32767 * 5
time = df.select("Seconds")

plt.plot(time, f, label="TELEM_STEERBRAKE_BRAKEF")
plt.plot(time, r, label="TELEM_STEERBRAKE_BRAKER")
plt.legend()
plt.show()
