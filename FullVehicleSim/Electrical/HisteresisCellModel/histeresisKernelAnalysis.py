import numpy as np
import polars as pl
import matplotlib.pyplot as plt

df = pl.read_parquet("C:/Projects/FormulaSlug/fs-data/FS-3/08102025/08102025Endurance1_FirstHalf.parquet")[5:]

I = "SME_TEMP_BusCurrent"
V = "SME_TEMP_DC_Bus_V"

dt = 0.01
kernel_duration = 20.0
kernel_size = int(kernel_duration / dt)
t = np.arange(kernel_size*dt,0, -dt)


sigmas = [0.1, 0.2, 0.3, 0.4, 0.5]

for sigma in sigmas:
    kernel = np.exp(-(t**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    plt.plot(kernel, label=f"{sigma=}")
plt.legend()
plt.show()

for sigma in sigmas:
    kernel = np.exp(-(t**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    kernel = np.append(kernel, np.zeros_like(kernel))
    plt.plot(np.convolve(df[I], kernel, mode="same")/50, label=f"convolved current - {sigma}")
plt.plot(df[I]/50, label="current")
plt.plot(-0.5 * (df[V] - df[V][100]), label="voltage drop")
plt.legend()
plt.show()

