import scipy
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

df = pl.read_parquet("../fs-data/FS-3/01112026/011026-5.parquet")

df = df.fill_null(strategy="forward").fill_null(strategy="backward")

df["ETC_STATUS_HE1"]
plt.plot(df["ETC_STATUS_HE1"])
plt.show()

fft = scipy.fft.fft(df["ETC_STATUS_HE1"].to_numpy())
fft
freqs = scipy.fft.fftfreq(len(df["ETC_STATUS_HE1"]), d=0.01)

plt.scatter(freqs, np.log(np.abs(fft)), s=0.25)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Frequency Spectrum")
plt.show()

attempts = [10, 10.5, 11, 11.5, 11.75, 12, 12.5]
ffts = []
iffts = []

for i, attempt in enumerate(attempts):
    ffts.append(fft.copy())
    ffts[-1][np.log(np.abs(fft)) < attempt] = 0
    ifft = scipy.fft.ifft(ffts[-1])
    iffts.append(ifft)

plt.plot(df["ETC_STATUS_HE1"], label="Real")

for ifft, attempt in zip(iffts, attempts):
    plt.plot(ifft.real, label=f"Attempt {attempt}")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.show()