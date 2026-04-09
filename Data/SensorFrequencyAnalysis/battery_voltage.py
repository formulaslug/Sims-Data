import scipy
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

df = pl.read_parquet('/Users/aanyajain/Documents/GitHub/fs-data/FS-3/01112026/011026-14.parquet')
df = df.fill_null(strategy="forward").fill_null(strategy="backward")
signal = df["ACC_POWER_PACK_VOLTAGE"].to_numpy()
fft = scipy.fft.fft(signal)
freqs = scipy.fft.fftfreq(len(signal), d=0.01)
attempts = [5.5]
ffts = []
iffts = []
for i, attempt in enumerate(attempts):
    ffts.append(fft.copy())
    ffts[-1][np.log(np.abs(fft)) < attempt] = 0
    iffts.append(scipy.fft.ifft(ffts[-1]))
plt.plot(signal, label="Real")
for ifft, attempt in zip(iffts, attempts):
    plt.plot(ifft.real, label=f"Attempt {attempt}")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.show()
plt.scatter(freqs, np.log(np.abs(fft)), s=0.25)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Battery Voltage - Frequency Spectrum")
plt.show()
