import scipy
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

df = pl.read_parquet('/Users/aanyajain/Documents/GitHub/fs-data/FS-3/11222025/11222025_1.parquet')
df = df.fill_null(strategy="forward").fill_null(strategy="backward")
signal = df["ACC_SEG0_TEMPS_CELL0"].to_numpy()

fft = scipy.fft.fft(signal)
freqs = scipy.fft.fftfreq(len(signal), d=0.01)

attempts = [4.5, 5.5, 6.5]
ffts = []
iffts = []
for i, attempt in enumerate(attempts):
    ffts.append(fft.copy())
    ffts[-1][np.log(np.abs(fft)) < attempt] = 0
    ifft = scipy.fft.ifft(ffts[-1])
    iffts.append(ifft)

RMSs = [np.sqrt(np.mean((ifft.real - signal)**2)) for ifft in iffts]
plt.plot(attempts, RMSs)
plt.xlabel("Attempt")
plt.ylabel("RMS")
plt.title("Cell Temp - RMS vs Attempt")
plt.show()

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
plt.title("Cell Temp - Frequency Spectrum")
plt.show()
