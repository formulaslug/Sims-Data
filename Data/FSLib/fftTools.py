import numpy as np
import polars as pl
import scipy.fftpack as fftpack


## Low pass filter that retains the fraction input of the signal closest to 0 Hz eg. 0.1 removes top 10% of frequencies. 0.7-0.9 is a good range for most signals.
def low_pass_filter(array, fraction):
    if type(array) == pl.Series:
        array = array.to_numpy()
    transform = fftpack.fft(array)
    length = array.shape[0]
    end_length = ((1 - fraction)/2)
    start = round(length*end_length)
    end = round(length*(1-end_length))
    transform[start:end] = np.zeros_like(transform[start:end])
    output = fftpack.ifft(transform)
    # output.real
    return output.real