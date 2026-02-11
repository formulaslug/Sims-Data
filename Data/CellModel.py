import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import NearestNDInterpolator

I = "Current"
Q = "Charge"
V = "Voltage"

dfvtc5a = pl.read_csv("../../fs-data/FS-3/voltageTableVTC5A.csv")
dfvtc5 = pl.read_csv("../../fs-data/FS-3/voltageTableVTC5.csv")

CellInterpolatorVTC5A = NearestNDInterpolator(np.array([dfvtc5a[I], dfvtc5a[Q]]).T, dfvtc5a[V])
CellInterpolatorVTC5 = NearestNDInterpolator(np.array([dfvtc5[I], dfvtc5[Q]]).T, dfvtc5[V])

# CellInterpolatorVTC5A([(1,1), (1,2)])