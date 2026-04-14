import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("./Data")
from Data.FSLib.IntegralsAndDerivatives import *
dfa = pl.read_parquet("fs-data/FS-3/08172025/08172025_27autox2&45C_35C_~28Cambient_100fans.parquet")
dfb = pl.read_parquet("fs-data/FS-3/08172025/08172025_28autox3&4_45C_40C_~29Cambient_0fans.parquet")
print(dfa)
print(dfb)
print(dfa.columns)
print(dfb.columns)

