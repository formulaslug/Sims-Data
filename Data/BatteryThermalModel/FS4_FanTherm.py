import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from Data.FSLib.IntegralsAndDerivatives import *
# from Data.integralsAndDerivatives import in_place_derive

def simpleTimeCol (dfa, dt=60/5035):
    return pl.Series("Time", np.arange(dfa.height) * dt)

dfa = pl.read_parquet("fs-data/FS-3/08172025/08172025_27autox2&45C_35C_~28Cambient_100fans.parquet")
dfb = pl.read_parquet("fs-data/FS-3/08172025/08172025_28autox3&4_45C_40C_~29Cambient_0fans.parquet")

dfa = dfa.with_columns(simpleTimeCol(dfa))[6788:]
dfb = dfb.with_columns(simpleTimeCol(dfb))[69330+370+6788:]


#time starts from 0
dfa = dfa.with_columns((pl.col("Time") - dfa["Time"][0]).alias("Time"))
dfb = dfb.with_columns((pl.col("Time") - dfb["Time"][0]).alias("Time"))
#rows = arr.shape[0]
#Derivative

temp_cols = [f"ACC_SEG{s}_TEMPS_CELL{c}" for s in range(5) for c in range(6)]

avg = 0
dfa = dfa.with_columns(
    dfa.select([pl.col(col).cast(pl.Float32) for col in temp_cols])
      .mean_horizontal()
      .alias("TempMean")
)

dfb = dfb.with_columns(
    dfb.select([pl.col(col).cast(pl.Float32) for col in temp_cols])
      .mean_horizontal()
      .alias("TempMean")
)

dfa = dfa.with_columns(pl.col("TempMean").rolling_mean(window_size=avg).alias("Smooth_TempMean"))
dfb = dfb.with_columns(pl.col("TempMean").rolling_mean(window_size=avg).alias("Smooth_TempMean"))

tempA = dfa["Smooth_TempMean"]
tempB = dfb["Smooth_TempMean"]

plt.plot(dfa["Time"], tempA, label="With Fans (RM)")
plt.plot(dfb["Time"], tempB, label="No Fans (RM)")
plt.xlabel("Time (s)")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.show()



timeA = dfa["Time"].to_numpy()
timeB = dfb["Time"].to_numpy()
tempA = dfa["TempMean"].to_numpy()
tempB = dfb["TempMean"].to_numpy()

dtA = np.mean(np.diff(timeA))  
dtB = np.mean(np.diff(timeB))

dTdt_A = in_place_derive(tempA, dtA)
dTdt_B = in_place_derive(tempB, dtB)
print(dTdt_A)
print(dTdt_B)

length = min(tempA.shape[0], tempB.shape[0])

def quadFit (x, a, b, c):
    return a*x**2 + b*x + c

ambientTemp = 28
fitLength = 4000
print(f"Length = {length}")
print(f"fitLength = {fitLength}")

args = curve_fit(quadFit, timeA[:fitLength], tempA[:fitLength]-ambientTemp, p0=[0,0,0], maxfev=20000)[0]
print(f"a,b,c = {args}")
a,b,c = args

plt.plot(tempA[:fitLength], label="Temp A")
plt.plot(quadFit(timeA[:fitLength], *args)+ambientTemp, label="Fit A")
plt.legend()
plt.show()

plt.plot(tempA[:fitLength]-ambientTemp, timeA[:fitLength]*2*a+b, label="Tangent A")
plt.legend()
plt.show()

args1 = curve_fit(quadFit, timeB[:fitLength], tempB[:fitLength]-ambientTemp, p0=[0,0,0], maxfev=20000)[0]
print(f"a2,b2,c2 = {args1}")
a1,b2,c2 = args1

plt.plot(tempB[:fitLength], label="Temp B")
plt.plot(quadFit(timeB[:fitLength], *args1)+ambientTemp, label="Fit B")
plt.legend()
plt.show()

plt.plot(tempB[:fitLength]-ambientTemp, timeB[:fitLength]*2*a+b, label="Tangent B")
plt.legend()
plt.show()

#prep the data
Y=2*a*timeA+b
Y2=2*a1*timeB+b2
X=np.array([np.ones_like(tempA)*100,tempA-ambientTemp])
X2=np.array([np.ones_like(tempB)*0,tempB-ambientTemp])

Ys = np.concatenate((Y, Y2), axis=0)
Xs = np.concatenate((X, X2), axis=1).T

mc = 967.6 # J/K
Area = 0.71 # m^2
print(f"Xs.shape = {Xs.shape}")
print(f"Ys.shape = {Ys.shape}")

def tempFit (x, c1, c2):
     ## Assuming first col of x is temp and second col is fan power
    fanPower = x[:,0]
    temp = x[:,1]
    coeff = (c2 + (1-c2)*fanPower/100)
    secondCoeff = coeff*c1*Area/mc
    return secondCoeff*(temp-ambientTemp)

args = curve_fit(tempFit, Xs, Ys, p0 = [0.3, 1])
print(f"c1,c2 = {args[0]}")

args = curve_fit(quadFit, timeB[:fitLength], tempB[:fitLength]-ambientTemp, p0=[0,0,0], maxfev=20000)[0]
print(f"a,b,c = {args}")

plt.plot(in_place_derive(tempA[:length]-tempB[:length]))
plt.show()

# print(1/(timeA[1] - timeA[0]))

"""print("dT/dt (Fans 100%):", dTdt_A[-1])
print("dT/dt (Fans 0%):", dTdt_B[-1])"""
