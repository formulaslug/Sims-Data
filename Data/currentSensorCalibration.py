import polars as pl
import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy as np

Expected = np.asarray([0,1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,150])*10
Actual = np.asarray([14.4,14.5,14.5,15,15.9,16.7,17.6,18.5,19.5,20.5,21,25,30,34,39,43,48,52,57,61,66,70,75,79,84,88,92,97,101,147.9])*10

sActual = pl.Series("Actual", Actual)
sExpected = pl.Series("Expected", Expected)
df = pl.DataFrame({"Actual": sActual, "Expected": sExpected})

# plt.plot(df["Actual"], df["Expected"], marker='o', linestyle='-')
# plt.show()

def curve_fun (x, bias, lin):
    return bias+x*lin

# args = opt.curve_fit(curve_fun, df["Actual"]/10, df["Expected"]/10)
args = opt.curve_fit(curve_fun, df["Actual"], df["Expected"],p0=[10,0.8])
bias, lin = args[0]
plt.plot(df["Expected"],curve_fun(df["Actual"], bias, lin))
plt.plot(df["Expected"], df["Actual"])
# plt.plot()
plt.legend(["Curve Fit", "Raw Data"])
plt.xlabel("Expected, Current (dA)")
plt.ylabel("Current (dA)")
plt.show()

args[0]

