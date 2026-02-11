import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from scipy.integrate import solve_ivp   #solve initial value problem

# Constants Parameters for the differential equation
R = 12.6316   #internal resistance of battery in ohms
m = 49.9   #mass of battery kg
c = 1000   #specific heat capacity of battery in J/(kg·K)

# Storing constants into a single variable
k = R/(m * c)     #let k = R/(m*c)

df = pl.read_parquet("Docs/Columns.csv")

df.select([a for a in df.columns if a.startswith("SME") and a[9] == "T"])

#Initital conditions
T0 = 22 # Tnitial temperature of the cell in °C
t_span = (0, 60)    #time span for the simulation in seconds
t_eval = np.linspace(0, 60, 600)   #interval to evaluate the solution

#differential equation
def thermal_ode(t_eval,I, k):
    """"
    This method returns the differential
    equation given current draw I from
    colums
    """
    I = df.select([a for a in df.columns if a.startswith("ACC") and a[9] == "T"]).mean_horizontal()  #takes average of all variables starting with ACC and the 10th index equaling T
    dT_dt = k * (I ** 2)
    # Numerical Integration using Runge Kutta method (RK45)
    return dT_dt
    
solution = solve_ivp(thermal_ode, T0, method = 'RK45', t_eval = t_eval, args = (k,), rtol = 1e-6, atol= 1e-9)

#plotting the results
plt.scatter(t_eval, thermal_ode(t_eval, df, k).y[0], s=1)
plt.xlabel("Time (s)")
plt.ylabel("Battery Temperature (°C)")
plt.title("Battery Thermal Model Over Time")
plt.show()
