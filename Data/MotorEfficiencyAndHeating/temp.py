import polars as pl
import numpy as np
import matplotlib.pyplot as plt

df = pl.read_csv("Data/MotorEfficiencyAndHeating/data.csv")

df1 = df.filter(pl.col("current") == 900).filter(pl.col("temperature") == 20)

plt.scatter(df1["rpm"], df1["torque"], c=df1["efficiency"], cmap="viridis")
plt.xlabel("RPM")
plt.ylabel("Torque")
plt.title("Motor Efficiency")
plt.colorbar(label="Efficiency")
plt.show()