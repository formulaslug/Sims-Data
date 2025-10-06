import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

# Load the CSV file

data = pl.read_csv("Data/LC0-6.csv", separator=",")

x_column = 'SA'
y_column = 'NFY'

plt.figure(figsize=(10, 6))
plt.scatter(data[x_column], data[y_column], alpha=0.5)

plt.xlabel(x_column)
plt.ylabel(y_column)
plt.grid(True)
plt.show()
