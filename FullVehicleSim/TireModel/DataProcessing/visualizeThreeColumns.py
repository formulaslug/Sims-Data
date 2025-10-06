import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

file_path = 'LC0-6.csv'
data = pd.read_csv(file_path)

x_column = 'SA'
y_column = 'SR'
z_column = 'NFX'

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(data[x_column], data[y_column], data[z_column], alpha=0.1, color='blue')

ax.set_xlabel(x_column)
ax.set_ylabel(y_column)
ax.set_zlabel(z_column)
ax.set_title('🐧 3d')

plt.show()
