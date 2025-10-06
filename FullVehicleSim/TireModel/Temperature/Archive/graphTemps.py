import polars as pl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#df = pl.read_csv('latSlipData.csv')
df = pl.read_csv('lateralDataSet.csv')


#df = df.filter(pl.col('NFX') <= 0.05)



v = df['SR'].to_numpy()
mph = df['SA'].to_numpy()
#time = df['TSTO'].to_numpy()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(v, mph, c='blue', marker='o')

ax.set_xlabel('SR')
ax.set_ylabel('SA')
ax.set_zlabel('TSTI')
ax.set_title('gg temp')

plt.show()

