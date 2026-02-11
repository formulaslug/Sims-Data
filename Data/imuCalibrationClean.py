from Data.FSLib.IMUCalibrationFunctions import *

df = pl.read_parquet("Nathaniel_IMU_Data/Orientation.pq")
# print(df.columns)
df = df.insert_column(0, (df["xAcc"].pow(2) + df["yAcc"].pow(2) + df["zAcc"].pow(2)).sqrt().alias("mA")) #Magnitude of the 3 axis vector
df = df.insert_column(0, pl.Series((np.ones(df.shape[0])*1000)).alias("outA")) #Insert expected output column. Will be cut down to size automatically
plt.plot(df["voltage"])
plt.show()

df.columns
args = vector_imu_calibrate(df=df, column_names=("time", "mA", "xAcc", "yAcc", "zAcc", "outA"), min_cut_size=500, cut_trigger_height=40, plot=True, verbose=False, median_filter=True, median_filter_value=200,starting_values=[.95, 1.01, .99, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
print(args)


print(df.columns)
df = df.insert_column(0, (df["xBField"].pow(2) + df["yBField"].pow(2) + df["zBField"].pow(2)).sqrt().alias("mB")) #Magnitude of the 3 axis vector
