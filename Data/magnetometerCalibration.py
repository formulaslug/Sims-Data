import polars as pl
import numpy as np
# import scipy.interpolate as itp
import scipy.optimize as opt
import matplotlib.pyplot as plt
from Data.FSLib.IntegralsAndDerivatives import *
from Data.FSLib.fftTools import *

# df = pl.read_parquet("Nathaniel_IMU_Data/Orientation.pq")
df = pl.read_parquet("Nathaniel_IMU_Data/GyroCalibration.parquet", ).vstack(pl.read_parquet("Nathaniel_IMU_Data/Orientation.pq")).vstack(pl.read_parquet("Nathaniel_IMU_Data/Rectangle2x_Smoothed.pq"))

xA_uncorrected = "xA_uncorrected"
yA_uncorrected = "yA_uncorrected"
zA_uncorrected = "zA_uncorrected"
xA = "xA"
yA = "yA"
zA = "zA"
xG = "xG"
yG = "yG"
zG = "zG"
xB = "xBField"
yB = "yBField"
zB = "zBField"
time = "time"
imuT = "IMU_temp"

fig = plt.figure()
ax = fig.add_subplot(1,2,1,projection='3d')
ax.plot(df['xBField'], df['yBField'], df['zBField'], 'o', markersize=1, alpha=0.5)
ax.set_xlabel('X Field (Gauss)')
ax.set_ylabel('Y Field (Gauss)')
ax.set_zlabel('Z Field (Gauss)') #type: ignore
ax.set_title('Raw Magnetometer Data')
ax.set_xlim((-1, 1)) 
ax.set_ylim((-1, 1)) 
ax.set_zlim((-1, 1)) #type: ignore

ax = fig.add_subplot(1,2,2,projection='3d')
filtering = 0.9
ax.plot(low_pass_filter(df[xB].to_numpy(), filtering), low_pass_filter(df[yB].to_numpy(), filtering), low_pass_filter(df[zB].to_numpy(), filtering), 'o', markersize=1, alpha=0.5)
ax.set_xlabel('X Field (Gauss)')
ax.set_ylabel('Y Field (Gauss)')
ax.set_zlabel('Z Field (Gauss)') #type: ignore
ax.set_title(f'LPF Magnetometer Data: {filtering}')
ax.set_xlim((-1, 1)) 
ax.set_ylim((-1, 1)) 
ax.set_zlim((-1, 1)) #type: ignore

plt.show()

plt.plot(np.sqrt(df[xB].pow(2) + df[yB].pow(2) + df[zB].pow(2)))
plt.show()


# def magnetometer_callibration_function(x, s1, s2, s3, s4, s5, s6, b1, b2, b3):

'''
Make a plot showing the normal distribution of the magnetic field
Also determine the meadian and use that to determine the expected magnitude of the magnetic field.
'''



# Convolution for edge detection
plt.plot(np.convolve(df["vA_uncorrected"], np.array([-2, -2, -2, -1, 0, 1, 2, 2, 2]),'same')) 
plt.scatter(df.filter(pl.col("dvA").abs() < 100)[time],df.filter(pl.col("dvA").abs() < 100)["vA"], s=0.5)
plt.plot(df["vA_uncorrected"])
# plt.plot(low_pass_filter(df["vA"].to_numpy(),0.98))
plt.plot(df["dvA"])
plt.show()
df.insert_column(-1, pl.Series(np.convolve(df["vA_uncorrected"], np.array([-2, -2, -2, -1, 0, 1, 2, 2, 2]),'same')).alias("vAConvolve"))

#Cuts for milliG IMU
plt.plot(df["vAConvolve"])
cuts = df.filter(pl.col("vAConvolve").abs() > 20)[time] #Look for places where the edge detection is large (above 50)
plt.scatter(cuts, np.ones(cuts.shape[0]), s=0.5)
plt.show()

for i in range(0, cuts.shape[0] - 1):
#checks every region bounded by 2 cut locations. If the region is large enough, save it to "regions"
    if i == 0:
        regions = []
    if cuts[i+1] - cuts[i] > 500: #!!!!# 500 for personal IMU, 3 for car IMU
        regions = [(cuts[i], cuts[i+1], cuts[i+1] - cuts[i])] + regions
filtered_chunks = pl.DataFrame()
medians = pl.DataFrame()


def compile_chunks_and_graph (regions, filter_decision, low_pass_filter_portion): # List, bool, float [0,1]
    filtered_chunks = pl.DataFrame()
    medians = pl.DataFrame()
    for (start, stop, l) in regions:
    # For every region selected, grab the median and standard deviation
    # Filter out any values greater than 1 standard deviation from the median
    # Calculate the new standard deviation without outliers. If the data set varies too much (>1) drop that set
        chunk = df.filter((pl.col(time) >= start) & (pl.col(time) < stop))
        med = chunk["vA_uncorrected"].median()
        std = chunk["vA_uncorrected"].std()
        print(f"std: {std}")
        print(f"med = {med}, std = {std}")
        num_stds = 1 #1 for home IMU and FS IMU
        filtered_chunk = chunk.filter((pl.col("vA_uncorrected") <= med + num_stds*std) & (pl.col("vA_uncorrected") >= med - num_stds*std)) #type: ignore
        std = filtered_chunk["vA_uncorrected"].std()
        med = filtered_chunk["vA_uncorrected"].median()
        if std < 1: #1 for home IMU and FS IMU
            if filter_decision:
                # print(f"shape before is {filtered_chunk.shape}")
                array = low_pass_filter(filtered_chunk["vA_uncorrected"].to_numpy(), low_pass_filter_portion)
                # print(f"shape after is {array.shape}")
                # print(array)
                series = pl.Series(array).alias("vA_uncorrected")
                insertion_index = filtered_chunk.get_column_index("vA_uncorrected")
                filtered_chunk.drop_in_place("vA_uncorrected")
                filtered_chunk.insert_column(insertion_index, series)
                # print(filtered_chunk["vA"])
            plt.scatter(filtered_chunk[time], filtered_chunk["vA_uncorrected"], s=0.5) 
            filtered_chunks = pl.concat([filtered_chunks, filtered_chunk],how = 'vertical')
            # print("here")
            medians = pl.concat([medians, filtered_chunk.filter(pl.col("vA_uncorrected") == (filtered_chunk["vA_uncorrected"].median()))], how = 'vertical')
    # print("here")
    plt.show()
    return (filtered_chunks, medians)

def optimize_magnetometer_function(x, s1, s2, s3, s4, s5, s6, b1, b2, b3):
    b = np.array([b1, b2, b3]).T
    R = np.array([[s1, s2, s3],[s2, s4, s5],[s3, s5, s6]])
    return np.matmul(np.matmul(x-b, R), (x-b).T)

filtered_chunks, medians = compile_chunks_and_graph(regions, False, 0.9)
args = opt.curve_fit(optimize_magnetometer_function,filtered_chunks,1000*np.ones(filtered_chunks.shape[0]),[1,1,1,0.2,-0.2,-0.1,0.2,0.2,-0.15,0.1,-0.2,0.1])
