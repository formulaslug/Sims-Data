'''
File for functions that do full calibration routines and return the parameters.
Author: Nathaniel Platt
'''

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from fftTools import *
from scipy import optimize as opt

def compile_chunks_and_graph (regions, filter_decision, low_pass_filter_portion, df, time, m, verbose, plot, median_filter, median_filter_value): # List, bool, float [0,1]
        filtered_chunks = pl.DataFrame()
        medians = pl.DataFrame()
        for (start, stop, l) in regions:
        # For every region selected, grab the median and standard deviation
        # Filter out any values greater than 1 standard deviation from the median
        # Calculate the new standard deviation without outliers. If the data set varies too much (>1) drop that set
            chunk = df.filter((pl.col(time) >= start) & (pl.col(time) < stop))
            med = chunk[m].median()
            std = chunk[m].std()
            if std == None:
                print(chunk)
            if verbose:
                print(f"std: {std}")
                print(f"med = {med}, std = {std}")
            num_stds = 1 #1 for home IMU and FS IMU
            filtered_chunk = chunk.filter((pl.col(m) <= med + num_stds*std) & (pl.col(m) >= med - num_stds*std))
            std = filtered_chunk[m].std()
            med = filtered_chunk[m].median()
            if std < 1: #1 for home IMU and FS IMU
                if filter_decision:
                    # print(f"shape before is {filtered_chunk.shape}")
                    array = low_pass_filter(filtered_chunk[m].to_numpy(), low_pass_filter_portion)
                    # print(f"shape after is {array.shape}")
                    # print(array)
                    series = pl.Series(array).alias(m)
                    med = series.median()
                    if median_filter:
                        # print(med)
                        # print(filtered_chunk[m].min())
                        if med < median_filter_value:
                            continue
                    insertion_index = filtered_chunk.get_column_index(m)
                    filtered_chunk.drop_in_place(m)
                    filtered_chunk.insert_column(insertion_index, series)
                    # print(filtered_chunk["vA"])
                if plot:
                    plt.scatter(filtered_chunk[time], filtered_chunk[m], s=0.5) 
                filtered_chunks = pl.concat([filtered_chunks, filtered_chunk],how = 'vertical')
                # print("here")
                medians = pl.concat([medians, filtered_chunk.filter(pl.col(m) == (filtered_chunk[m].median()))], how = 'vertical')
        # print("here")
        if plot:
            plt.show()
        return (filtered_chunks, medians)


def vector_imu_calibrate(df, column_names, min_cut_size, cut_trigger_height, lpf=False, lpfVal=0.9, plot=True, verbose=False, starting_values=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], median_filter=False, median_filter_value=0):
    '''
    Generic function to offer calibration values for a specific type of filter for any 3axis sensor in an IMU.
    df: DataFrame with time and magnitude(of the 3 axis vector) columns, and columns for the 3 axes.
    '''
    time, m, x, y, z, out = column_names
    mCV = "mConvolve"
    df.insert_column(-1, pl.Series(np.convolve(df[m], np.array([-2, -2, -2, -1, 0, 1, 2, 2, 2]),'same')).alias(mCV))
    
    #Cuts for Gs (Car IMU)
    if plot:
        plt.plot(df[mCV])
        plt.show()
    cuts = df.filter(pl.col(mCV).abs() > cut_trigger_height)[time] #Look for places where the edge detection is large (above 50)
    if plot:
        plt.scatter(cuts, np.ones(cuts.shape[0]), s=0.5)
        plt.show()

    for i in range(0, cuts.shape[0] - 1):
    #checks every region bounded by 2 cut locations. If the region is large enough, save it to "regions"
        if i == 0:
            regions = []
        if cuts[i+1] - cuts[i] > min_cut_size: #!!!!# 500 for personal IMU, 3 for car IMU
            regions = [(cuts[i], cuts[i+1], cuts[i+1] - cuts[i])] + regions
    
    filtered_chunks = pl.DataFrame()
    medians = pl.DataFrame()

    filtered_chunks, medians = compile_chunks_and_graph(regions, True, 0.95, df=df,time=time, m=m, verbose=verbose, plot=plot, median_filter=median_filter, median_filter_value=median_filter_value)
    # if plot:
    #     print(filtered_chunks)
    #     print(filtered_chunks[m].min())
    #     plt.scatter(filtered_chunks[time], filtered_chunks[m], label="Filtered Data", color='lightcoral', s=0.5)
    #     plt.show()

    def lsq_fun (v, Sx, Sy, Sz, a1, a2, a3, a4, a5, a6, b1, b2, b3):
        scalar_matrix = np.array([[Sx, 0, 0],
                                [0, Sy, 0],
                                [0, 0, Sz]])
        off_axis_matrix = np.array([[1, a1, a2],
                                    [a3, 1, a4],
                                    [a5, a6, 1]])
        first_matrix = np.matmul(scalar_matrix, off_axis_matrix)
        bias1 = np.ones((1,v[x,y,z].shape[0]))*b1
        bias2 = np.ones((1,v[x,y,z].shape[0]))*b2
        bias3 = np.ones((1,v[x,y,z].shape[0]))*b3
        bias_matrix = np.concatenate([bias1,bias2,bias3], axis=0)
        matrix = v[x,y,z].to_numpy().T
        biased_matrix = matrix-bias_matrix
        vectors = np.matmul(first_matrix,biased_matrix)
        mag = np.sqrt(vectors[0,:]**2 + vectors[1,:]**2+vectors[2,:]**2)
        # print(np.matmul(np.matmul(np.array([[Sx, 0, 0],[0, Sy, 0],[0,0,Sz]]),np.array([[1, a1, a2],[a3, 1, a4],[a5, a6, 1]])),np.array([[y[0,0]-b1],[y[1,0]-b2],[y[2,0]-b3]])))
        # print(f"error = {error}")
        return mag
    
    args = opt.curve_fit(lsq_fun,filtered_chunks,filtered_chunks[out],starting_values)
    if plot:
        plt.scatter(filtered_chunks[time], filtered_chunks[m], s=0.5, label="Filtered Data")
        plt.scatter(filtered_chunks[time], lsq_fun(filtered_chunks, *args[0]), color='goldenrod', s=0.5, label="Fitted Curve")
        plt.legend()
        plt.show()
    return args[0]
