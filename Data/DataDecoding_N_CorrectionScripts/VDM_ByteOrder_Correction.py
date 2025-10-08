import polars as pl
import os
import struct

# get all filenames in a given directory
directory = 'FS-3/08172025raw/08172025parquetNew copy'  # change to your target directory

# filenames = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.startswith('08102025')]
filenames = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.startswith('08172025')]
# filenames = ["08172025_27.parquet"]

# df = pl.read_parquet("FS-3/08172025raw/08172025parquet copy/08172025_20.parquet")
# df = pl.read_parquet("FS-3/08102025CoupleMoreFastWarmupsPostMechFixes2.parquet")

# df = pl.read_parquet("FS-3/08172025Endurance2.parquet")

# [i for i in df.columns if i[0:3] == "VDM"]

# df.columns
# plt.plot(df["VDM_GPS_VALID1"])



# def byteSwap4 (x): ## This actually worked lol but the other one is better
#     return x//(2**24) + ((x%(2**24)) // (2**16))*(2**8) + ((x%(2**16)) // (2**8)) * (2**16) + ((x%(2**8))) * (2**24)

# byteSwap4(1650293)

def byteSwap (num, length, signed, typeString):
    # print(f"num: {num}, length: {length}, signed: {signed}, typeString: {typeString}")
    byteset = num.to_bytes(length = length, byteorder = 'little', signed = signed)
    # bytesetReordered = byteset[::-1]
    if typeString == "i":
        return int.from_bytes(byteset, byteorder = 'big', signed = signed)
    elif typeString == "f":
        return struct.unpack('>f', byteset)[0]

def byteSwapFun (length, signed, typeString):
    def swap(num):
        return byteSwap(num, length, signed, typeString)
    return swap

# byteSwap(1650293, 4, False)

# df["VDM_GPS_Latitude"].map_elements(byteSwapFun (4, True, "f"))
# df["VDM_GPS_Latitude"]
# plt.plot()
# df["VDM_GPS_SPEED"].dtype
# # plt.scatter(np.arange(df.shape[0]),df["VDM_Z_AXIS_ACCELERATION"]%1, s=0.1)

# dfAdjusted = df.with_columns(
#         [
#             pl.col("VDM_GPS_Latitude").map_elements(byteSwapFun (4, True, "f")),
#             pl.col("VDM_GPS_Longitude").map_elements(byteSwapFun (4, True, "f")),
#             pl.col("VDM_GPS_SPEED").cast(pl.UInt16).map_elements(byteSwapFun (2, False, "i")),
#             pl.col("VDM_GPS_ALTITUDE").map_elements(byteSwapFun (2, True, "i")),
#             pl.col("VDM_GPS_TRUE_COURSE").cast(pl.UInt16).map_elements(byteSwapFun (2, False, "i")),
#             pl.col("VDM_X_AXIS_ACCELERATION").cast(pl.Int16).map_elements(byteSwapFun (2, True, "i")),
#             pl.col("VDM_Y_AXIS_ACCELERATION").cast(pl.Int16).map_elements(byteSwapFun (2, True, "i")),
#             pl.col("VDM_Z_AXIS_ACCELERATION").cast(pl.Int16).map_elements(byteSwapFun (2, True, "i")),
#             pl.col("VDM_X_AXIS_YAW_RATE").cast(pl.Int16).map_elements(byteSwapFun (2, True, "i")),
#             pl.col("VDM_Y_AXIS_YAW_RATE").cast(pl.Int16).map_elements(byteSwapFun (2, True, "i")),
#             pl.col("VDM_Z_AXIS_YAW_RATE").cast(pl.Int16).map_elements(byteSwapFun (2, True, "i"))
#         ]
#     )


# dfValid = dfAdjusted.filter(pl.col("VDM_GPS_VALID1") == 1)
# # type(dfValid["VDM_GPS_Latitude"][0])
# # dfValid["VDM_GPS_Latitude"][1]

# plt.plot(dfValid["VDM_GPS_Longitude"].map_elements(byteSwapFun(4, True, "f")),dfValid["VDM_GPS_Latitude"].map_elements(byteSwapFun(4, True, "f")))
# plt.plot(dfValid["VDM_GPS_Latitude"], dfValid["VDM_GPS_Longitude"])
# plt.plot(dfValid["VDM_GPS_SPEED"]/100)
# plt.plot(dfValid["VDM_GPS_ALTITUDE"])
# plt.plot(dfValid["VDM_GPS_TRUE_COURSE"]/100)
# plt.plot(dfValid["VDM_X_AXIS_ACCELERATION"]*0.000244141)
# plt.plot(dfValid["VDM_Y_AXIS_ACCELERATION"]*0.000244141)
# plt.plot(dfValid["VDM_Z_AXIS_ACCELERATION"]*0.000244141)
# plt.plot(dfValid["VDM_X_AXIS_YAW_RATE"]*0.015258789)
# plt.plot(dfValid["VDM_Y_AXIS_YAW_RATE"]*0.015258789)
# plt.plot(dfValid["VDM_Z_AXIS_YAW_RATE"]*0.015258789)
# plt.show()

for filename in filenames:
    path = os.path.join(directory, filename)
    df = pl.read_parquet(path)
    dfNew = df.with_columns(
        [
            pl.col("VDM_GPS_Latitude").map_elements(byteSwapFun (4, True, "f")),
            pl.col("VDM_GPS_Longitude").map_elements(byteSwapFun (4, True, "f")),
            pl.col("VDM_GPS_SPEED").cast(pl.UInt16).map_elements(byteSwapFun (2, False, "i")),
            pl.col("VDM_GPS_ALTITUDE").map_elements(byteSwapFun (2, True, "i")),
            pl.col("VDM_GPS_TRUE_COURSE").cast(pl.UInt16).map_elements(byteSwapFun (2, False, "i")),
            pl.col("VDM_X_AXIS_ACCELERATION").cast(pl.Int16).map_elements(byteSwapFun (2, True, "i")),
            pl.col("VDM_Y_AXIS_ACCELERATION").cast(pl.Int16).map_elements(byteSwapFun (2, True, "i")),
            pl.col("VDM_Z_AXIS_ACCELERATION").cast(pl.Int16).map_elements(byteSwapFun (2, True, "i")),
            pl.col("VDM_X_AXIS_YAW_RATE").cast(pl.Int16).map_elements(byteSwapFun (2, True, "i")),
            pl.col("VDM_Y_AXIS_YAW_RATE").cast(pl.Int16).map_elements(byteSwapFun (2, True, "i")),
            pl.col("VDM_Z_AXIS_YAW_RATE").cast(pl.Int16).map_elements(byteSwapFun (2, True, "i"))
        ]
    )
    dfNew.write_parquet(path)
