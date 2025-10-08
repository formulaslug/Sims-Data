import polars as pl
import cantools.database as db
import numpy as np
import struct

def readFSDAQ (filepath, lightlyVerbose=False, verbose=False):
    '''
    Decodes the fsdaq file and returns it as a parquet
    '''
    v = verbose
    v1 = lightlyVerbose
    types_dict = {
        "i3": np.int8,
        "i4": np.int16,
        "i5": np.int32,
        "i6": np.int64,
        "u3": np.uint8,
        "u4": np.uint16,
        "u5": np.uint32,
        "u6": np.uint64,
        "f4": np.float16,
        "f5": np.float32,
        "f6": np.float64,
        "b0": bool
    }

    np_polars_typeDict = {
        np.int8 : pl.Int8,
        np.int16 : pl.Int16,
        np.int32 : pl.Int32,
        np.int64 : pl.Int64,
        np.uint8 : pl.UInt8,
        np.uint16 : pl.UInt16,
        np.uint32 : pl.UInt32,
        np.uint64 : pl.UInt64,
        np.float16 : pl.Float32,
        np.float32 : pl.Float32,
        np.float64 : pl.Float64,
        bool : pl.Boolean
    }

    with open(filepath, "rb" ) as f:
        data = f.read()
        header = ascii(data[:8])
        m = np.frombuffer(data[8:12], dtype=np.uint32, count=1)[0]
        n = np.frombuffer(data[12:16], dtype=np.uint32, count=1)[0]
        pos = 16
        colTitles = []
        print(f"Header: {header}, m: {m}, n: {n}")
        for i in range(m):
            length = data[pos]
            pos += 1
            title = data[pos:pos+length].decode('ascii')
            colTitles.append(title)
            pos += length
        colTypes = []
        for i in range(m):
            colType = data[pos:pos+2].decode('ascii')
            colTypes.append((types_dict[colType], int((2**int(colType[1]))/8*n)))
            pos += 2
        data_left = len(data[pos:])
        len_of_frame = sum([x[1] for x in colTypes])
        chunks = np.floor(data_left / len_of_frame)
        misc_bytes = (data_left % len_of_frame) #Used to be -8 for footer but ignoring footer
        print(f"chunks left: {chunks}")
        print(f"misc bytes left: {misc_bytes}")
        frames = []
        if v1:
            for x in zip(colTitles, colTypes):
                print(f"title = {x[0]}, type = {x[1]}")
        for i in range(int(chunks)):
            frame_pieces = []
            for i in range(m):
                col_byte_len = colTypes[i][1]
                col_type = colTypes[i][0]
                # if(len(data) < (col_byte_len)):
                #     df.write_csv(args.output_file)
                #     exit(0)
                if col_type != bool:
                    if(v or v1):
                        print(f"n = {n}; col_byte_len = {col_byte_len}; col_type = {col_type}")
                    col_bit = np.frombuffer(data[pos:pos+col_byte_len], dtype=col_type, count=n)
                else:
                    # print(f"n = {n}; int(n/8) = {int(n/8)}")
                    if(v or v1):
                        print(f"type = {col_type}")
                    col_bit = np.frombuffer(data[pos:pos+col_byte_len], dtype=np.uint8, count=int(n/8))
                    col_bit1 = np.unpackbits(col_bit)
                    col_bit = [True if x == 1 else False for x in col_bit1]
                if(v):
                    print(f"col_bit: {col_bit}")
                
                frame_piece_series = pl.Series(colTitles[i], col_bit, dtype=np_polars_typeDict[col_type])
                frame_pieces.append(frame_piece_series)
                pos += col_byte_len
            frames.append(pl.DataFrame(frame_pieces))
        df = pl.concat(frames, how="vertical")
        # print(df)
        if v: 
            print(f"Stuff Left: {ascii(data[pos:])}")
        return df
    
def byteSwap (num, length, signed, typeString):
    # print(f"num: {num}, length: {length}, signed: {signed}, typeString: {typeString}")
    try:
        byteset = num.to_bytes(length = length, byteorder = 'little', signed = signed) if typeString == "i" else struct.pack("<f", num)
    except:
        print(f"Failed on type {typeString}, signed = {signed}, length = {length}, num = {num}")
    # bytesetReordered = byteset[::-1]
    if typeString == "i":
        return int.from_bytes(byteset, byteorder = 'big', signed = signed)
    elif typeString == "f":
        return struct.unpack('>f', byteset)[0]

def byteSwapFun (length, signed, typeString):
    def swap(num):
        return byteSwap(num, length, signed, typeString)
    return swap
    
def byteOrderCorrection (df):
    '''
    Applies the correction that swaps the byte order for the incorrectly labled Motorola endianness on the VDM messages with multiple bytes.
    '''
    return df.with_columns(
        [
            pl.col("VDM_GPS_Latitude").map_elements(byteSwapFun (4, True, "f"), return_dtype=pl.Float32),
            pl.col("VDM_GPS_Longitude").map_elements(byteSwapFun (4, True, "f"), return_dtype=pl.Float32),
            pl.col("VDM_GPS_SPEED").map_elements(byteSwapFun (2, False, "i"), return_dtype=pl.UInt16),
            pl.col("VDM_GPS_ALTITUDE").map_elements(byteSwapFun (2, True, "i"), return_dtype=pl.Int16),
            pl.col("VDM_GPS_TRUE_COURSE").map_elements(byteSwapFun (2, False, "i"), return_dtype=pl.UInt16),
            pl.col("VDM_X_AXIS_ACCELERATION").map_elements(byteSwapFun (2, True, "i"), return_dtype=pl.Int16),
            pl.col("VDM_Y_AXIS_ACCELERATION").map_elements(byteSwapFun (2, True, "i"), return_dtype=pl.Int16),
            pl.col("VDM_Z_AXIS_ACCELERATION").map_elements(byteSwapFun (2, True, "i"), return_dtype=pl.Int16),
            pl.col("VDM_X_AXIS_YAW_RATE").map_elements(byteSwapFun (2, True, "i"), return_dtype=pl.Int16),
            pl.col("VDM_Y_AXIS_YAW_RATE").map_elements(byteSwapFun (2, True, "i"), return_dtype=pl.Int16),
            pl.col("VDM_Z_AXIS_YAW_RATE").map_elements(byteSwapFun (2, True, "i"), return_dtype=pl.Int16)
        ]
    )

def applyDBC_ScaleAndOffset (dbc, df):
    '''
    Using DBC, applies the missing scaling and offsets that some fsdaq files require.
    '''
    for message in dbc.messages: # type: ignore
        for signal in message.signals:
            if signal.name in df.columns:
                df = df.with_columns(
                    (pl.col(signal.name) * signal.scale) + signal.offset ## Use this line if you are applying scale and offset to a fresh batch of data
                    # (pl.col(signal.name) - signal.offset) / signal.scale  ## Use this line to UNDO a scale and offset
                )
    return df


def readCorrectedFSDAQ (fsdaqFilePath, dbcFilePath, verbose=False):
    '''
    This function does work but seems to take a decent bit longer than I remember. Should investigate making it faster.
    '''
    df = readFSDAQ(fsdaqFilePath, verbose)
    df1 = byteOrderCorrection(df)
    dbc = db.load_file(dbcFilePath)
    df2 = applyDBC_ScaleAndOffset(dbc, df1)
    return df2
