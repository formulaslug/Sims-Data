import polars as pl
import os
# import matplotlib.pyplot as plt
import cantools.database as db

# get all filenames in a given directory
directory = 'FS-3/08172025raw/08172025parquetNew copy'  # change to your target directory
filenames = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.startswith('08172025')]
# filenames = ["08172025_something20.parquet"]

dbc = db.load_file('../fs-3/CANbus.dbc')

for file in filenames:
    df = pl.read_parquet(os.path.join(directory, file))
    for message in dbc.messages: # type: ignore
        for signal in message.signals:
            if signal.name in df.columns:
                df = df.with_columns(
                    (pl.col(signal.name) * signal.scale) + signal.offset ## Use this line if you are applying scale and offset to a fresh batch of data
                    # (pl.col(signal.name) - signal.offset) / signal.scale  ## Use this line to UNDO a scale and offset
                )
    df.write_parquet(os.path.join(directory, f"{file}"))

# dfCorrected = pl.read_parquet(os.path.join(directory, f"corrected_08102025Endurance1_FirstHalf.parquet"))
# dfFiltered = dfCorrected.filter(pl.col("VDM_GPS_VALID1") == 1)

# plt.plot(dfFiltered["VDM_GPS_Latitude"], dfFiltered["VDM_GPS_Longitude"], label='GPS Path')
# plt.show()
