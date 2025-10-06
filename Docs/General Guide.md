# Guide to fs-data

## Table of Contents

1. [Basic Read Script](#brs)
1. [File Types](#filetypes)
1. [Folders](#folders)

<h2 id="brs"> Basic Reading Script </h2>

```python
import polars as pl # Polars for parsing data
import numpy as np # Numpy for array operations + a bunch of useful stuff
import matplotlib.pyplot as plt # Matplotlib's Pyplot for plotting

df = pl.read_parquet("FS-3/08102025/08102025Endurance1_FirstHalf.parquet") # Import Parquet file! This is an FS-3 run from Aug 10, 2025 at Bluemax. It is specifically the first half of an endurance run (14 laps).

print(df.columns) # To get a list of all the columns
print([i for i in df.columns if i.startswith("VDM")]) # To get a list of all the columns that begin with "VDM"

# It is useful to set these up so you don't have to type them out all the time. Variable names also autocomplete unlike strings you're trying to type out.
lat = "VDM_GPS_Latitude" 
long = "VDM_GPS_Longitude"

dfFiltered = df.filter(pl.col(lat) != 0).filter(pl.col(long) != 0) # remove 0 values from incorrect GPS readings
plt.plot(dfFiltered[lat], dfFiltered[long]) # Plots a map of the track you drove! You can throw on a third dimension with color to see aspects of driving that are specific to different parts of the track.
plt.show()

plt.scatter(dfFiltered[lat], dfFiltered[long]) # Scatter plot to see how big the gaps between measurements are and where points off the track may be located.
plt.show()

```

<h2 id ="filetypes"> File Types </h2>

1. ```.csv```
    - Refers to comma spaced values. They are an ascii representation of data so they tend to be very large. It is nice that they are human readable. First row is headers (column names) and the following rows are the values in ascii. You can read this with polars by doing ```pl.read_csv(<path>)```
1. ```.tsv``` 
    - Related to CSV, TSV stands for tab spaced values. You can read this with ```pl.read_csv(<path>, separator='\t')```. Last I checked doing ```tab``` and creating 4 spaces in python will not work for interpreting tsvs. 
1. ```.parquet```
    - This is our standard method of storing data. It is a compressed file type that contains a dataFrame from the library Polars.
1. ```.fsdaq``` 
    - This is a custom binary format developed by our team for the competition in 2025. It is a custom logging format you can read more about in [2025 Custom Binary](<2025 Custom Binary.md>). You can look at the implementation of the decoder in [fsdaq_decoder.py](../DataDecoding_N_CorrectionScripts/fsdaq_decoder.py)
1. ```.tdms```  (Sill don't know what this means.)
    - This is a standard binary format used by FSAE EV for logging e-meter data and is the format by which we receive it from them. There is a python library ```nptdms``` you can get to read it. Look at implementation in [emeterDecode.py](../emeterDecode.py)
1. ```.daq``` 
    - You may find some of these from FS-2. They are a file format used by the company AEM for their compressed storage. We used the AEM Carbon 7 on FS-2 to do logging. To read these, use the program ```AEM data``` which can be downloaded for windows. Load a file, then export to csv and make sure you set the correct frequency (typically 100Hz for us as of 8/28/25).

<h2 id ="folders"> Folders </h2>

1. ```FS-2/``` and ```FS-3/``` contain data from these two cars. They have subfolders with more specific bits of data.
1. ```DataDecoding&CorrectionScipts/``` contains
1. AEMData Templates contains stuff to flash the dash with the latest program (Nathaniel: "I think" Date: 2025-01-20 )
1. Nathaniel_IMU_Data is from Nathaniel's personal IMU to log data that can be used to learn how to characterize and model the error in IMUs.
1. Parquet. This folder is for the Daq files after being converted into CSVs and then Parquets. CSVs are too large to store on their own so we store them as Parquets which are compressed but still accessible by pandas and polars.
1. Temp: Folder for temporarily holding CSVs before making them into parquets
1. Tire Data: Data bought from the TTC (Tire Testing Consortiom) in fall of 2024 for fs-2's tires. Missing lots of data and not as useful as we hoped.
1. TireDataCSV is the tire data but in CSVs 
1. Remaining files are all python or readme etc. for different data analysis purposes.