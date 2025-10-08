## Template code for converting a CSV to a parquet.
## Made and commented by Nathaniel Platt

import polars as pl

df = pl.read_csv("Nathaniel_IMU_Data/GyroTest2.csv",infer_schema_length=10000,ignore_errors=True).with_columns(pl.all().cast(pl.Float32, strict=False))
df.write_parquet("Nathaniel_IMU_Data/GyroTest2.parquet")






df = pl.read_csv("Temp/2025-03-05.csv",infer_schema_length=10000,ignore_errors=True) # Imports csv, infers column types with the first 10000 rows. Generally better to just set to floats. S

df = pl.read_parquet("Parquet/Run7_060924.pq") # Reading a parquet into a dataframe
df["GLV_"]

df = df.drop(["GPSi_Altitude:Sensor",":Time",":Lap",":LapTime","GPSi_Altitude","GPSi_CNoAverage","GPSi_CNoMax","GPSi_Course","GPSi_DayUTC","GPSi_HoursUTC","GPSi_Latitude","GPSi_Longitude","GPSi_MinutesUTC","GPSi_MonthUTC","GPSi_NumHighCNo","GPSi_SatelliteCount","GPSi_SatellitesInUse","GPSi_SecondsUTC","GPSi_Speed","GPSi_YearUTC"])
df
df.columns
df.filter(pl.col("Seconds") == 6982.049)
df[":Time"].max()

df.select()

timeCut = 6420

frame = df.filter(pl.col("Seconds") < timeCut)
df = df.filter(pl.col("Seconds") >= timeCut)
frame.write_parquet("Parquet/Run13_060924.pq")
df

df
df.columns



df = pl.read_csv("Temp/91124.csv",infer_schema_length=10000,ignore_errors=True)
df.write_parquet("Temp/91124.pq")