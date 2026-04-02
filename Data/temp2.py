import polars as pl
from Data.FSLib.AnalysisFunctions import *
from scipy.stats import linearregress

rpm = "SME_TRQSPD_Speed"
lat = "VDM_GPS_Latitude"
long = "VDM_GPS_Longitude"
xA = "VDM_X_AXIS_ACCELERATION"
yA = "VDM_Y_AXIS_ACCELERATION"
speed = "VDM_GPS_SPEED" ## mph

df1 = readValid("C:/Projects/FormulaSlug/fs-data/FS-3/08102025/08102025Endurance1_FirstHalf.parquet") ## 276,1075
df1 = df1.insert_column(0, simpleTimeCol(df1))
df1 = df1.filter(pl.col("Time") >= 276).filter(pl.col("Time") <= 1075)
df1 = df1.with_columns(
    pl.col("Time") - pl.col("Time").min()
)
df1 = df1.insert_column(0, lapSegmentation(df1))

df2 = readValid("C:/Projects/FormulaSlug/fs-data/FS-3/08102025/08102025Endurance1_SecondHalf.parquet") ## 77, 862
df2 = df2.insert_column(0, simpleTimeCol(df2))
df2 = df2.filter(pl.col("Time") >= 77).filter(pl.col("Time") <= 862)
df2 = df2.with_columns(
    pl.col("Time") - pl.col("Time").min()
)
df2 = df2.insert_column(0, lapSegmentation(df2))
df2 = df2.with_columns(
    pl.col("Lap") + df1["Lap"].max() + 1
)

df3 = readValid("C:/Projects/FormulaSlug/fs-data/FS-3/08172025/08172025_20_Endurance1P1.parquet") ## 72, 1120 -- Drop laps 8 and 9
df3 = df3.insert_column(0, simpleTimeCol(df3))
df3 = df3.filter(pl.col("Time") >= 72).filter(pl.col("Time") <= 1120)
df3 = df3.with_columns(
    pl.col("Time") - pl.col("Time").min()
)
df3 = df3.insert_column(0, lapSegmentation(df3))
df3_1 = df3.filter(pl.col("Lap") < 8)
df3_2 = df3.filter(pl.col("Lap") > 9).with_columns(
    pl.col("Lap") - 2
)
df3 = df3_1.vstack(df3_2)
df3 = df3.with_columns(
    pl.col("Lap") + df2["Lap"].max() + 1
)

df1 = df1.select(df3.columns)
df2 = df2.select(df3.columns)

df1f = df1._cast_all_from_to(df1, frozenset((pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32)), pl.Float64)
df2f = df2._cast_all_from_to(df2, frozenset((pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32)), pl.Float64)
df3f = df3._cast_all_from_to(df3, frozenset((pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32)), pl.Float64)

df = df1f.vstack(df2f).vstack(df3f)
df = df.drop("Time").insert_column(0, simpleTimeCol(df))

fig = plt.figure(layout="constrained")
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
ax1.plot(df["Time"], df[rpm], label="RPM")
ax2.plot(df["Time"], df["Lap"], label="Lap")
fig.legend()
plt.show()

for i in df["Lap"].unique():
    plt.plot(df.filter(pl.col("Lap") == i)[lat], df.filter(pl.col("Lap") == i)[long], label=f"Lap {i}")
plt.legend()
plt.show()

# df.drop("Time").write_parquet("C:/Projects/FormulaSlug/fs-data/FS-3/PreparedData/CombinedEndurance_0810_0817_2025.parquet")
# df["Lap"].max()

df = pl.read_parquet("C:/Projects/FormulaSlug/fs-data/FS-3/PreparedData/CombinedEndurance_0810_0817_2025.parquet")
df = df.with_columns(
        (((df["ETC_STATUS_BRAKE_SENSE_VOLTAGE"]/1000)-0.33)/2.64*2000).alias("Brake_Pedal_Pressure_PSI"),
        (df[xA]*9.81*df[speed]*0.44704*300/1000).alias("Braking_Power_kW")
        )

fig = plt.figure(layout="constrained")
ax1 = fig.add_subplot(111)
ax1.scatter(df[long], df[lat], c=df[xA]*9.81*df[speed]*0.44704*300*-1/1000, cmap='viridis', s=2, alpha=0.3, label="Brake Power (kW)") ## convert mph to m/s by multiplying by 0.44704
ax1.set_aspect('equal', adjustable='datalim')
plt.colorbar(ax1.collections[0], ax=ax1, label="Braking Power (kW)")
ax1.set_title("Track Map Colored by Braking Power (kW)")
fig.show()

dfBraking = df.filter(pl.col("Braking_Power_kW") > 20).filter(pl.col("Brake_Pedal_Pressure_PSI") > 100)

fig2 = plt.figure(layout="constrained")
ax2 = fig2.add_subplot(111)
ax2.scatter(dfBraking["Brake_Pedal_Pressure_PSI"], dfBraking["Braking_Power_kW"])
ax2.set_title("Brake Pedal Pressure vs Braking Power (kW)")
ax2.set_xlabel("Brake Pedal Pressure (PSI)")
ax2.set_ylabel("Braking Power (kW)")
fig2.show()