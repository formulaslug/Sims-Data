import polars as pl

import matplotlib.pyplot as plt

da = pl.read_csv("/Users/daniel/Documents/Coding/FormulaSlug/sim-test/VehicleDynamics/factorScalingAvon/rawData/B1706raw27.dat", separator="	", skip_rows=1, skip_rows_after_header=1)
db = pl.read_csv("/Users/daniel/Documents/Coding/FormulaSlug/sim-test/VehicleDynamics/factorScalingAvon/rawData/B1706raw30.dat", separator="	", skip_rows=1, skip_rows_after_header=1)
dc = pl.read_csv("/Users/daniel/Documents/Coding/FormulaSlug/sim-test/VehicleDynamics/factorScalingAvon/rawData/B1706raw33.dat", separator="	", skip_rows=1, skip_rows_after_header=1)
dd = pl.read_csv("/Users/daniel/Documents/Coding/FormulaSlug/sim-test/VehicleDynamics/factorScalingAvon/rawData/B1706raw36.dat", separator="	", skip_rows=1, skip_rows_after_header=1)
de = pl.read_csv("/Users/daniel/Documents/Coding/FormulaSlug/sim-test/VehicleDynamics/factorScalingAvon/rawData/B1706raw39.dat", separator="	", skip_rows=1, skip_rows_after_header=1)
df = pl.read_csv("/Users/daniel/Documents/Coding/FormulaSlug/sim-test/VehicleDynamics/factorScalingAvon/rawData/B1706raw42.dat", separator="	", skip_rows=1, skip_rows_after_header=1)

da = da.filter((pl.col("SA") < 0.025) & (pl.col("SA") > -0.025))
da = da.with_columns((pl.col("FX") * 0.505).alias("T"))
da.write_csv("torqueData/d1.csv")

db= db.filter((pl.col("SA") < 0.025) & (pl.col("SA") > -0.025))
db = db.with_columns((pl.col("FX") * 0.505).alias("T"))
db.write_csv("torqueData/d2.csv")

dc = dc.filter((pl.col("SA") < 0.025) & (pl.col("SA") > -0.025))
dc = dc.with_columns((pl.col("FX") * 0.513).alias("T"))
dc.write_csv("torqueData/d3.csv")

dd = dd.filter((pl.col("SA") < 0.025) & (pl.col("SA") > -0.025))
dd = dd.with_columns((pl.col("FX") * 0.513).alias("T"))
dd.write_csv("torqueData/d4.csv")

de = de.filter((pl.col("SA") < 0.025) & (pl.col("SA") > -0.025))
de = de.with_columns((pl.col("FX") * 0.507).alias("T"))
de.write_csv("torqueData/d5.csv")

df = df.filter((pl.col("SA") < 0.025) & (pl.col("SA") > -0.025))
df = df.with_columns((pl.col("FX") * 0.507).alias("T"))
df.write_csv("torqueData/d6.csv")
