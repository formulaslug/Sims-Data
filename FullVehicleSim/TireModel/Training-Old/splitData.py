import polars as pl
import matplotlib.pyplot as plt

df = pl.read_csv("slipAngleData.dat", separator="	", skip_rows=1, skip_rows_after_header=1)

#df = df.filter((pl.col("SA") > 0.1) | (pl.col("SA") < -0.1))
df.write_csv('train-slipangle.csv')
#df = df.filter((pl.col("SR") >= 0))
#df.write_csv("torqueData/d1.csv")

threshold1 = 400 
threshold2 = 700

#segment1 = df.filter(pl.col("ET") <= threshold1)
#segment2 = df.filter((pl.col("ET") > threshold1) & (pl.col("ET") <= threshold2))
#segment3 = df.filter(pl.col("ET") > threshold2)

df.write_csv('train-slipangle.csv')
#segment2.write_csv('val.csv')
#segment3.write_csv('test.csv')
