# Documentation https://docs.google.com/document/d/1oGsGDnY0DEKWpE3S6481A9yZ0F9qUEwWkSXJwTSz4E4/edit?pli=1&tab=t.93l7bg7vvy8c
import polars as pl

import matplotlib.pyplot as plt


train = pl.read_csv("inputData.dat", separator="	", skip_rows=1, skip_rows_after_header=1) #pl.read_csv("train-slipangle.csv", separator=",")") < 10))

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(train["SR"], train["SA"], train["NFY"], c='red', marker='o')

plt.show()


#print("WRITE")
train.write_csv('srData.csv') 