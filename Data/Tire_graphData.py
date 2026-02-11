## This file is for visualizing the tire data we bought fall 2024.
## All code should still be functional - Nathaniel 1/11/25

import polars as pl
from matplotlib import pyplot as plt

# folder = r""
# files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
# files = [folder+"\\"+f for f in files if f[-4:] == ".dat"]
# files.sort(key = lambda f:int(re.match(r".+?([0-9]+)\.dat", f).group(1)))

files = [r"TireData\B1965raw2.dat"]

timeKey = r"ET"
size = 1000

# see = df.columns
see = ["FY", "FX", "P"]

t1 = 0
t2 = float("inf")

def readfile(filename):
    if filename[-3:] == ".pq":
        df = pl.read_parquet(filename)
    elif filename[-4:] == ".csv":
        df = pl.read_csv(filename, infer_schema_length=10000, ignore_errors=True)
    elif filename[-4:] == ".dat":
        with open(filename, "r") as file:
            text = file.readlines()
        text.pop(0)
        text.pop(1)
        text = [row.strip() for row in text]
        rows = [row.split("\t") for row in text]
        header = rows[0]
        rows = [[float(i) for i in row] for row in rows[1:]]
        # frame = {h:[for row in text] for n, h in enumerate(rows[0])}
        df = pl.DataFrame(rows, schema=header, orient="row")
    else:
        print("can't read")
        return None

    return df
    
def getTime(df):
    print(df.columns)
    t = df[timeKey]
    print(t, len(t))
    return t

def plot(x, y, lab = ""):
    skip = max(1, int(len(x)/size))
    plt.plot(x[::skip], y[::skip], linestyle="", marker="o")
    plt.xlabel(lab)
    plt.show()

def getIndexes(t1, t2, t):
    t2 = min(t[-1], t2)
    i1 = int((t1/t[-1])*len(t))
    i2 = int((t2/t[-1])*len(t))
    return i1, i2

for f in files:
    df = readfile(f)
    if isinstance(df, type(None)):
        continue
    # df.write_csv(f.replace(folder, "TireDataCSV").replace(".dat",".csv"))
    t = getTime(df)
    i1, i2 = getIndexes(t1, t2, t)
    for c in see:
        plot(t[i1:i2], df[c][i1:i2], f + " " + c)

