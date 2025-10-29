'''
Not a good idea in the end. Do not use.
'''


import polars as pl
import os

filename = '9-6-data'

#mm:ss.ddd
timeSplits = '''
00:47.351
04:45.351
08:19.851
13:28.551
17:06.851
19:47.801
24:04.601
27:29.186
30:11.367
32:20.501
34:09.301
35:38.351
37:45.401
39:35.601
40:50.201
42:27.751
43:50.651
47:20.301
51:24.550
55:59.201
58:32.196
1:01:50.151
1:13:30.501
1:16:07.801
1:18:49.503
1:21:52.851
1:25:03.051
1:27:57.451
1:31:44.051
1:33:03.501
1:39:36.448
1:47:22.551
1:48:55.351
1:52:17.301
'''

df = pl.read_csv("CSV\\%s.csv"%(filename),infer_schema_length=10000,ignore_errors=True)

# Specify the directory name
directory_name = "Parquet\\"+filename

# Create the directory
try:
    os.mkdir(directory_name)
    print(f"Directory '{directory_name}' created successfully.")
except FileExistsError:
    print(f"Directory '{directory_name}' already exists.")
except PermissionError:
    print(f"Permission denied: Unable to create '{directory_name}'.")
except Exception as e:
    print(f"An error occurred: {e}")

index = 0
print(df[":Time"][-1])
for split in timeSplits.split("\n") + [df[":Time"][-1]]:
    if isinstance(split, str):
        split = split.strip()
        if len(split) == 0:
            continue
        split = split.split(":")
        t = 0
        for n, i in enumerate(split):
            t += float(i) * (60**(len(split)-n-1))
    else:
        t = split
    print(t)

    frame = df.filter(pl.col(":Time") < t)
    df = df.filter(pl.col(":Time") >= t)

    frame.write_parquet(directory_name + "\\%s-%s.pq"%(filename, index))

    prev = t
    index += 1