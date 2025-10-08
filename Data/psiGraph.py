# WARNING: UNFINISHED

import matplotlib.axes
import polars as pl
import matplotlib.pyplot as plt

# df = pl.read_parquet("./Parquet/2024-12-02-Part1-100Hz.pq")
# df = pl.read_parquet("./Parquet/2024-12-02-Part2-100Hz.pq")
# "FS-2/Parquet/2024-12-02-Part1-100Hz.pq"
df = pl.read_parquet("FS-2/Parquet/2025-03-06-BrakingTests1.parquet").with_columns(pl.all().cast(pl.Float32, strict=False))

brakes = "Brakes"
steerbrakef = "TELEM_STEERBRAKE_BRAKEF"
steerbraker = "TELEM_STEERBRAKE_BRAKER"
sec = "Seconds"

time1 = 0 # 2038 
time2 = 99999 # 2082

brakesColUnchanged = df.select(brakes).slice(time1, time2).to_series()
brakesCol = (brakesColUnchanged * 1.5 - 5) * (2000 / 4)
brakesCol2 = ((brakesColUnchanged / 4095) * 1.5 - 0.5) * 2000 / (4.5 - 0.5)
brakesCol3 = ((brakesColUnchanged / 32767 * 5) - 0.5) * 2000

steerbrakefUnchanged = df.select(steerbrakef).slice(time1, time2).to_series()
steerbrakerUnchanged = df.select(steerbraker).slice(time1, time2).to_series()
steerbrakef1 = ((steerbrakefUnchanged / 32767 * 5) - 0.5) * 2000
steerbraker1 = ((steerbrakerUnchanged / 32767 * 5) - 0.5) * 2000

secondsCol = df.select(sec).slice(time1, time2).to_series()

# plt.plot(secondsCol, brakesColUnchanged)
# plt.plot(secondsCol, brakesCol3)
# plt.plot(secondsCol, brakesColUnchanged)

plt.plot(secondsCol, steerbrakef1)
plt.plot(secondsCol, steerbraker1)
plt.xlabel("time")
plt.ylabel("???")
plt.legend([steerbrakef, steerbraker])

plt.show()
