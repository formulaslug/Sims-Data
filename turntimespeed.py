## Problem right now is that the laps are not being recognized as valid, thus cant determine turn speed
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

FILE = "08102025/CombinedEndurance_0810_0817_2025.parquet"
df = pl.read_parquet(FILE)

print(f"Raw rows in file: {df.height}")


def timeCol(df):
    seconds = df["VDM_UTC_TIME_SECONDS"].to_numpy()
    t = np.zeros(len(seconds))
    start = seconds[0]
    mask = seconds == start
    n = mask.sum()
    t[:n] = np.linspace(0, n * (60 / 5035), n)

    idx = n
    last = start + 1
    while idx < len(seconds):
        if seconds[idx] != last:
            chunk = idx - n
            if chunk > 0:
                dt = 60 / chunk
                arr = np.arange(dt, 60 + dt, dt) + t[n - 1]
                t[n:n + chunk] = arr[:chunk]
            n += max(chunk, 1)
            last += 1
        idx += 1

    return pl.Series("Time", t)

if "Time" not in df.columns:
    df = df.with_columns(timeCol(df))


df = df.filter(
    (pl.col("VDM_GPS_VALID1") == 1) &
    (pl.col("VDM_GPS_Latitude") != 0) &
    (pl.col("VDM_GPS_Longitude") != 0)
)

print(f"Rows with valid GPS: {df.height}")

R = 6378137
lat0 = np.radians(df["VDM_GPS_Latitude"][0])
lon0 = np.radians(df["VDM_GPS_Longitude"][0])

lat = np.radians(df["VDM_GPS_Latitude"].to_numpy())
lon = np.radians(df["VDM_GPS_Longitude"].to_numpy())

x = (lon - lon0) * np.cos(lat0) * R
y = (lat - lat0) * R

dx = np.diff(x, prepend=x[0])
dy = np.diff(y, prepend=y[0])
ds = np.sqrt(dx**2 + dy**2)
s = np.cumsum(ds)

df = df.with_columns([
    pl.Series("X", x),
    pl.Series("Y", y),
    pl.Series("S", s)
])

lap = np.zeros(len(df), dtype=int)
lap[0] = 1
cur = 1

for i in range(1, len(df)):
    if df["X"][i] > 80 and df["X"][i - 1] < 80:
        cur += 1
    lap[i] = cur

df = df.with_columns(pl.Series("Lap", lap))
laps = sorted(df["Lap"].unique())
print(f"Laps detected: {len(laps)}")


turn_indices = {
    1: 148819,
    4: 192977
}

turn_s_global = {k: float(df["S"][v]) for k, v in turn_indices.items()}

lap_lengths = df.group_by("Lap").agg(
    (pl.col("S").max() - pl.col("S").min()).alias("len")
)["len"].to_numpy()

lap_len = np.median(lap_lengths)

turn_s = {k: v % lap_len for k, v in turn_s_global.items()}


def nearest_by_s(lap_df, target):
    s_rel = lap_df["S"].to_numpy() - lap_df["S"][0]
    return int(np.argmin(np.abs(s_rel - target)))


entry_speeds = []
durations = []
valid_laps = []

for lap_no in laps:
    lap_df = df.filter(pl.col("Lap") == lap_no)
    if lap_df.height < 100:
        continue

    s_rel = lap_df["S"].to_numpy() - lap_df["S"][0]
    t_rel = lap_df["Time"].to_numpy() - lap_df["Time"][0]

    i1 = nearest_by_s(lap_df, turn_s[1])
    i4 = nearest_by_s(lap_df, turn_s[4])

    s1, s4 = s_rel[i1], s_rel[i4]
    t1, t4 = t_rel[i1], t_rel[i4]

    if s4 < s1:
        s4 += lap_len
        t4 += t_rel[-1] - t_rel[0]

    duration = t4 - t1
    if duration < 3 or duration > 25:
        continue

    v_entry = float(lap_df["VDM_GPS_SPEED"][i1]) * 0.44704

    entry_speeds.append(v_entry)
    durations.append(duration)
    valid_laps.append(lap_no)

print(f"Valid laps used: {len(valid_laps)}")
print(f"Duration range: {min(durations):.2f}s → {max(durations):.2f}s")


plt.figure(figsize=(12, 7))
plt.scatter(entry_speeds, durations, s=90, edgecolor="black")

for i, lap in enumerate(valid_laps):
    plt.text(entry_speeds[i], durations[i], f"L{lap}", fontsize=9)

plt.xlabel("Turn 1 Entry Speed (m/s)")
plt.ylabel("Time from Turn 1 → Turn 4 (s)")
plt.title("Turn 1 Entry Speed vs T1→T4 Time (ALL LAPS, WRAP-SAFE)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()