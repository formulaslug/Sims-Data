import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path

PATH = "/workspaces/Sims-Data/fs-data/FS-3/01112026/011026-1.parquet"
OUT = Path("data/01112026")
OUT.mkdir(parents=True, exist_ok=True)

window   = (40_000, 95_000)
baseline = (40_000, 45_000)
spring_rate = 200.0

cols = [
    "Time_ms",
    "TPERIPH_FR_DATA_SUSTRAVEL",
    "TPERIPH_BR_DATA_SUSTRAVEL",
    "VDM_X_AXIS_ACCELERATION",
    "ETC_STATUS_BRAKE_SENSE_VOLTAGE",
    "ETC_STATUS_PEDAL_TRAVEL",
]

df = pl.read_parquet(PATH).select(cols).filter(
    (pl.col("Time_ms") >= window[0]) & (pl.col("Time_ms") <= window[1])
)
df = df.with_columns([pl.col(c).forward_fill() for c in cols if c != "Time_ms"]).drop_nulls()

base = df.filter((pl.col("Time_ms") >= baseline[0]) & (pl.col("Time_ms") <= baseline[1]))
fr_base = float(base["TPERIPH_FR_DATA_SUSTRAVEL"].median())
br_base = float(base["TPERIPH_BR_DATA_SUSTRAVEL"].median())
df = df.with_columns([
    ((pl.col("TPERIPH_FR_DATA_SUSTRAVEL") - fr_base) / 25.4 * spring_rate * 2).alias("FRONT_LB"),
    ((pl.col("TPERIPH_BR_DATA_SUSTRAVEL") - br_base) / 25.4 * spring_rate * 2).alias("REAR_LB"),
])
df = df.with_columns((pl.col("FRONT_LB") + pl.col("REAR_LB")).alias("TOTAL_LB"))

t = df["Time_ms"].to_numpy() / 1000
fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True)

axes[0].plot(t, df["TOTAL_LB"].to_numpy(), color="black")
axes[0].axhline(0, color="k", lw=0.5, ls="--")
axes[0].set_ylabel("Load [lb]")
axes[0].grid(alpha=0.3)

axes[1].plot(t, df["VDM_X_AXIS_ACCELERATION"].to_numpy(), color="tab:red")
axes[1].axhline(0, color="k", lw=0.5, ls="--")
axes[1].set_ylabel("Lon accel [g]")
axes[1].grid(alpha=0.3)

axes[2].plot(t, df["ETC_STATUS_BRAKE_SENSE_VOLTAGE"].to_numpy(), color="tab:red")
axes[2].set_ylabel("Brake [V]")
axes[2].grid(alpha=0.3)

axes[3].plot(t, df["ETC_STATUS_PEDAL_TRAVEL"].to_numpy(), color="tab:green")
axes[3].set_ylabel("Throttle")
axes[3].set_xlabel("Time [s]")
axes[3].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "validation.png", dpi=150)
plt.close()

print(f"saved: {OUT / 'validation.png'}")