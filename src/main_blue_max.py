# src/main_blue_max.py

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

# --- columns we care about in this file ---
LAT_COL = "VDM_GPS_Latitude"
LON_COL = "VDM_GPS_Longitude"
SPEED_COL = "VDM_GPS_SPEED"

# path to the session you want to analyze
DATA_PATH = "FS-3/08102025/08102025Endurance1_FirstHalf.parquet"
# or your debug file:
# DATA_PATH = "FS-3/08102025/08102025Debug25.parquet"


def main() -> None:
    # 1) Load parquet
    df = pl.read_parquet(DATA_PATH)
    print("[main] raw rows:", df.height)
    print("[main] columns:", df.columns)

    # 2) Build a simple time column (same 60/5035dt idea from last year)
    n = df.height
    dt = 60.0 / 5035.0          # seconds between samples (approx)
    t = np.arange(n) * dt       # 0, dt, 2*dt, ...
    df = df.with_columns(pl.Series("time", t))

    # 3) Keep only rows with valid GPS
    if LAT_COL not in df.columns or LON_COL not in df.columns:
        print("[main] ERROR: this file has no GPS columns.")
        return

    df = df.filter((pl.col(LAT_COL) != 0) & (pl.col(LON_COL) != 0))
    print("[main] rows after GPS filter:", df.height)
    if df.height == 0:
        print("[main] ERROR: after filtering GPS==0, no data left.")
        return

    # 4) Compute total time from first to last valid GPS sample
    t_start = df["time"].min()
    t_end = df["time"].max()
    total_time = t_end - t_start
    print(f"[main] total time from first to last GPS sample: {total_time:.2f} s")

    # 5) Prepare data for plotting
    lon = df[LON_COL].to_numpy()
    lat = df[LAT_COL].to_numpy()
    speed = df[SPEED_COL].to_numpy() if SPEED_COL in df.columns else None

    # 6) Plot track outline
    plt.figure(figsize=(8, 6))

    if speed is not None:
        # scatter with color = speed
        sc = plt.scatter(lon, lat, c=speed, s=3, cmap="viridis")
        cbar = plt.colorbar(sc)
        cbar.set_label("Speed (units from VDM_GPS_SPEED)")
    else:
        # fallback: simple line if no speed column
        plt.plot(lon, lat, linewidth=1.0, label="Driving line")
        plt.legend()

    plt.axis("scaled")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Track Outline — total time ≈ {total_time:.1f} s")

    plt.tight_layout()

    # 7) Save + show figure so you can compare runs later
    out_name = "track_outline.png"
    plt.savefig(out_name, dpi=200)
    print(f"[main] saved figure to {out_name}")

    plt.show()


if __name__ == "__main__":
    main()
