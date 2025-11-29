# Blue Max session analyzer
#   1. Load a single .parquet log file
#   2. Make a simple time column (using dt ≈ 60/5035 s)
#   3. Throw away rows with bad GPS (0,0)
#   4. Find laps by watching when the car drives through a small GPS box
#   5. Compute:
#        - basic stint stats (time, speed, power, SOC, temps)
#        - lap times
#   6. Plot:
#        - track colored by speed
#        - track colored by lap number
#        - speed & power for the whole run
#        - line per lap (colored by speed)
#        - power vs time for each lap

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

# ------------------ column names in the log ------------------ #

LAT_COL = "VDM_GPS_Latitude"
LON_COL = "VDM_GPS_Longitude"
SPEED_COL = "VDM_GPS_SPEED"          # vehicle speed

BUS_V_COL = "SME_TEMP_DC_Bus_V"      # motor controller DC bus voltage
BUS_I_COL = "SME_TEMP_BusCurrent"    # motor controller DC bus current

SOC_COL = "ACC_POWER_SOC"            # state of charge (%)
MOTOR_TEMP_COL = "SME_TEMP_MotorTemperature"

LAP_COL = "Lap"

# GPS box for start/finish line (from last year's analysis)
# box = ((lon_min, lat_max), (lon_max, lat_min))
START_FINISH_BOX = (
    (-121.7330999, 38.5759097),
    (-121.7328352, 38.5757670),
)

# use a raw string on Windows so backslashes don't explode
DATA_PATH = r"FS-3\08102025\08102025Endurance1_FirstHalf.parquet"

# sample period used in the old code
DT = 60.0 / 5035.0   # seconds per sample


# lap detection

def inside_box(lat_arr: np.ndarray, lon_arr: np.ndarray, box) -> np.ndarray:
    # Return True where a (lat, lon) sample falls inside the GPS box
    (lon_min, lat_max), (lon_max, lat_min) = box
    return (
        (lat_arr >= lat_min) & (lat_arr <= lat_max) &
        (lon_arr >= lon_min) & (lon_arr <= lon_max)
    )


def tag_laps(df: pl.DataFrame) -> pl.DataFrame:
    # add a Lap column to the DataFrame by detecting passes through the GPS box
    lats = df[LAT_COL].to_numpy()
    lons = df[LON_COL].to_numpy()

    in_box = inside_box(lats, lons, START_FINISH_BOX)

    laps = np.zeros(len(lats), dtype=int)
    lap_idx = 0
    was_in_box = False

    for i, now_in_box in enumerate(in_box):
        # rising edge: outside -> inside
        if now_in_box and not was_in_box:
            lap_idx += 1
        laps[i] = lap_idx
        was_in_box = now_in_box

    print(f"[tag_laps] found laps 1..{lap_idx}")
    return df.with_columns(pl.Series(LAP_COL, laps))

# load session and clean data

def load_session(path: str) -> pl.DataFrame:
    """Read parquet, add a time column, and drop rows with junk GPS."""
    df = pl.read_parquet(path)
    print("[load_session] raw rows:", df.height)
    print("[load_session] columns:", df.columns)

    # time column based on constant dt
    t = np.arange(df.height) * DT
    df = df.with_columns(pl.Series("time", t))

    if LAT_COL not in df.columns or LON_COL not in df.columns:
        raise RuntimeError("No GPS columns in this file.")

    # toss out rows where GPS is (0,0) RELEASE EM
    df = df.filter((pl.col(LAT_COL) != 0) & (pl.col(LON_COL) != 0))
    print("[load_session] rows after GPS filter:", df.height)

    if df.height == 0:
        raise RuntimeError("All GPS samples were (0,0) after filtering.")

    return df


def basic_session_stats(df: pl.DataFrame) -> dict:
    """Compute stint-level stats: time, speed, power, SOC, temps."""
    t = df["time"].to_numpy()
    duration = float(t[-1] - t[0])

    stats: dict[str, float | None] = {"total_time_s": duration}

    # speed stats
    if SPEED_COL in df.columns:
        v = df[SPEED_COL].to_numpy()
        stats["speed_avg"] = float(v.mean())
        stats["speed_max"] = float(v.max())
    else:
        stats["speed_avg"] = None
        stats["speed_max"] = None

    # power / energy
    if BUS_V_COL in df.columns and BUS_I_COL in df.columns:
        v = df[BUS_V_COL].to_numpy()
        i = df[BUS_I_COL].to_numpy()
        p = v * i  # watts-ish

        stats["power_peak_W"] = float(p.max())

        # numerical integral of power over time -> energy
        # numpy wants seconds on the x-axis; divide by 3600 to get Wh
        energy_Wh = np.trapezoid(p, t) / 3600.0
        stats["energy_kWh"] = energy_Wh / 1000.0
    else:
        stats["power_peak_W"] = None
        stats["energy_kWh"] = None

    # SOC
    if SOC_COL in df.columns:
        soc = df[SOC_COL].to_numpy()
        stats["soc_start"] = float(soc[0])
        stats["soc_end"] = float(soc[-1])

    # motor temp
    if MOTOR_TEMP_COL in df.columns:
        stats["motor_temp_max"] = float(df[MOTOR_TEMP_COL].max())

    return stats


def lap_table(df: pl.DataFrame) -> pl.DataFrame:
    """
    Build a DataFrame with one row per lap:
      Lap, t_start, t_end, lap_time_s
    """
    if LAP_COL not in df.columns:
        raise RuntimeError("No Lap column – run tag_laps() first.")

    laps = (
        df.group_by(LAP_COL)
          .agg(
              pl.col("time").min().alias("t_start"),
              pl.col("time").max().alias("t_end"),
          )
          .filter(pl.col(LAP_COL) > 0)  # ignore “lap 0” warmup
          .with_columns(
              (pl.col("t_end") - pl.col("t_start")).alias("lap_time_s")
          )
          .sort(LAP_COL)
    )

    print("\n[lap_table] Lap times:")
    print(laps)
    return laps

# plottinggg

def plot_track_speed(df: pl.DataFrame, stats: dict) -> None:
    """Track map colored by speed."""
    lon = df[LON_COL].to_numpy()
    lat = df[LAT_COL].to_numpy()

    plt.figure(figsize=(11, 4))

    if SPEED_COL in df.columns:
        v = df[SPEED_COL].to_numpy()
        sc = plt.scatter(lon, lat, c=v, s=4, cmap="viridis")
        cbar = plt.colorbar(sc)
        cbar.set_label("Speed (VDM_GPS_SPEED)")
    else:
        plt.plot(lon, lat, lw=1.0, label="line")
        plt.legend()

    plt.axis("scaled")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    total_t = stats.get("total_time_s", 0.0)
    plt.title(f"Track outline — stint time ≈ {total_t:.1f} s")

    plt.tight_layout()
    plt.savefig("track_outline.png", dpi=200)
    print("[plot_track_speed] saved track_outline.png")


def plot_track_laps(df: pl.DataFrame) -> None:
    """Track map colored by lap number."""
    if LAP_COL not in df.columns:
        print("[plot_track_laps] no Lap column, skipping.")
        return

    lon = df[LON_COL].to_numpy()
    lat = df[LAT_COL].to_numpy()
    laps = df[LAP_COL].to_numpy()

    plt.figure(figsize=(11, 4))
    sc = plt.scatter(lon, lat, c=laps, s=4, cmap="tab20")
    cbar = plt.colorbar(sc)
    cbar.set_label("Lap #")

    plt.axis("scaled")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Track outline by lap")

    plt.tight_layout()
    plt.savefig("track_by_lap.png", dpi=200)
    print("[plot_track_laps] saved track_by_lap.png")


def plot_speed_power_whole_stint(df: pl.DataFrame) -> None:
    """Speed and power vs time for the entire run."""
    t = df["time"].to_numpy()

    fig, ax1 = plt.subplots(figsize=(11, 4))

    if SPEED_COL in df.columns:
        v = df[SPEED_COL].to_numpy()
        ax1.plot(t, v, label="Speed", lw=1.0)
        ax1.set_ylabel("Speed")
    ax1.set_xlabel("Time (s)")

    if BUS_V_COL in df.columns and BUS_I_COL in df.columns:
        v_bus = df[BUS_V_COL].to_numpy()
        i_bus = df[BUS_I_COL].to_numpy()
        p_kw = v_bus * i_bus / 1000.0

        ax2 = ax1.twinx()
        ax2.plot(t, p_kw, color="orange", label="Power", lw=1.0)
        ax2.set_ylabel("Power (kW)")

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    plt.title("Speed and power vs time")
    plt.tight_layout()
    plt.savefig("speed_power_vs_time.png", dpi=200)
    print("[plot_speed_power_whole_stint] saved speed_power_vs_time.png")

def plot_lines_per_lap(
    df: pl.DataFrame,
    laps_df: pl.DataFrame,
    max_laps: int = 6,
) -> None:
    """Little track map for each lap (colored by speed), in chunks."""

    all_laps = laps_df[LAP_COL].to_numpy()
    lap_times = laps_df["lap_time_s"].to_numpy()

    n_total = len(all_laps)
    if n_total == 0:
        print("[plot_lines_per_lap] no laps to plot.")
        return

    ncols = 2

    # walk through all laps in chunks: 0–5, 6–11, 12–17, ...
    for start in range(0, n_total, max_laps):
        end = min(start + max_laps, n_total)

        laps_chunk = all_laps[start:end]
        times_chunk = lap_times[start:end]
        n_laps = len(laps_chunk)

        nrows = int(np.ceil(n_laps / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(11, 3 * nrows),
            squeeze=False,
        )

        for idx in range(n_laps):
            lap_num = int(laps_chunk[idx])
            lap_time = float(times_chunk[idx])
            ax = axes[idx // ncols][idx % ncols]

            dlap = df.filter(pl.col(LAP_COL) == lap_num)
            lon = dlap[LON_COL].to_numpy()
            lat = dlap[LAT_COL].to_numpy()
            speed = (
                dlap[SPEED_COL].to_numpy()
                if SPEED_COL in dlap.columns
                else None
            )

            if speed is not None:
                ax.scatter(lon, lat, c=speed, s=4, cmap="viridis")
            else:
                ax.plot(lon, lat, lw=1.0)

            ax.axis("scaled")
            ax.set_xlabel("Lon")
            ax.set_ylabel("Lat")
            ax.set_title(f"Lap {lap_num} — {lap_time:.1f} s")

        # delete unused empty subplots in this figure
        for j in range(n_laps, nrows * ncols):
            fig.delaxes(axes[j // ncols][j % ncols])

        first_lap = int(laps_chunk[0])
        last_lap = int(laps_chunk[-1])
        fig.suptitle(f"Driving line per lap (Laps {first_lap}–{last_lap})", y=0.99)
        fname = f"lines_per_lap_{first_lap}_{last_lap}.png"
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(fname, dpi=200)
        print(f"[plot_lines_per_lap] saved {fname}")


def plot_power_per_lap(
    df: pl.DataFrame,
    laps_df: pl.DataFrame,
    chunk_size: int = 6,
) -> None:
    """Power vs time for each lap, saved in chunks (like line plots)."""

    if BUS_V_COL not in df.columns or BUS_I_COL not in df.columns:
        print("[plot_power_per_lap] no bus voltage/current, skipping.")
        return

    all_laps = laps_df[LAP_COL].to_numpy()
    t_start = laps_df["t_start"].to_numpy()
    lap_times = laps_df["lap_time_s"].to_numpy()

    n_total = len(all_laps)
    if n_total == 0:
        print("[plot_power_per_lap] no laps to plot.")
        return

    ncols = 2

    # walk through laps in chunks: 1–6, 7–12, 13–18, ...
    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)

        laps_chunk = all_laps[start:end]
        times_chunk = lap_times[start:end]
        t0_chunk = t_start[start:end]

        n_laps = len(laps_chunk)
        nrows = int(np.ceil(n_laps / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(11, 3 * nrows),
            squeeze=False,
        )

        for idx in range(n_laps):
            lap_num = int(laps_chunk[idx])
            t0 = float(t0_chunk[idx])
            lap_time = float(times_chunk[idx])
            ax = axes[idx // ncols][idx % ncols]

            dlap = df.filter(pl.col(LAP_COL) == lap_num)
            t_rel = dlap["time"].to_numpy() - t0

            v_bus = dlap[BUS_V_COL].to_numpy()
            i_bus = dlap[BUS_I_COL].to_numpy()
            p_kw = v_bus * i_bus / 1000.0

            ax.plot(t_rel, p_kw, lw=1.0)
            ax.set_xlabel("Time in lap (s)")
            ax.set_ylabel("Power (kW)")
            ax.set_title(f"Lap {lap_num} — {lap_time:.1f} s")

        # remove empty subplots
        for j in range(n_laps, nrows * ncols):
            fig.delaxes(axes[j // ncols][j % ncols])

        first_lap = int(laps_chunk[0])
        last_lap = int(laps_chunk[-1])

        fig.suptitle(
            f"Power vs Time per Lap (Laps {first_lap}–{last_lap})",
            y=0.99
        )

        fname = f"power_per_lap_{first_lap}_{last_lap}.png"
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(fname, dpi=200)

        print(f"[plot_power_per_lap] saved {fname}")


# entry point


def main() -> None:
    df = load_session(DATA_PATH)
    df = tag_laps(df)

    stint_stats = basic_session_stats(df)
    laps_df = lap_table(df)

    print("\n=== Session summary ===")
    print(f"Total time         : {stint_stats['total_time_s']:.1f} s")
    if stint_stats["speed_avg"] is not None:
        print(f"Avg speed          : {stint_stats['speed_avg']:.1f}")
        print(f"Max speed          : {stint_stats['speed_max']:.1f}")
    if stint_stats["power_peak_W"] is not None:
        print(f"Peak power         : {stint_stats['power_peak_W'] / 1000.0:.1f} kW")
        print(f"Energy used (rough): {stint_stats['energy_kWh']:.2f} kWh")
    if "soc_start" in stint_stats:
        print(f"SOC start → end    : {stint_stats['soc_start']:.1f}% → "
              f"{stint_stats['soc_end']:.1f}%")
    if "motor_temp_max" in stint_stats:
        print(f"Max motor temp     : {stint_stats['motor_temp_max']:.1f} °C")

    # plots
    plot_track_speed(df, stint_stats)
    plot_track_laps(df)
    plot_speed_power_whole_stint(df)
    plot_lines_per_lap(df, laps_df, max_laps=6)
    plot_power_per_lap(df, laps_df, chunk_size=6)

    plt.show()


if __name__ == "__main__":
    main()


