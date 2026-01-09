#!/usr/bin/env python3

import argparse
import sys
from typing import Optional, Tuple, List

import numpy as np
import polars as pl
import matplotlib.pyplot as plt


# Column names
COL_TORQUE_DEMAND = "SME_THROTL_TorqueDemand"
COL_BUS_V         = "SME_TEMP_DC_Bus_V"
COL_BUS_C         = "SME_TEMP_BusCurrent"

COL_LAT           = "VDM_GPS_Latitude"
COL_LON           = "VDM_GPS_Longitude"
COL_SPEED_MPH     = "VDM_GPS_SPEED"
COL_GPS_VALID     = "VDM_GPS_VALID1"

COL_AX_G          = "VDM_X_AXIS_ACCELERATION"
COL_AY_G          = "VDM_Y_AXIS_ACCELERATION"

COL_BRAKE_VOLT    = "ETC_STATUS_BRAKE_SENSE_VOLTAGE"
COL_BRAKES_F      = "TMAIN_DATA_BRAKES_F"
COL_BRAKES_R      = "TMAIN_DATA_BRAKES_R"
COL_BRAKE_FLAG    = "SME_TRQSPD_Pedal_Brake"

BLUEMAX_GPS_SQUARE = ((-121.7330999, 38.5759097), (-121.7328352, 38.5757670))


def mph_to_mps(x: pl.Expr) -> pl.Expr:
    return x * 0.44704


def infer_time_column(df: pl.DataFrame) -> Tuple[pl.DataFrame, str]:
    """Find existing time column or create synthetic one."""
    candidates = [
        "Time", "time", "t", "Seconds", "seconds",
        "Timestamp", "timestamp", "Time_s", "time_s",
        "VDM_Time", "VDM_TIME", "VMD_Time", "VMD_TIME",
    ]
    for c in candidates:
        if c in df.columns:
            if df[c].dtype in (pl.Int64, pl.Int32, pl.Float64, pl.Float32, pl.UInt64, pl.UInt32):
                sample = df.select(pl.col(c)).head(5000)[c].to_numpy()
                if len(sample) >= 3:
                    diffs = np.diff(sample.astype(float))
                    if np.nanmean(diffs >= 0) > 0.9:
                        return df, c

    dt = 0.02
    df = df.with_row_index(name="_idx").with_columns((pl.col("_idx") * dt).alias("Time_s")).drop("_idx")
    return df, "Time_s"


def add_lap_segmentation(df: pl.DataFrame,
                         lat_col: str,
                         lon_col: str,
                         time_col: str,
                         square: Tuple[Tuple[float, float], Tuple[float, float]],
                         gps_valid_col: Optional[str] = None,
                         min_lap_time_s: float = 10.0) -> pl.DataFrame:
    (lon1, lat1), (lon2, lat2) = square
    lon_min, lon_max = (min(lon1, lon2), max(lon1, lon2))
    lat_min, lat_max = (min(lat1, lat2), max(lat1, lat2))

    inside_expr = (
        (pl.col(lon_col) >= lon_min) & (pl.col(lon_col) <= lon_max) &
        (pl.col(lat_col) >= lat_min) & (pl.col(lat_col) <= lat_max)
    )

    if gps_valid_col and gps_valid_col in df.columns:
        inside_expr = inside_expr & (pl.col(gps_valid_col) != 0) & pl.col(lat_col).is_not_null() & pl.col(lon_col).is_not_null()
    else:
        inside_expr = inside_expr & pl.col(lat_col).is_not_null() & pl.col(lon_col).is_not_null()

    df = df.with_columns(inside_expr.cast(pl.Int8).alias("_inside"))
    df = df.with_columns([
        pl.col("_inside").shift(1).fill_null(0).alias("_inside_prev"),
        (pl.col(time_col)).alias("_t"),
    ]).with_columns([
        ((pl.col("_inside") == 1) & (pl.col("_inside_prev") == 0)).cast(pl.Int8).alias("_raw_cross")
    ])

    t = df["_t"].to_numpy().astype(float)
    raw = df["_raw_cross"].to_numpy().astype(int)

    valid = np.zeros_like(raw, dtype=int)
    last_cross_t = -1e18
    for i in range(len(raw)):
        if raw[i] == 1 and (t[i] - last_cross_t) >= min_lap_time_s:
            valid[i] = 1
            last_cross_t = t[i]

    lap = np.cumsum(valid).astype(int)
    df = df.with_columns([
        pl.Series("Lap", lap)
    ])

    df = df.drop(["_inside", "_inside_prev", "_raw_cross", "_t"])
    return df


def add_track_distance_per_lap(df: pl.DataFrame, lat_col: str, lon_col: str, lap_col: str) -> pl.DataFrame:
    df = df.with_columns([
        pl.col(lat_col).cast(pl.Float64),
        pl.col(lon_col).cast(pl.Float64),
    ])

    df = df.with_columns([
        pl.col(lat_col).mean().over(lap_col).alias("_lat0"),
    ])

    df = df.with_columns([
        (111320.0 * (pl.col(lat_col) - pl.col("_lat0"))).alias("_y_m"),
        (111320.0 * (pl.col(lon_col) - pl.col(lon_col).mean().over(lap_col)) * pl.col("_lat0").radians().cos()).alias("_x_m"),
    ])

    df = df.with_columns([
        pl.col("_x_m").shift(1).over(lap_col).alias("_x_prev"),
        pl.col("_y_m").shift(1).over(lap_col).alias("_y_prev"),
    ])

    df = df.with_columns([
        (((pl.col("_x_m") - pl.col("_x_prev"))**2 + (pl.col("_y_m") - pl.col("_y_prev"))**2).sqrt())
        .fill_null(0.0)
        .alias("ds_m")
    ]).with_columns([
        pl.col("ds_m").cum_sum().over(lap_col).alias("s_m"),
    ])

    df = df.with_columns([
        (pl.col("s_m") / pl.col("s_m").max().over(lap_col)).alias("s_norm")
    ])

    df = df.drop(["_lat0", "_x_m", "_y_m", "_x_prev", "_y_prev"])
    return df


def add_loss_powers(df: pl.DataFrame,
                    speed_mps_col: str,
                    accel_mps2_col: str,
                    rho: float,
                    CdA: float,
                    mass_kg: float) -> pl.DataFrame:
    df = df.with_columns([
        (0.5 * rho * CdA * (pl.col(speed_mps_col) ** 2)).alias("DragForce_N"),
        (0.5 * rho * CdA * (pl.col(speed_mps_col) ** 3)).alias("DragPower_W"),
    ])

    df = df.with_columns([
        pl.when(pl.col(accel_mps2_col) < 0)
        .then((-mass_kg * pl.col(accel_mps2_col) * pl.col(speed_mps_col)))
        .otherwise(0.0)
        .alias("BrakePower_W")
    ])

    df = df.with_columns([
        (pl.col("DragPower_W") + pl.col("BrakePower_W")).alias("TotalLossPower_W")
    ])
    return df


def plot_multiple_laps_comparison(df: pl.DataFrame, lap_nums: List[int], metric_col: str,
                                  lat_col: str, lon_col: str, title: str):
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray']
    all_z = []
    lap_data = []
    for lap_num in lap_nums:
        lap_df = (
            df.filter(pl.col("Lap") == lap_num)
              .filter(pl.col(lat_col).is_not_null() & pl.col(lon_col).is_not_null())
              .filter((pl.col(lat_col) != 0) & (pl.col(lon_col) != 0))
              .sort("s_norm")  # Sort by normalized distance for smooth line
        )
        
        if lap_df.height < 5:
            print(f"[plot_multiple_laps_comparison] Not enough GPS points for lap {lap_num}")
            continue
        
        x = lap_df[lon_col].to_numpy()
        y = lap_df[lat_col].to_numpy()
        z = lap_df[metric_col].to_numpy() / 1000.0
        
        all_z.extend(z[~np.isnan(z)])
        lap_data.append((lap_num, x, y, z))
    
    if not lap_data:
        print("[plot_multiple_laps_comparison] No valid lap data to plot")
        return
    
    vmin = np.nanmin(all_z) if all_z else 0
    vmax = np.nanmax(all_z) if all_z else 1
    
    for idx, (lap_num, x, y, z) in enumerate(lap_data):
        color = colors[idx % len(colors)]
        sc = ax.scatter(x, y, c=z, s=8, alpha=0.7, cmap='viridis', 
                       vmin=vmin, vmax=vmax, label=f'Lap {lap_num}')
        ax.plot(x, y, color=color, alpha=0.3, linewidth=1, linestyle='--')
    
    plt.colorbar(sc, ax=ax, label=f"{metric_col} (kW)")
    
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()


def quick_lap_stats(df: pl.DataFrame, time_col: str):
    laps = df.select(pl.col("Lap").max().alias("max_lap"))["max_lap"][0]
    if laps is None or laps <= 0:
        print("[quick_lap_stats] No laps found (Lap column <= 0).")
        return

    lap_times = (
        df.filter(pl.col("Lap") > 0)
          .group_by("Lap")
          .agg([
              pl.min(time_col).alias("t0"),
              pl.max(time_col).alias("t1"),
              pl.count().alias("rows"),
          ])
          .with_columns((pl.col("t1") - pl.col("t0")).alias("lap_time_s"))
          .sort("Lap")
    )
    print("\nLap summary (first 15):")
    print(lap_times.head(15))
    print("\nLap time stats:")
    print(lap_times.select([
        pl.count().alias("lap_count"),
        pl.mean("lap_time_s").alias("mean_s"),
        pl.median("lap_time_s").alias("median_s"),
        pl.min("lap_time_s").alias("min_s"),
        pl.max("lap_time_s").alias("max_s"),
    ]))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to parquet file")
    ap.add_argument("--CdA", type=float, default=0.7, help="Drag Cd*A (m^2)")
    ap.add_argument("--rho", type=float, default=1.225, help="Air density kg/m^3")
    ap.add_argument("--mass", type=float, default=300.0, help="Vehicle mass kg")
    ap.add_argument("--min_lap_time", type=float, default=10.0, help="Min time between lap triggers (s)")
    args = ap.parse_args()

    print(f"Loading: {args.file}")
    df = pl.read_parquet(args.file)

    required = [COL_LAT, COL_LON, COL_SPEED_MPH, COL_AX_G]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("ERROR: Missing required columns:")
        for c in missing:
            print("  -", c)
        print("\nAvailable columns sample:", df.columns[:30], "...")
        sys.exit(1)

    df, time_col = infer_time_column(df)
    print(f"Using time column: {time_col}")

    if COL_GPS_VALID in df.columns:
        df = df.filter((pl.col(COL_GPS_VALID) != 0) & (pl.col(COL_LAT) != 0) & (pl.col(COL_LON) != 0))
    else:
        df = df.filter((pl.col(COL_LAT) != 0) & (pl.col(COL_LON) != 0))

    df = df.with_columns([
        mph_to_mps(pl.col(COL_SPEED_MPH).cast(pl.Float64)).alias("Speed_mps"),
        (pl.col(COL_AX_G).cast(pl.Float64) * 9.81).alias("Ax_mps2"),
    ])

    print("Segmenting laps (GPS gate)...")
    df = add_lap_segmentation(
        df,
        lat_col=COL_LAT,
        lon_col=COL_LON,
        time_col=time_col,
        square=BLUEMAX_GPS_SQUARE,
        gps_valid_col=COL_GPS_VALID if COL_GPS_VALID in df.columns else None,
        min_lap_time_s=args.min_lap_time
    )

    quick_lap_stats(df, time_col)

    print("Computing distance along lap...")
    df = add_track_distance_per_lap(df, lat_col=COL_LAT, lon_col=COL_LON, lap_col="Lap")

    print("Computing drag/brake/total loss powers...")
    df = add_loss_powers(
        df,
        speed_mps_col="Speed_mps",
        accel_mps2_col="Ax_mps2",
        rho=args.rho,
        CdA=args.CdA,
        mass_kg=args.mass
    )

    # ========================================================================
    # LAP COMPARISON CONFIGURATION
    # ========================================================================
    # To change which laps are compared, modify the list below.
    # Examples:
    #   comparison_laps = [1, 2, 5]        # Compare laps 1, 2, and 5
    #   comparison_laps = [3, 4, 6, 7]     # Compare laps 3, 4, 6, and 7
    #   comparison_laps = [1, 5, 10]        # Compare laps 1, 5, and 10
    #   comparison_laps = list(range(1, 6)) # Compare laps 1 through 5
    # ========================================================================
    comparison_laps = [1, 2, 5]  # <-- EDIT THIS LIST TO CHANGE WHICH LAPS ARE COMPARED
    
    lap_str = ", ".join(map(str, comparison_laps))
    print(f"Generating comparison plots for laps {lap_str}...")
    plot_multiple_laps_comparison(
        df, comparison_laps, "BrakePower_W", COL_LAT, COL_LON,
        title=f"Lap Comparison: BrakePower_W on Track (kW) - Laps {lap_str}"
    )
    plot_multiple_laps_comparison(
        df, comparison_laps, "DragPower_W", COL_LAT, COL_LON,
        title=f"Lap Comparison: DragPower_W on Track (kW) - Laps {lap_str}"
    )
    plot_multiple_laps_comparison(
        df, comparison_laps, "TotalLossPower_W", COL_LAT, COL_LON,
        title=f"Lap Comparison: TotalLossPower_W on Track (kW) - Laps {lap_str}"
    )

    print("Done.")


if __name__ == "__main__":
    main()
