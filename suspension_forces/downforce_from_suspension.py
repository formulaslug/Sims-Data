import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# need to fill in real values before running

DATA_DIR = Path("data")
PARQUET_FILE = Path("/fs-data/FS-3/01112026/011026-1.parquet")

# Time window to analyze (milliseconds)
TIME_MIN_MS = 40_000
TIME_MAX_MS = 95_000

# Baseline window — must be a period where the car is STATIONARY
# (used to zero out the spring deflection at rest)
BASELINE_START_MS = 40_000
BASELINE_END_MS   = 45_000

# Smoothing — rolling mean window (samples)
ROLLING_WINDOW = 5

# Spring rates
# Units: lb/in  (wheel rate, not spring rate)
# need to replace with real values 
SPRING_RATES_LB_PER_IN = {
    "FL": 200.0,
    "FR": 200.0,
    "BL": 200.0,
    "BR": 200.0,
}


TRAVEL_COLS = {
    "FL": "TPERIPH_FL_DATA_SUSTRAVEL",
    "FR": "TPERIPH_FR_DATA_SUSTRAVEL",
    "BL": "TPERIPH_BL_DATA_SUSTRAVEL",
    "BR": "TPERIPH_BR_DATA_SUSTRAVEL",
}

# True: larger sensor value means more compression (spring more loaded)
# False: larger sensor value means more extension  (spring less loaded)
COMPRESSION_IS_POSITIVE = True


# Wheel travel / spring travel.  Typical FSAE range: 0.6 – 0.9.
# At MR=1.0 this has no effect, so safe to leave until we get have real values.
MOTION_RATIOS = {
    "FL": 1.0,
    "FR": 1.0,
    "BL": 1.0,
    "BR": 1.0,
}



def validate_inputs():
    if not PARQUET_FILE.exists():
        raise FileNotFoundError(f"Parquet file not found: {PARQUET_FILE}")
    if BASELINE_END_MS <= BASELINE_START_MS:
        raise ValueError("BASELINE_END_MS must be greater than BASELINE_START_MS")
    if TIME_MAX_MS <= TIME_MIN_MS:
        raise ValueError("TIME_MAX_MS must be greater than TIME_MIN_MS")
    if not (TIME_MIN_MS <= BASELINE_START_MS and BASELINE_END_MS <= TIME_MAX_MS):
        raise ValueError("Baseline window must fall inside the analysis time window")



def load_and_filter_data() -> pl.DataFrame:
    needed_cols = ["Time_ms"] + list(TRAVEL_COLS.values())
    df = pl.read_parquet(PARQUET_FILE).select(needed_cols)
    df = df.filter(
        (pl.col("Time_ms") >= TIME_MIN_MS) &
        (pl.col("Time_ms") <= TIME_MAX_MS)
    )
    if df.height == 0:
        raise ValueError("No data found in selected time window")
    return df



def smooth_travel_signals(df: pl.DataFrame) -> pl.DataFrame:
    """Apply a rolling mean to remove high-frequency sensor noise."""
    return df.with_columns([
        pl.col(col).rolling_mean(window_size=ROLLING_WINDOW).alias(col)
        for col in TRAVEL_COLS.values()
    ]).drop_nulls()


def compute_baseline(df: pl.DataFrame) -> dict:
    """
    Median suspension travel in the baseline window.
    This represents the static ride height — the reference for zero aero load.
    The baseline window MUST be a period where the car is stationary and settled.
    """
    baseline_df = df.filter(
        (pl.col("Time_ms") >= BASELINE_START_MS) &
        (pl.col("Time_ms") <= BASELINE_END_MS)
    )
    if baseline_df.height == 0:
        raise ValueError(
            "Baseline window has no rows. "
            "Check BASELINE_START_MS / BASELINE_END_MS."
        )
    return {
        corner: float(baseline_df[TRAVEL_COLS[corner]].median())
        for corner in ["FL", "FR", "BL", "BR"]
    }


def add_relative_travel(df: pl.DataFrame, baseline: dict) -> pl.DataFrame:
    """
    Subtract the static baseline from each corner's travel signal.
    Result is the ADDITIONAL compression beyond the static ride height.
    Positive = more compressed than baseline (more load).
    Negative = more extended than baseline (less load / aero lift).
    """
    exprs = []
    for corner in ["FL", "FR", "BL", "BR"]:
        travel_col = TRAVEL_COLS[corner]
        rel_col    = f"{corner}_REL_MM"
        if COMPRESSION_IS_POSITIVE:
            expr = (pl.col(travel_col) - baseline[corner]).alias(rel_col)
        else:
            expr = (baseline[corner] - pl.col(travel_col)).alias(rel_col)
        exprs.append(expr)
    return df.with_columns(exprs)



def add_corner_loads(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert relative wheel travel to an apparent additional load at each corner.

    Formula per corner:
        spring_displacement_in = (wheel_travel_mm / 25.4) / motion_ratio
        apparent_load_lb       = spring_rate_lb_per_in × spring_displacement_in

    Notes:
    - Dividing by motion ratio is correct: a MR < 1 means the spring moves
      LESS than the wheel, so the spring displacement is smaller.
    - This gives the ADDITIONAL load vs. static, not absolute corner weight.
    - With placeholder spring rates the magnitudes are not meaningful —
      only the shape of the curves is valid.
    """
    load_exprs = []
    force_cols = []

    for corner in ["FL", "FR", "BL", "BR"]:
        rel_col  = f"{corner}_REL_MM"
        load_col = f"{corner}_APPARENT_LOAD_LB"
        k  = SPRING_RATES_LB_PER_IN[corner]
        mr = MOTION_RATIOS[corner]

        # mm → in, then divide by MR to get spring displacement, then × spring rate
        load_exprs.append(
            ((pl.col(rel_col) / 25.4) / mr * k).alias(load_col)
        )
        force_cols.append(load_col)

    df = df.with_columns(load_exprs)

    # Axle and total sums
    df = df.with_columns([
        (pl.col("FL_APPARENT_LOAD_LB") + pl.col("FR_APPARENT_LOAD_LB"))
            .alias("FRONT_APPARENT_LOAD_LB"),
        (pl.col("BL_APPARENT_LOAD_LB") + pl.col("BR_APPARENT_LOAD_LB"))
            .alias("REAR_APPARENT_LOAD_LB"),
        sum(pl.col(c) for c in force_cols)
            .alias("TOTAL_APPARENT_DOWNFORCE_LB"),
    ])

    # Also keep SI units
    df = df.with_columns([
        (pl.col("TOTAL_APPARENT_DOWNFORCE_LB") * 4.44822)
            .alias("TOTAL_APPARENT_DOWNFORCE_N")
    ])

    return df



# OUTPUT

def save_summary(df: pl.DataFrame):
    summary = pl.DataFrame({
        "metric": [
            "mean_total_apparent_downforce_lb",
            "max_total_apparent_downforce_lb",
            "min_total_apparent_downforce_lb",
            "mean_total_apparent_downforce_N",
            "mean_front_apparent_load_lb",
            "mean_rear_apparent_load_lb",
        ],
        "value": [
            float(df["TOTAL_APPARENT_DOWNFORCE_LB"].mean()),
            float(df["TOTAL_APPARENT_DOWNFORCE_LB"].max()),
            float(df["TOTAL_APPARENT_DOWNFORCE_LB"].min()),
            float(df["TOTAL_APPARENT_DOWNFORCE_N"].mean()),
            float(df["FRONT_APPARENT_LOAD_LB"].mean()),
            float(df["REAR_APPARENT_LOAD_LB"].mean()),
        ],
    })
    summary.write_csv(DATA_DIR / "downforce_summary.csv")


def make_plots(df: pl.DataFrame):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    time_s = df["Time_ms"].to_numpy() / 1000.0

    def _save(name: str):
        plt.grid(True, linewidth=0.5, alpha=0.6)
        plt.tight_layout()
        plt.savefig(DATA_DIR / name, dpi=200)
        plt.close()

    # 1. Raw travel
    plt.figure(figsize=(12, 5))
    for corner in ["FL", "FR", "BL", "BR"]:
        plt.plot(time_s, df[TRAVEL_COLS[corner]].to_numpy(), label=corner)
    plt.xlabel("Time [s]")
    plt.ylabel("Suspension travel [mm]")
    plt.title("Suspension travel vs time")
    plt.legend()
    _save("suspension_travel_vs_time.png")

    # 2. Relative compression
    plt.figure(figsize=(12, 5))
    for corner in ["FL", "FR", "BL", "BR"]:
        plt.plot(time_s, df[f"{corner}_REL_MM"].to_numpy(), label=f"{corner} rel")
    plt.axhline(0, color="k", linewidth=0.8, linestyle="--", label="baseline")
    plt.xlabel("Time [s]")
    plt.ylabel("Additional compression vs baseline [mm]")
    plt.title("Relative suspension compression vs time")
    plt.legend()
    _save("relative_compression_vs_time.png")

    # 3. Corner apparent additional loads
    plt.figure(figsize=(12, 5))
    for corner in ["FL", "FR", "BL", "BR"]:
        plt.plot(time_s, df[f"{corner}_APPARENT_LOAD_LB"].to_numpy(), label=corner)
    plt.axhline(0, color="k", linewidth=0.8, linestyle="--")
    plt.xlabel("Time [s]")
    plt.ylabel("Apparent additional load [lb]")
    plt.title("Corner apparent additional loads vs time\n"
              "(delta from static — NOT absolute corner weight)")
    plt.legend()
    _save("corner_apparent_loads_vs_time.png")

    # 4. Total apparent downforce (both units)
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    lb_data = df["TOTAL_APPARENT_DOWNFORCE_LB"].to_numpy()
    ax1.plot(time_s, lb_data, color="tab:blue", label="Total [lb]")
    ax2.plot(time_s, lb_data * 4.44822, color="tab:orange", alpha=0.0)  # hidden, just for scale
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Apparent additional downforce [lb]", color="tab:blue")
    ax2.set_ylabel("Apparent additional downforce [N]", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    # sync axes
    ax2.set_ylim(ax1.get_ylim()[0] * 4.44822, ax1.get_ylim()[1] * 4.44822)
    ax1.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax1.set_title("Estimated total apparent downforce vs time\n"
                  "(requires real spring rates for valid magnitude)")
    ax1.grid(True, linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(DATA_DIR / "estimated_downforce_vs_time.png", dpi=200)
    plt.close()

    # 5. Front vs rear
    plt.figure(figsize=(12, 5))
    plt.plot(time_s, df["FRONT_APPARENT_LOAD_LB"].to_numpy(), label="Front axle")
    plt.plot(time_s, df["REAR_APPARENT_LOAD_LB"].to_numpy(),  label="Rear axle")
    plt.axhline(0, color="k", linewidth=0.8, linestyle="--")
    plt.xlabel("Time [s]")
    plt.ylabel("Apparent additional load [lb]")
    plt.title("Front vs rear apparent additional load")
    plt.legend()
    _save("front_vs_rear_apparent_load.png")



def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    validate_inputs()

    df = load_and_filter_data()
    print(f"Loaded {df.height} rows ({TIME_MIN_MS/1000:.1f}s – {TIME_MAX_MS/1000:.1f}s)")

    df = smooth_travel_signals(df)

    baseline = compute_baseline(df)
    print("\nBaseline suspension travel (static ride height) [mm]:")
    for corner, val in baseline.items():
        print(f"  {corner}: {val:.3f}")

    df = add_relative_travel(df, baseline)
    df = add_corner_loads(df)

    # Print sanity-check summary
    print("\nApparent additional downforce summary:")
    print(f"  Mean  : {df['TOTAL_APPARENT_DOWNFORCE_LB'].mean():.1f} lb  "
          f"({df['TOTAL_APPARENT_DOWNFORCE_N'].mean():.1f} N)")
    print(f"  Max   : {df['TOTAL_APPARENT_DOWNFORCE_LB'].max():.1f} lb")
    print(f"  Min   : {df['TOTAL_APPARENT_DOWNFORCE_LB'].min():.1f} lb")
    print("\n  NOTE: Magnitudes are placeholder until real spring rates are entered.")

    df.write_csv(DATA_DIR / "downforce_estimate_output.csv")
    save_summary(df)
    make_plots(df)

    print("\nSaved outputs:")
    for f in [
        "downforce_estimate_output.csv",
        "downforce_summary.csv",
        "suspension_travel_vs_time.png",
        "relative_compression_vs_time.png",
        "corner_apparent_loads_vs_time.png",
        "estimated_downforce_vs_time.png",
        "front_vs_rear_apparent_load.png",
    ]:
        print(f"  {DATA_DIR / f}")


if __name__ == "__main__":
    main()
