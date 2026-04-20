import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
PARQUET_FILE = Path("/fs-data/FS-3/01112026/011026-1.parquet")

TIME_MIN_MS = 40000
TIME_MAX_MS = 95000

BASELINE_START_MS = 40000
BASELINE_END_MS = 45000

ROLLING_WINDOW = 5

# Replace with the real values of spring rates (unable to find)
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

# True if larger sensor value means more compression
COMPRESSION_IS_POSITIVE = True

# optional motion ratios
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


def load_and_filter_data():
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
    return df.with_columns([
        pl.col(TRAVEL_COLS["FL"]).rolling_mean(window_size=ROLLING_WINDOW).alias(TRAVEL_COLS["FL"]),
        pl.col(TRAVEL_COLS["FR"]).rolling_mean(window_size=ROLLING_WINDOW).alias(TRAVEL_COLS["FR"]),
        pl.col(TRAVEL_COLS["BL"]).rolling_mean(window_size=ROLLING_WINDOW).alias(TRAVEL_COLS["BL"]),
        pl.col(TRAVEL_COLS["BR"]).rolling_mean(window_size=ROLLING_WINDOW).alias(TRAVEL_COLS["BR"]),
    ]).drop_nulls()


def compute_baseline(df: pl.DataFrame):
    baseline_df = df.filter(
        (pl.col("Time_ms") >= BASELINE_START_MS) &
        (pl.col("Time_ms") <= BASELINE_END_MS)
    )

    if baseline_df.height == 0:
        raise ValueError("Baseline window has no rows. Choose a different baseline time range.")

    baseline = {
        corner: float(baseline_df[TRAVEL_COLS[corner]].median())
        for corner in ["FL", "FR", "BL", "BR"]
    }

    return baseline


def add_relative_travel(df: pl.DataFrame, baseline: dict) -> pl.DataFrame:
    exprs = []

    for corner in ["FL", "FR", "BL", "BR"]:
        travel_col = TRAVEL_COLS[corner]
        rel_col = f"{corner}_REL_MM"

        if COMPRESSION_IS_POSITIVE:
            expr = (pl.col(travel_col) - baseline[corner]).alias(rel_col)
        else:
            expr = (baseline[corner] - pl.col(travel_col)).alias(rel_col)

        exprs.append(expr)

    return df.with_columns(exprs)


def add_corner_loads(df: pl.DataFrame) -> pl.DataFrame:
    exprs = []
    force_cols = []

    for corner in ["FL", "FR", "BL", "BR"]:
        rel_col = f"{corner}_REL_MM"
        load_col = f"{corner}_APPARENT_LOAD_LB"

        k = SPRING_RATES_LB_PER_IN[corner]
        mr = MOTION_RATIOS[corner]

        # mm to in, then spring force
        exprs.append(
            ((pl.col(rel_col) / 25.4) * k * mr).alias(load_col)
        )
        force_cols.append(load_col)

    df = df.with_columns(exprs)

    df = df.with_columns([
        (pl.col("FL_APPARENT_LOAD_LB") + pl.col("FR_APPARENT_LOAD_LB")).alias("FRONT_APPARENT_LOAD_LB"),
        (pl.col("BL_APPARENT_LOAD_LB") + pl.col("BR_APPARENT_LOAD_LB")).alias("REAR_APPARENT_LOAD_LB"),
        sum(pl.col(c) for c in force_cols).alias("TOTAL_APPARENT_DOWNFORCE_LB"),
    ])

    df = df.with_columns([
        (pl.col("TOTAL_APPARENT_DOWNFORCE_LB") * 4.44822).alias("TOTAL_APPARENT_DOWNFORCE_N")
    ])

    return df


def save_summary(df: pl.DataFrame):
    summary = pl.DataFrame({
        "metric": [
            "mean_total_apparent_downforce_lb",
            "max_total_apparent_downforce_lb",
            "min_total_apparent_downforce_lb",
            "mean_front_apparent_load_lb",
            "mean_rear_apparent_load_lb",
        ],
        "value": [
            float(df["TOTAL_APPARENT_DOWNFORCE_LB"].mean()),
            float(df["TOTAL_APPARENT_DOWNFORCE_LB"].max()),
            float(df["TOTAL_APPARENT_DOWNFORCE_LB"].min()),
            float(df["FRONT_APPARENT_LOAD_LB"].mean()),
            float(df["REAR_APPARENT_LOAD_LB"].mean()),
        ]
    })

    summary.write_csv(DATA_DIR / "downforce_summary.csv")


def make_plots(df: pl.DataFrame):
    time_s = df["Time_ms"].to_numpy() / 1000.0

    # 1. Raw travel
    plt.figure(figsize=(12, 6))
    plt.plot(time_s, df[TRAVEL_COLS["FL"]].to_numpy(), label="FL")
    plt.plot(time_s, df[TRAVEL_COLS["FR"]].to_numpy(), label="FR")
    plt.plot(time_s, df[TRAVEL_COLS["BL"]].to_numpy(), label="BL")
    plt.plot(time_s, df[TRAVEL_COLS["BR"]].to_numpy(), label="BR")
    plt.xlabel("Time [s]")
    plt.ylabel("Suspension travel [mm]")
    plt.title("Suspension Travel vs Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(DATA_DIR / "suspension_travel_vs_time.png", dpi=200)
    plt.show()

    # 2. Relative compression
    plt.figure(figsize=(12, 6))
    plt.plot(time_s, df["FL_REL_MM"].to_numpy(), label="FL rel")
    plt.plot(time_s, df["FR_REL_MM"].to_numpy(), label="FR rel")
    plt.plot(time_s, df["BL_REL_MM"].to_numpy(), label="BL rel")
    plt.plot(time_s, df["BR_REL_MM"].to_numpy(), label="BR rel")
    plt.xlabel("Time [s]")
    plt.ylabel("Relative compression [mm]")
    plt.title("Relative Suspension Compression vs Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(DATA_DIR / "relative_compression_vs_time.png", dpi=200)
    plt.show()

    # 3. Corner loads
    plt.figure(figsize=(12, 6))
    plt.plot(time_s, df["FL_APPARENT_LOAD_LB"].to_numpy(), label="FL")
    plt.plot(time_s, df["FR_APPARENT_LOAD_LB"].to_numpy(), label="FR")
    plt.plot(time_s, df["BL_APPARENT_LOAD_LB"].to_numpy(), label="BL")
    plt.plot(time_s, df["BR_APPARENT_LOAD_LB"].to_numpy(), label="BR")
    plt.xlabel("Time [s]")
    plt.ylabel("Apparent load [lb]")
    plt.title("Corner Apparent Loads vs Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(DATA_DIR / "corner_apparent_loads_vs_time.png", dpi=200)
    plt.show()

    # 4. Total load
    plt.figure(figsize=(12, 6))
    plt.plot(time_s, df["TOTAL_APPARENT_DOWNFORCE_LB"].to_numpy(), label="Total apparent load")
    plt.xlabel("Time [s]")
    plt.ylabel("Apparent total load [lb]")
    plt.title("Estimated Downforce / Apparent Vertical Load vs Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(DATA_DIR / "estimated_downforce_vs_time.png", dpi=200)
    plt.show()

    # 5. Front vs rear
    plt.figure(figsize=(12, 6))
    plt.plot(time_s, df["FRONT_APPARENT_LOAD_LB"].to_numpy(), label="Front")
    plt.plot(time_s, df["REAR_APPARENT_LOAD_LB"].to_numpy(), label="Rear")
    plt.xlabel("Time [s]")
    plt.ylabel("Apparent load [lb]")
    plt.title("Front vs Rear Apparent Load")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(DATA_DIR / "front_vs_rear_apparent_load.png", dpi=200)
    plt.show()


def main():
    validate_inputs()

    df = load_and_filter_data()
    df = smooth_travel_signals(df)

    baseline = compute_baseline(df)
    print("Baseline suspension travel [mm]:")
    for k, v in baseline.items():
        print(f"  {k}: {v:.3f}")

    df = add_relative_travel(df, baseline)
    df = add_corner_loads(df)

    df.write_csv(DATA_DIR / "downforce_estimate_output.csv")
    save_summary(df)
    make_plots(df)

    print("\nSaved files:")
    print(DATA_DIR / "downforce_estimate_output.csv")
    print(DATA_DIR / "downforce_summary.csv")
    print(DATA_DIR / "suspension_travel_vs_time.png")
    print(DATA_DIR / "relative_compression_vs_time.png")
    print(DATA_DIR / "corner_apparent_loads_vs_time.png")
    print(DATA_DIR / "estimated_downforce_vs_time.png")
    print(DATA_DIR / "front_vs_rear_apparent_load.png")


if __name__ == "__main__":
    main()
