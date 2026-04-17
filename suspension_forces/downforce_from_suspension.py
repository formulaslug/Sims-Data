import polars as pl
import matplotlib.pyplot as plt
import numpy as np

# ========= SETTINGS =========
PARQUET_FILE = "data/fs3norcal_100ms.parquet"

TIME_MIN_MS = 40000
TIME_MAX_MS = 95000

BASELINE_START_MS = 40000
BASELINE_END_MS = 45000

ROLLING_WINDOW = 5

# replace with real values if you have them
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

# set this depending on how your sensors behave
COMPRESSION_IS_POSITIVE = True


def main():
    df = pl.read_parquet(PARQUET_FILE).select(["Time_ms"] + list(TRAVEL_COLS.values()))

    df = df.filter(
        (pl.col("Time_ms") >= TIME_MIN_MS) &
        (pl.col("Time_ms") <= TIME_MAX_MS)
    )

    # smooth travel signals
    df = df.with_columns([
        pl.col(TRAVEL_COLS["FL"]).rolling_mean(window_size=ROLLING_WINDOW).alias(TRAVEL_COLS["FL"]),
        pl.col(TRAVEL_COLS["FR"]).rolling_mean(window_size=ROLLING_WINDOW).alias(TRAVEL_COLS["FR"]),
        pl.col(TRAVEL_COLS["BL"]).rolling_mean(window_size=ROLLING_WINDOW).alias(TRAVEL_COLS["BL"]),
        pl.col(TRAVEL_COLS["BR"]).rolling_mean(window_size=ROLLING_WINDOW).alias(TRAVEL_COLS["BR"]),
    ]).drop_nulls()

    # baseline ride height window
    baseline_df = df.filter(
        (pl.col("Time_ms") >= BASELINE_START_MS) &
        (pl.col("Time_ms") <= BASELINE_END_MS)
    )

    baseline = {
        corner: float(baseline_df[TRAVEL_COLS[corner]].mean())
        for corner in ["FL", "FR", "BL", "BR"]
    }

    # relative compression from baseline
    rel_exprs = []
    force_exprs = []
    force_cols = []

    for corner in ["FL", "FR", "BL", "BR"]:
        travel_col = TRAVEL_COLS[corner]
        rel_col = f"{corner}_REL_MM"
        force_col = f"{corner}_APPARENT_LOAD_LB"

        if COMPRESSION_IS_POSITIVE:
            rel_expr = (pl.col(travel_col) - baseline[corner]).alias(rel_col)
        else:
            rel_expr = (baseline[corner] - pl.col(travel_col)).alias(rel_col)

        rel_exprs.append(rel_expr)

        # mm -> in, then multiply by spring rate
        force_exprs.append(
            ((pl.col(rel_col) / 25.4) * SPRING_RATES_LB_PER_IN[corner]).alias(force_col)
        )
        force_cols.append(force_col)

    df = df.with_columns(rel_exprs)
    df = df.with_columns(force_exprs)

    df = df.with_columns(
        sum(pl.col(c) for c in force_cols).alias("TOTAL_APPARENT_DOWNFORCE_LB")
    )

    # save csv if you want
    df.write_csv("data/downforce_estimate_output.csv")

    time_s = df["Time_ms"].to_numpy() / 1000.0

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
    plt.savefig("data/suspension_travel_vs_time.png", dpi=200)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(time_s, df["TOTAL_APPARENT_DOWNFORCE_LB"].to_numpy(), label="Estimated downforce")
    plt.xlabel("Time [s]")
    plt.ylabel("Apparent downforce [lb]")
    plt.title("Estimated Downforce vs Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("data/estimated_downforce_vs_time.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
