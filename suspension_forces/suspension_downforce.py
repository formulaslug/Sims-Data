import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path

# --- config ---
files = [
    "/workspaces/Sims-Data/fs-data/FS-3/01112026/011026-1.parquet",
    "/workspaces/Sims-Data/fs-data/FS-3/01172026/011726-1.parquet",
]
travel_cols = {
    "FL": "TPERIPH_FL_DATA_SUSTRAVEL",
    "FR": "TPERIPH_FR_DATA_SUSTRAVEL",
    "BL": "TPERIPH_BL_DATA_SUSTRAVEL",
    "BR": "TPERIPH_BR_DATA_SUSTRAVEL",
}
SPRING_RATE = 200.0
base = (40_000, 45_000)
window   = (40_000, 95_000) 
OUT = Path("data")

def run(path):
    run_id = Path(path).parent.name
    out = OUT / run_id
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n--- {run_id} ---")

    df = pl.read_parquet(path).filter(
        (pl.col("Time_ms") >= window[0]) & (pl.col("Time_ms") <= window[1])
    )
    corners = [c for c, col in travel_cols.items()
               if col in df.columns and df[col].null_count() < df.height]
    print(f"corners with data: {corners}")
    if not corners:
        return
    df = df.with_columns([pl.col(travel_cols[c]).forward_fill() for c in corners])
    df = df.drop_nulls(subset=[travel_cols[c] for c in corners])
    base_df = df.filter(
        (pl.col("Time_ms") >= base[0]) & (pl.col("Time_ms") <= base[1])
    )
    baseline = {c: float(base_df[travel_cols[c]].median()) for c in corners}
    print(f"baseline [mm]: {baseline}")
    for c in corners:
        df = df.with_columns(
            ((pl.col(travel_cols[c]) - baseline[c]) / 25.4 * SPRING_RATE)
            .alias(f"{c}_LOAD_LB")
        )
    def axle(side_cols):
        return sum(pl.col(f"{c}_LOAD_LB") for c in side_cols) * (2 / len(side_cols))
    front = [c for c in corners if c in ("FL", "FR")]
    rear  = [c for c in corners if c in ("BL", "BR")]
    df = df.with_columns([
        axle(front).alias("FRONT_LB") if front else pl.lit(None).alias("FRONT_LB"),
        axle(rear ).alias("REAR_LB")  if rear  else pl.lit(None).alias("REAR_LB"),
    ])
    df = df.with_columns((pl.col("FRONT_LB") + pl.col("REAR_LB")).alias("TOTAL_LB"))

    m = df["TOTAL_LB"].mean() 
    mx = df["TOTAL_LB"].max() 
    mn = df["TOTAL_LB"].min()
    print(f"TOTAL lb  mean={m:.1f}  max={mx:.1f}  min={mn:.1f}  range={mx-mn:.1f}")

    t = df["Time_ms"].to_numpy() / 1000
    plt.figure(figsize=(12, 5))
    plt.plot(t, df["TOTAL_LB"].to_numpy(), label="Total")
    plt.plot(t, df["FRONT_LB"].to_numpy(), label="Front", alpha=0.7)
    plt.plot(t, df["REAR_LB"].to_numpy(),  label="Rear",  alpha=0.7)
    plt.axhline(0, color="k", lw=0.8, ls="--")
    plt.xlabel("Time [s]")
    plt.ylabel("Apparent additional load [lb]")
    plt.title(f"{run_id} — apparent vertical load delta (no-aero baseline)")
    plt.legend(); plt.grid(alpha=0.4); plt.tight_layout()
    plt.savefig(out / "downforce_vs_time.png", dpi=150)
    plt.close()

    df.write_csv(out / "output.csv")
    return {"run": run_id, "mean": m, "max": mx, "min": mn}

results = [r for r in (run(f) for f in files) if r]
print("\n=== summary ===")
for r in results:
    print(f"{r['run']}: mean={r['mean']:.1f}  max={r['max']:.1f}  min={r['min']:.1f}")