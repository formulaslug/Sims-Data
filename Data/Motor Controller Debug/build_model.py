"""

for others using this, run it once. it:
  1. loads all parquet files that contain fault code 26 events
  2. extracts 30s-before / 10s-after windows around each fault onset
  3. adds derived electrical features (voltage drop, resistance estimate, etc.)
  4. trains XGBoost on those windows
  5. saves the trained model + feature dataset to model_data.parquet + model.joblib

after this, just run run_scenario.py to explore different scenarios in the dashboard.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from xgboost import XGBClassifier
from explainerdashboard import ClassifierExplainer
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent / "gitrepos" / "Sims-Data-2"))
from Data.FSLib.AnalysisFunctions import *

# ─── File paths ───────────────────────────────────────────────────────────────
# only files confirmed to have fault code 26 events
FAULT_FILES = [
    "../fs-data/FS-3/08172025/08172025_22_6LapsAndWeirdCurrData.parquet",   # @ 518538, 560592, 685456
    "../fs-data/FS-3/08102025/08102025RollingResistanceTestP3.parquet",     # @ 111263
    "../fs-data/FS-3/08102025/08102025RollingResistanceTestP1.parquet",     # @ 201624
    "../fs-data/FS-3/08102025/08102025RollingResistanceTestP4.parquet",     # @ 33955
    "../fs-data/FS-3/08172025/08172025_26autox1.parquet",                        # @ 123328, 313745
    "../fs-data/FS-3/10082025/fixed_wheels_nathaniel_inv_test_w_fault.parquet",  # @ 320629
    "../fs-data/FS-3/10112025/firstDriveMCError30.parquet",                      # referenced in original script
]

SEARCH_BASES = [
    Path("C:/Users/Mid0r/Downloads/fs-data-main/fs-data-main"),
    Path("C:/Users/Mid0r/updatedfsdata/fs-data"),
    Path("C:/Projects/FormulaSlug/fs-data"),
]

# window config
WINDOW_BEFORE_S = 30    # seconds of data before each fault onset to keep
WINDOW_AFTER_S  = 10    # seconds after
LABEL_POS_START = -5    # rows in this range → Label=1 (imminent fault)
LABEL_NEG_END   = -10   # rows before this   → Label=0 (baseline / normal)

# columns 
BUS_V      = "SME_TEMP_DC_Bus_V"
BUS_C      = "SME_TEMP_BusCurrent"
TS_V       = "TS_Voltage"
FAULT_CODE = "SME_TEMP_FaultCode"
CONTACTOR  = "SME_TRQSPD_contactor_closed"
MC_TEMP    = "SME_TEMP_ControllerTemperature"
TORQUE     = "SME_THROTL_TorqueDemand"

DROP_COLS = [
    # Target leakage
    "SME_TEMP_FaultCode", "SME_TEMP_FaultLevel", "SME_TEMP_DC_Bus_V", "Seconds", "index",
    # GPS
    "VDM_GPS_Latitude", "VDM_GPS_Longitude", "VDM_GPS_TRUE_COURSE",
    "VDM_GPS_SPEED", "VDM_GPS_VALID1", "VDM_GPS_UTC_TIME",
    # IMU
    "VDM_X_AXIS_ACCELERATION", "VDM_Y_AXIS_ACCELERATION", "VDM_Z_AXIS_ACCELERATION",
    "VDM_X_AXIS_YAW_RATE", "VDM_Y_AXIS_YAW_RATE", "VDM_Z_AXIS_YAW_RATE",
    "IMU_XAxis_Acceleration_mps", "IMU_YAxis_Acceleration_mps", "IMU_ZAxis_Acceleration_mps",
    "xA", "yA", "zA", "vA",
    # suspension
    "TELEM_FR_SUSTRAVEL", "TELEM_FL_SUSTRAVEL", "TELEM_BR_SUSTRAVEL", "TELEM_BL_SUSTRAVEL",
    # Wheel speeds
    "TPERIPH_FL_DATA_WHEELSPEED", "TPERIPH_FR_DATA_WHEELSPEED",
    "TPERIPH_BL_DATA_WHEELSPEED", "TPERIPH_BR_DATA_WHEELSPEED",
    # driver inputs
    "ETC_STATUS_PEDAL_TRAVEL", "ETC_STATUS_BRAKE_SENSE_VOLTAGE",
    "TMAIN_DATA_BRAKES_F", "TMAIN_DATA_BRAKES_R",
    # motor flags (lowkey too vague to be useful, and can be noisy / leaky)
    "SME_TRQSPD_MotorFlags",
]


# stuff
def resolve(path_str):
    p = Path(path_str)
    if p.exists():
        return p
    parts = p.parts
    for n in (3, 2):
        tail = Path(*parts[-n:])
        for base in SEARCH_BASES:
            c = base / tail
            if c.exists():
                return c
    return None


def sample_hz(df):
    if "Time" not in df.columns or len(df) < 10:
        return 100.0
    times = df["Time"].cast(pl.Float64).drop_nulls()
    dt = float((times[-1] - times[0]) / (len(times) - 1))
    return 1.0 / dt if dt > 0 else 100.0


def add_derived(df, hz):
    """Add features that directly proxy the intermittent-resistance theory."""
    w = max(2, int(0.5 * hz))
    exprs = []

    bat_col = next((c for c in (TS_V, "ACC_POWER_PACK_VOLTAGE") if c in df.columns), None)

    if BUS_V in df.columns and bat_col:
        exprs.append((pl.col(bat_col) - pl.col(BUS_V)).alias("voltage_drop_connection"))
        if BUS_C in df.columns:
            exprs.append(
                pl.when(pl.col(BUS_C).abs() > 1.0)
                  .then((pl.col(bat_col) - pl.col(BUS_V)) / pl.col(BUS_C) * 1000)
                  .otherwise(0.0)
                  .alias("est_resistance_mohm")
            )

    if BUS_V in df.columns:
        exprs += [
            pl.col(BUS_V).diff(1).alias("dBusV_dt"),
            pl.col(BUS_V).rolling_std(window_size=w).alias("busV_std_05s"),
            pl.col(BUS_V).rolling_min(window_size=w).alias("busV_min_05s"),
        ]

    if BUS_C in df.columns:
        exprs += [
            pl.col(BUS_C).diff(1).alias("dBusC_dt"),
            pl.col(BUS_C).rolling_max(window_size=w).alias("busC_max_05s"),
        ]

    if BUS_V in df.columns and BUS_C in df.columns:
        exprs.append((pl.col(BUS_V) * pl.col(BUS_C)).alias("mc_power_W"))

    if exprs:
        df = df.with_columns(exprs)
    return df


def extract_windows(df, hz, source_file):
    """Cut out [-WINDOW_BEFORE_S, +WINDOW_AFTER_S] around each fault code 26 onset."""
    if FAULT_CODE not in df.columns:
        return None

    fault_idxs = (df[FAULT_CODE] == 26).arg_true().to_list()
    if not fault_idxs:
        return None

    # Only keep first index of each sustained burst (gap > 3s)
    min_gap = int(3 * hz)
    onsets = [fault_idxs[0]]
    for idx in fault_idxs[1:]:
        if idx - onsets[-1] > min_gap:
            onsets.append(idx)

    print(f"  {len(onsets)} fault onset(s) at row(s): {onsets}")

    before = int(WINDOW_BEFORE_S * hz)
    after  = int(WINDOW_AFTER_S * hz)
    chunks = []

    for fi in onsets:
        start = max(0, fi - before)
        end   = min(len(df), fi + after)
        chunk = df.slice(start, end - start)

        offsets = np.arange(start - fi, end - fi, dtype=np.float64)
        t_to_fault = offsets / hz

        labels = np.where(
            (t_to_fault >= LABEL_POS_START) & (t_to_fault < 0), 1,
            np.where(
                (t_to_fault >= -WINDOW_BEFORE_S) & (t_to_fault < LABEL_NEG_END), 0,
                -1,
            ),
        ).astype(np.int32)

        chunk = chunk.with_columns([
            pl.Series("time_to_fault", t_to_fault),
            pl.Series("Label", labels),
            pl.lit(source_file).alias("source_file"),
        ])
        chunks.append(chunk)

    return pl.concat(chunks, how="diagonal_relaxed")


# ─── Main ─────────────────────────────────────────────────────────────────────
print("=" * 65)
print("build_model.py — building fault window dataset + XGBoost model")
print("=" * 65)

all_windows = []

for fpath in FAULT_FILES:
    print(f"\n→ {fpath}")
    p = resolve(fpath)
    if p is None:
        print("  NOT FOUND — skipping")
        continue

    df = pl.read_parquet(p)
    df = df.with_columns(pl.all().fill_null(strategy="forward"))
    df = df.with_columns(pl.all().fill_null(strategy="backward"))
    df = df.rename({c: c.replace(".", "_") for c in df.columns if "." in c})

    hz = sample_hz(df)
    print(f"  {len(df):,} rows  ~{hz:.0f} Hz")

    df = add_derived(df, hz)
    windows = extract_windows(df, hz, source_file=p.name)

    if windows is None:
        print("  No fault code 26 events — skipping")
        continue

    n1 = int((windows["Label"] == 1).sum())
    n0 = int((windows["Label"] == 0).sum())
    print(f"  Windows: label=1 (imminent): {n1:,}  |  label=0 (baseline): {n0:,}")
    all_windows.append(windows)

if not all_windows:
    sys.exit("\nNo fault windows found. Check that the parquet file paths resolve correctly.")

combined = pl.concat(all_windows, how="diagonal_relaxed")
print(f"\nTotal rows across all files: {len(combined):,}")

# save the full windowed dataset (including time_to_fault, source_file, Label)
# so run_scenario.py can filter it without re-loading everything
combined.write_parquet("windows_dataset.parquet")
print("Saved: windows_dataset.parquet")

# build training set for XGBoost
labeled = combined.filter(pl.col("Label") >= 0)

actual_drops = [c for c in DROP_COLS + ["Time", "time_to_fault", "source_file"] if c in labeled.columns]
feature_df = (
    labeled.drop(*actual_drops)
    .with_columns(pl.col(pl.Boolean).cast(pl.Int32))
    .with_columns(pl.all().fill_null(0))
)
labels_s   = labeled["Label"]

X = feature_df.to_pandas()
y = labels_s.to_pandas()

print(f"\nTraining set: {len(X):,} rows  |  {X.shape[1]} features")
print(f"  Positive (imminent): {int((y==1).sum()):,}  |  Negative (baseline): {int((y==0).sum()):,}")

pos_weight = max(1, int((y == 0).sum() / max((y == 1).sum(), 1)))
print(f"  scale_pos_weight = {pos_weight}")

bst = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    base_score=0.5,
    scale_pos_weight=pos_weight,
    eval_metric="logloss",
    random_state=42,
)
bst.fit(X, y)

# save model + feature columns so run_scenario.py can reconstruct X without re-fitting
joblib.dump({"model": bst, "feature_cols": list(X.columns)}, "model.joblib")
print("\nSaved: model.joblib")

# quick importance printout
importances = pd.Series(bst.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 15 features:")
for i, (feat, score) in enumerate(importances.head(15).items(), 1):
    print(f"  {i:2d}. {feat:<50s} {score:.4f}")

print("\nDone. Now run:  python run_scenario.py")
