"""

Builds and saves an explainer joblib for every scenario without launching
any dashboards. Run this once and all files will be ready for testingjoblib.py.

Requires model.joblib and windows_dataset.parquet (run build_model.py first).
"""

import joblib
import json
import os
import tempfile
import numpy as np
import pandas as pd
import polars as pl
from xgboost import XGBClassifier
from explainerdashboard import ClassifierExplainer

# ─── Config ───────────────────────────────────────────────────────────────────
MAX_ROWS = 4500 # max rows per scenario so it is manageable in the dashboard 

# skip any that already exist so that it can re-run without rebuilding everything
SKIP_EXISTING = True

# column names
BUS_C     = "SME_TEMP_BusCurrent"
TORQUE    = "SME_THROTL_TorqueDemand"
CONTACTOR = "SME_TRQSPD_contactor_closed"
MC_TEMP   = "SME_TEMP_ControllerTemperature"
V_DROP    = "voltage_drop_connection"

META_COLS = ["Label", "Time", "time_to_fault", "source_file",
             "SME_TEMP_FaultCode", "SME_TEMP_FaultLevel", "SME_TEMP_DC_Bus_V",
             "Seconds", "index"]

DROP_COLS = [
    "VDM_GPS_Latitude", "VDM_GPS_Longitude", "VDM_GPS_TRUE_COURSE",
    "VDM_GPS_SPEED", "VDM_GPS_VALID1", "VDM_GPS_UTC_TIME",
    "VDM_X_AXIS_ACCELERATION", "VDM_Y_AXIS_ACCELERATION", "VDM_Z_AXIS_ACCELERATION",
    "VDM_X_AXIS_YAW_RATE", "VDM_Y_AXIS_YAW_RATE", "VDM_Z_AXIS_YAW_RATE",
    "IMU_XAxis_Acceleration_mps", "IMU_YAxis_Acceleration_mps", "IMU_ZAxis_Acceleration_mps",
    "xA", "yA", "zA", "vA",
    "TELEM_FR_SUSTRAVEL", "TELEM_FL_SUSTRAVEL", "TELEM_BR_SUSTRAVEL", "TELEM_BL_SUSTRAVEL",
    "TPERIPH_FL_DATA_WHEELSPEED", "TPERIPH_FR_DATA_WHEELSPEED",
    "TPERIPH_BL_DATA_WHEELSPEED", "TPERIPH_BR_DATA_WHEELSPEED",
    "ETC_STATUS_PEDAL_TRAVEL", "ETC_STATUS_BRAKE_SENSE_VOLTAGE",
    "TMAIN_DATA_BRAKES_F", "TMAIN_DATA_BRAKES_R",
    "SME_TRQSPD_MotorFlags",
]

# scenario filters 
def scenario_all(df):              return df
def scenario_high_current(df):     return df.filter(pl.col(BUS_C) > 150)      if BUS_C in df.columns else df
def scenario_very_high_current(df):return df.filter(pl.col(BUS_C) > 300)      if BUS_C in df.columns else df
def scenario_high_torque(df):      return df.filter(pl.col(TORQUE) > 50)      if TORQUE in df.columns else df
def scenario_contactor_stable(df): return df.filter(pl.col(CONTACTOR) == 1)   if CONTACTOR in df.columns else df
def scenario_hot_controller(df):   return df.filter(pl.col(MC_TEMP) > 40)     if MC_TEMP in df.columns else df
def scenario_large_v_drop(df):     return df.filter(pl.col(V_DROP) > 5)       if V_DROP in df.columns else df
def scenario_early_in_session(df):
    if "Time" not in df.columns: return df
    mid = df["Time"].cast(pl.Float64).mean()
    return df.filter(pl.col("Time").cast(pl.Float64) < mid)
def scenario_late_in_session(df):
    if "Time" not in df.columns: return df
    mid = df["Time"].cast(pl.Float64).mean()
    return df.filter(pl.col("Time").cast(pl.Float64) >= mid)
def scenario_contactor_flicker(df):
    if CONTACTOR not in df.columns: return df
    flicker_files = df.filter(pl.col(CONTACTOR) == 0).select("source_file").unique()
    return df.join(flicker_files, on="source_file", how="inner")
def scenario_autox_files(df):       return df.filter(pl.col("source_file").str.contains("autox"))
def scenario_rolling_resistance(df):return df.filter(pl.col("source_file").str.contains("Rolling"))

SCENARIOS = {
    "all":                scenario_all,
    "high_current":       scenario_high_current,
    "very_high_current":  scenario_very_high_current,
    "high_torque":        scenario_high_torque,
    "contactor_stable":   scenario_contactor_stable,
    "contactor_flicker":  scenario_contactor_flicker,
    "hot_controller":     scenario_hot_controller,
    "large_v_drop":       scenario_large_v_drop,
    "early_in_session":   scenario_early_in_session,
    "late_in_session":    scenario_late_in_session,
    "autox_files":        scenario_autox_files,
    "rolling_resistance": scenario_rolling_resistance,
}

#  
def fix_base_score(bst):
    """Save/reload via JSON to strip XGBoost's bracket format from base_score."""
    _fd, _tmp = tempfile.mkstemp(suffix=".json")
    os.close(_fd)
    bst.save_model(_tmp)
    with open(_tmp, "r") as f:
        mj = json.load(f)
    bs = mj["learner"]["learner_model_param"]["base_score"]
    mj["learner"]["learner_model_param"]["base_score"] = str(float(bs.strip("[]")))
    with open(_tmp, "w") as f:
        json.dump(mj, f)
    fixed = XGBClassifier()
    fixed.load_model(_tmp)
    os.unlink(_tmp)
    return fixed

def stratified_sample(df, max_rows):
    if len(df) <= max_rows:
        return df
    pos = df.filter(pl.col("Label") == 1)
    neg = df.filter(pl.col("Label") == 0)
    n_pos = max(1, int(max_rows * len(pos) / len(df)))
    n_neg = max_rows - n_pos
    pos_s = pos.sample(min(n_pos, len(pos)), seed=42)
    neg_s = neg.sample(min(n_neg, len(neg)), seed=42)
    return pl.concat([pos_s, neg_s]).sample(fraction=1.0, seed=42)

#  load shared data once 
print("=" * 65)
print("build_all_scenarios.py — building all scenario explainers")
print("=" * 65)

print("Loading windows_dataset.parquet ...")
dataset = pl.read_parquet("windows_dataset.parquet")
labeled = dataset.filter(pl.col("Label") >= 0)

print("Loading and fixing model.joblib ...")
bundle       = joblib.load("model.joblib")
bst          = fix_base_score(bundle["model"])
feature_cols = bundle["feature_cols"]

all_drops = set(META_COLS + DROP_COLS)

# build and save explainers for each scenario
results = []

for name, filter_fn in SCENARIOS.items():
    out_file = f"explainer_{name}.joblib"

    if SKIP_EXISTING and os.path.exists(out_file):
        print(f"\n[SKIP] {out_file} already exists")
        results.append((name, "skipped", "-"))
        continue

    print(f"\n{'─'*65}")
    print(f"Scenario: {name}")

    filtered = filter_fn(labeled)
    if len(filtered) == 0:
        print(f"  [SKIP] 0 rows after filter")
        results.append((name, "skipped — 0 rows", "-"))
        continue

    n1 = int((filtered["Label"] == 1).sum())
    n0 = int((filtered["Label"] == 0).sum())
    print(f"  {len(filtered):,} rows  |  fault: {n1:,}  baseline: {n0:,}")

    if n1 == 0 or n0 == 0:
        print(f"  [SKIP] need both classes present")
        results.append((name, "skipped — missing class", "-"))
        continue

    sampled = stratified_sample(filtered, MAX_ROWS)

    feature_df = sampled.drop(*[c for c in all_drops if c in sampled.columns])
    feature_df = (
        feature_df
        .with_columns(pl.col(pl.Boolean).cast(pl.Int32))
        .with_columns(pl.all().fill_null(0))
    )

    X = feature_df.to_pandas()
    y = sampled["Label"].to_pandas()

    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_cols]

    print(f"  Building explainer ({len(X):,} rows × {X.shape[1]} cols) ...")
    explainer = ClassifierExplainer(
        bst, X, y,
        model_output="logodds",
        labels=["Baseline", "Imminent Fault"],
        descriptions={
            "voltage_drop_connection": "V_battery − V_mc: voltage lost between battery and MC.",
            "est_resistance_mohm":     "Estimated connection resistance in milliohms (V_drop / I × 1000).",
            "busV_std_05s":            "Rolling voltage variability over 0.5s. High = flickering.",
            "dBusV_dt":                "Rate of MC voltage change. Large negative = sudden collapse.",
            "busC_max_05s":            "Rolling peak current over 0.5s.",
            "dBusC_dt":                "Rate of current change. Spike = controller compensating.",
            "mc_power_W":              "Power at MC terminals (V × I).",
            "busV_min_05s":            "Rolling minimum voltage over 0.5s.",
        }
    )
    explainer.dump(out_file)
    print(f"  Saved: {out_file}")
    results.append((name, "done", out_file))

# summary 
print(f"\n{'='*65}")
print("Done. Summary:")
print(f"{'='*65}")
for name, status, fname in results:
    print(f"  {name:<22s}  {status}")

print(f"\nTo view any scenario, set LOAD_FILE in testingjoblib.py and run:")
print("  python testingjoblib.py")
