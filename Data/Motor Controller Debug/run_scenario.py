"""
run_scenario.py

Change ACTIVE_SCENARIO below to switch between scenarios, then run this file.
It loads the pre-built model + windowed dataset, filters to your scenario,
samples down to MAX_ROWS, and opens the explainer dashboard at localhost:8050.

Run build_model.py first if you don't have model.joblib and windows_dataset.parquet yet.
"""

import joblib
import json
import os
import tempfile
import numpy as np
import pandas as pd
import polars as pl
from xgboost import XGBClassifier
from explainerdashboard import ClassifierExplainer, ExplainerDashboard

# ─── Change this to switch scenarios ─────────────────────────────────────────
ACTIVE_SCENARIO = "early_in_session"  
# Options:
#   "all"               — everything, no filter (baseline view)
#   "high_current"      — only windows where peak current > 150A  (load-dependent fault?)
#   "very_high_current" — only windows where peak current > 300A  (extreme load events)
#   "high_torque"       — only windows where torque demand > 50%
#   "contactor_stable"  — contactor stayed closed the whole window (rules out bounce)
#   "contactor_flicker" — contactor changed state at some point in the window
#   "hot_controller"    — MC temp was above 40°C at some point in window
#   "large_v_drop"      — voltage_drop_connection > 5V at some point (resistance already visible)
#   "early_in_session"  — first half of each file's time range (before things heat up)
#   "late_in_session"   — second half of each file's time range (after things heat up)
#   "autox_files"       — only autox / high-performance run files
#   "rolling_resistance"— only rolling resistance test files

MAX_ROWS = 4500   # keep under 5000 for dashboard performance

# ─── Column names ─────────────────────────────────────────────────────────────
BUS_C      = "SME_TEMP_BusCurrent"
TORQUE     = "SME_THROTL_TorqueDemand"
CONTACTOR  = "SME_TRQSPD_contactor_closed"
MC_TEMP    = "SME_TEMP_ControllerTemperature"
V_DROP     = "voltage_drop_connection"

# Columns to strip before building the explainer (metadata, not features)
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


# ─── Scenario definitions ─────────────────────────────────────────────────────
# Each scenario is a function that takes the full labeled polars DataFrame
# and returns a filtered version. Use pl.col() expressions.

def scenario_all(df):
    return df

def scenario_high_current(df):
    if BUS_C not in df.columns:
        print(f"  WARNING: {BUS_C} not in dataset — returning all rows")
        return df
    # Keep windows where current exceeded 150A at any point
    max_current = df.group_by("source_file").agg(pl.col(BUS_C).max().alias("max_c"))
    # Per-row filter: keep rows whose source file had high current
    return df.filter(pl.col(BUS_C) > 150)

def scenario_very_high_current(df):
    if BUS_C not in df.columns:
        return df
    return df.filter(pl.col(BUS_C) > 300)

def scenario_high_torque(df):
    if TORQUE not in df.columns:
        print(f"  WARNING: {TORQUE} not in dataset — returning all rows")
        return df
    return df.filter(pl.col(TORQUE) > 50)

def scenario_contactor_stable(df):
    if CONTACTOR not in df.columns:
        print(f"  WARNING: {CONTACTOR} not in dataset — returning all rows")
        return df
    return df.filter(pl.col(CONTACTOR) == 1)

def scenario_contactor_flicker(df):
    if CONTACTOR not in df.columns:
        return df
    # Keep rows where the contactor is NOT continuously closed
    # (i.e., at least one 0 somewhere in that window)
    flicker_files = (
        df.filter(pl.col(CONTACTOR) == 0)
          .select("source_file")
          .unique()
    )
    return df.join(flicker_files, on="source_file", how="inner")

def scenario_hot_controller(df):
    if MC_TEMP not in df.columns:
        print(f"  WARNING: {MC_TEMP} not in dataset — returning all rows")
        return df
    return df.filter(pl.col(MC_TEMP) > 40)

def scenario_large_v_drop(df):
    if V_DROP not in df.columns:
        print(f"  WARNING: {V_DROP} not in dataset — returning all rows")
        return df
    return df.filter(pl.col(V_DROP) > 5)

def scenario_early_in_session(df):
    if "Time" not in df.columns:
        return df
    midpoint = df["Time"].cast(pl.Float64).mean()
    return df.filter(pl.col("Time").cast(pl.Float64) < midpoint)

def scenario_late_in_session(df):
    if "Time" not in df.columns:
        return df
    midpoint = df["Time"].cast(pl.Float64).mean()
    return df.filter(pl.col("Time").cast(pl.Float64) >= midpoint)

def scenario_autox_files(df):
    return df.filter(pl.col("source_file").str.contains("autox"))

def scenario_rolling_resistance(df):
    return df.filter(pl.col("source_file").str.contains("Rolling"))

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


# ─── Load data + model ────────────────────────────────────────────────────────
print("=" * 65)
print(f"run_scenario.py  —  scenario: '{ACTIVE_SCENARIO}'")
print("=" * 65)

print("Loading windows_dataset.parquet ...")
dataset = pl.read_parquet("windows_dataset.parquet")

print("Loading model.joblib ...")
model_bundle = joblib.load("model.joblib")
bst          = model_bundle["model"]

# XGBoost serializes base_score as '[5E-1]' which shap can't parse as float.
# Fix: save to JSON, strip the brackets, reload.
_fd, _tmp = tempfile.mkstemp(suffix=".json")
os.close(_fd)
bst.save_model(_tmp)
with open(_tmp, "r") as f:
    _mj = json.load(f)
_bs = _mj["learner"]["learner_model_param"]["base_score"]
_mj["learner"]["learner_model_param"]["base_score"] = str(float(_bs.strip("[]")))
with open(_tmp, "w") as f:
    json.dump(_mj, f)
bst = XGBClassifier()
bst.load_model(_tmp)
os.unlink(_tmp)

feature_cols = model_bundle["feature_cols"]

# ─── Apply scenario filter ────────────────────────────────────────────────────
if ACTIVE_SCENARIO not in SCENARIOS:
    raise ValueError(f"Unknown scenario '{ACTIVE_SCENARIO}'. Options: {list(SCENARIOS.keys())}")

labeled = dataset.filter(pl.col("Label") >= 0)
filtered = SCENARIOS[ACTIVE_SCENARIO](labeled)

print(f"\nRows after scenario filter: {len(filtered):,}")
print(f"  Positive (imminent fault): {int((filtered['Label']==1).sum()):,}")
print(f"  Negative (baseline):       {int((filtered['Label']==0).sum()):,}")

if len(filtered) == 0:
    raise ValueError("Scenario filter returned 0 rows. Try a different scenario or check column names.")

# ─── Sample down to MAX_ROWS (stratified) ─────────────────────────────────────
def stratified_sample(df: pl.DataFrame, max_rows: int) -> pl.DataFrame:
    if len(df) <= max_rows:
        return df
    pos = df.filter(pl.col("Label") == 1)
    neg = df.filter(pl.col("Label") == 0)
    n_pos = max(1, int(max_rows * len(pos) / len(df)))
    n_neg = max_rows - n_pos
    pos_s = pos.sample(min(n_pos, len(pos)), seed=42)
    neg_s = neg.sample(min(n_neg, len(neg)), seed=42)
    return pl.concat([pos_s, neg_s]).sample(fraction=1.0, seed=42)  # shuffle

sampled = stratified_sample(filtered, MAX_ROWS)
print(f"\nSampled to {len(sampled):,} rows (max {MAX_ROWS})")
print(f"  Positive: {int((sampled['Label']==1).sum()):,}  |  Negative: {int((sampled['Label']==0).sum()):,}")

# ─── Build feature matrix ─────────────────────────────────────────────────────
all_drops = set(META_COLS + DROP_COLS)
feature_df = sampled.drop(*[c for c in all_drops if c in sampled.columns])
feature_df = (
    feature_df
    .with_columns(pl.col(pl.Boolean).cast(pl.Int32))
    .with_columns(pl.all().fill_null(0))
)

# Align columns to what the model was trained on
X = feature_df.to_pandas()
y = sampled["Label"].to_pandas()

# Add any missing columns the model expects (filled with 0)
for col in feature_cols:
    if col not in X.columns:
        X[col] = 0.0

# Keep only columns the model knows (in the same order)
X = X[feature_cols]

print(f"\nFeature matrix: {X.shape[0]} rows × {X.shape[1]} columns")

# ─── Label Distribution Visualization ─────────────────────────────────────────
import matplotlib.pyplot as plt

label_counts = sampled['Label'].value_counts().sort('Label')
label_0_count = int((sampled['Label'] == 0).sum())
label_1_count = int((sampled['Label'] == 1).sum())
total = label_0_count + label_1_count

print(f"\n=== LABELED DATA DISTRIBUTION ===")
print(f"Total rows in scenario: {total:,}")
print(f"Label=0 (baseline): {label_0_count:,} ({100*label_0_count/total:.1f}%)")
print(f"Label=1 (imminent fault): {label_1_count:,} ({100*label_1_count/total:.1f}%)")

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(['Baseline\n(Label=0)', 'Imminent Fault\n(Label=1)'],
              [label_0_count, label_1_count],
              color=['#2ecc71', '#e74c3c'],
              edgecolor='black',
              linewidth=1.5)
ax.set_ylabel('Number of Rows', fontsize=12, fontweight='bold')
ax.set_title(f'Label Distribution — {ACTIVE_SCENARIO}\n{total:,} total rows', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(label_0_count, label_1_count) * 1.15)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}\n({100*height/total:.1f}%)',
            ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(f'label_distribution_{ACTIVE_SCENARIO}.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: label_distribution_{ACTIVE_SCENARIO}.png")
plt.show()

# ─── Build explainer + launch dashboard ───────────────────────────────────────
print("\nBuilding explainer (SHAP values — may take 30–60 seconds) ...")
explainer = ClassifierExplainer(
    bst, X, y,
    model_output="logodds",
    labels=["Baseline", "Imminent Fault"],
    descriptions={
        "voltage_drop_connection": "V_battery − V_mc: voltage lost between battery and MC. High = bad connection.",
        "est_resistance_mohm":     "Estimated resistance of the connection in milliohms (Ohm's law: V_drop / I × 1000).",
        "busV_std_05s":            "Rolling std dev of MC voltage over 0.5s. High = voltage flickering.",
        "dBusV_dt":                "Rate of change of MC voltage. Large negative = sudden voltage collapse.",
        "busC_max_05s":            "Rolling peak current over 0.5s. High = heavy load on the connection.",
        "dBusC_dt":                "Rate of change of current. Positive spike = controller drawing extra current.",
        "mc_power_W":              "Power at MC terminals (V × I). High power + fault = load-dependent issue.",
        "busV_min_05s":            "Rolling minimum voltage over 0.5s. Captures momentary dips.",
    }
)

explainer.dump(f"explainer_{ACTIVE_SCENARIO}.joblib")
print(f"Saved: explainer_{ACTIVE_SCENARIO}.joblib")

print(f"\nLaunching dashboard at http://127.0.0.1:8050")
print(f"Scenario: '{ACTIVE_SCENARIO}'  |  {len(X)} rows  |  {int((y==1).sum())} fault / {int((y==0).sum())} baseline")
print("─" * 65)
print("TIPS:")
print("  Feature Importance tab  — which signals matter most overall")
print("  SHAP Summary tab        — direction: red=high value, blue=low")
print("  SHAP Dependence tab     — pick 'voltage_drop_connection' to find threshold")
print("  Individual Predictions  — pick a row at time_to_fault ≈ -10s to see buildup")
print("  What-If tab             — manually adjust signal values, watch prediction change")
print("─" * 65)

db = ExplainerDashboard(explainer)
db.run(host="127.0.0.1", port=8050, use_waitress=True)
