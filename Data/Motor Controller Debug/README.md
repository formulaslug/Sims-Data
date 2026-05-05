# Code 26 Voltage Dip Fault Analysis

## The Problem

FS-3 throws code 26 faults (voltage dips) randomly. Nathaniel thought it was a bad wire connection. I built a machine learning model to see what actually predicts the faults.

## What I Did

- Extracted 24,851 fault windows (30s before + 10s after each code 26 event)
- Trained XGBoost to predict: "Is this data from 5 seconds before a fault?"
- Used SHAP to understand which signals the model actually uses
- Tested across 12 scenarios to verify the findings

## Key Finding

**Motor temperature is the #1 predictor**, followed by throttle position and motor speed. 

The weird part: **Voltage drop and estimated resistance are LOWER during faults**, not higher. This proves it's not a bad wire—the connections are actually fine during faults.

**Conclusion:** The fault is probably a thermal limit or firmware limit (speed/torque cutoff), not a hardware issue.

---

## How to Use

**First time only:**
```bash
python build_model.py
```

**Then explore with the dashboard:**
```bash
python run_scenario.py
# Edit ACTIVE_SCENARIO to try different scenarios
# Opens at http://localhost:8050
```

## Files

**Scripts:**
- `build_model.py` — Loads data, trains the model
- `run_scenario.py` — Interactive dashboard (SHAP analysis)
- `build_all_scenarios.py` — Batch run all 12 scenarios

**Data:**
- `windows_dataset.parquet` — 24,851 fault windows
- `model.joblib` — Trained XGBoost classifier
- `labeled_fault_data.csv` — Actual fault rows with signal values
- `aggregated_statistics.csv` — Mean/median/min/max per signal
- `explainer_*.joblib` — Pre-computed SHAP explainers for each scenario
- `label_distribution_*.png` — Bar charts of baseline vs fault splits

## Dashboard Tabs

- **Feature Importance** — Which signals matter most
- **SHAP Summary** — Direction of each signal (red = pushes toward fault)
- **SHAP Dependence** — Pick one signal, see how it relates to faults
- **Individual Predictions** — Inspect a single fault row
- **What-If** — Change signal values, watch the prediction change

## The Evidence

1. Motor temp is #1 predictor (faults happen when hot)
2. Voltage drop is lower during faults, not higher
3. Resistance is lower during faults, not higher
4. Same pattern across all 12 scenarios (not load/time dependent)
5. Manual testing (What-If tab): changing motor temp/speed flips the prediction, changing voltage drop barely matters

---

## Next Steps (for Nathaniel)

1. Check firmware for thermal limits or speed cutoffs
2. Verify speed sensor is calibrated correctly at high RPM
3. Check if controller actively limits current when hot (explains low voltage drop)
4. Log actual motor temp during next fault event to confirm

---

## Raw Data for Inspection

Open `labeled_fault_data.csv` to see what the signals look like right before faults:
- Motor temperature: ~38°C (vs 25°C baseline)
- Motor speed: ~1800 RPM (vs 940 RPM baseline)
- Throttle position: ~900 (vs 586 baseline)
- **Bus current: ~5.5A (very light load, so not overload)**
- **Voltage drop: ~1.6V (no significant drop, so not a bad wire)**
