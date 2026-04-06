# Fourier Transform Sensor Analysis

## Meeting
- **When:** 2:00 PM
- **Topic:** Discuss Fourier transforms
- **Pre-work:** Research Fourier transforms beforehand if time allows

---

## Tasks

### Monday Morning
Generate a **graph for each relevant signal** (sensors we have data for) showing the highest-frequency activity. Graphs saved to `graphs/` folder.

### Wednesday Night
Find the **highest frequency** for each sensor.

---

## Setup
- **Sensor data location:** `fs-data` repo (private, formulaslug GitHub org)
- **Code location:** `Sims-Data/Data/FourierAnalysis/fft_analysis.py`
- **Data used:** FS-3 runs from Jan 11 2026, Jan 17 2026, Mar 16 2026, Aug/Nov 2025 (for tire temp)

---

## Results — Highest Frequency Per Sensor

| Sensor | Highest Freq (Hz) | Source File |
|---|---|---|
| Strain Gauge BR | 39.038 | 011726-11.parquet |
| Suspension Travel BR | 0.423 | 011026-22.parquet |
| Suspension Travel FR | 0.116 | 011026-23.parquet |
| APPS | 0.114 | 011026-23.parquet |
| Battery Current | 0.057 | 11222025_22.parquet |
| Brake Pressure Rear | 0.042 | 5.parquet |
| Tire Temp BL | 0.041 | 11222025_9.parquet |
| Battery Voltage | 0.035 | 011026-14.parquet |
| Wheel Speed BR | 0.035 | 011026-18.parquet |
| Wheel Speed FR | 0.034 | 011026-18.parquet |
| Cell Voltage | 0.033 | 011026-14.parquet |
| Wheel Speed FL | 0.027 | 11222025_21.parquet |
| Wheel Speed BL | 0.027 | 11222025_21.parquet |
| Suspension Travel FL | 0.017 | 11222025_8.parquet |
| Brake Pressure Front | 0.010 | 5.parquet |
| Tire Temp FL | 0.010 | 11222025_1.parquet |
| Suspension Travel BL | 0.009 | 011726-10.parquet |
| Battery Tray Temp | 0.009 | 11222025_1.parquet |
| Cell Temp | 0.008 | 11222025_1.parquet |
| Brake Temp | 0.006 | 11222025_1.parquet |

---

## Sensors With No Data
| Sensor | Reason |
|---|---|
| Tire Temp FR, BR | No data in any file — sensor likely not connected |
| Strain Gauge FL, FR, BL | No data in any file — sensor likely not connected |
| Brake Pedal Position Sensor | Not yet installed on car |
| Air Pressure and Speed | Future aero sensor, not yet installed |

---

## Why No February 2026 Data
When listing the fs-data repo folders, only January and March 2026 folders exist (`01112026`, `01172026`, `03162026`). No February folder was found — data was either not collected or not uploaded that month.

## Why Sims-Data Has No Sensor Data
Sims-Data contains only Python simulation code, model files, and documentation (`.py`, `.json`, `.md`, `.ipynb`). All real car sensor data is stored in the fs-data repo as `.parquet` files.

---

## Problems Encountered

### 1. Float32/Float64 Type Mismatch
**Problem:** When concatenating parquet files from different sessions, polars threw a `SchemaError: type Float32 is incompatible with expected type Float64`.
**Fix:** Cast all float columns to Float64 before loading each file:
```python
df_temp = df_temp.with_columns([
    pl.col(c).cast(pl.Float64) for c in df_temp.columns
    if df_temp[c].dtype in [pl.Float32, pl.Float64]
])
```

### 2. NaN Sample Rate
**Problem:** Computing sample rate by concatenating all files gave NaN because timestamps reset between sessions.
**Fix:** Compute sample rate per file individually using median of time differences within each file.

### 3. Git Authentication
**Problem:** fs-data repo is private. Git was not authenticated to the formulaslug GitHub org.
**Fix:** Generated a Personal Access Token on GitHub and used it to clone the repo.

---

## Code
See `fft_analysis.py` in this folder.
