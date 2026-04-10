# Figuring Out Fault Code 26


**The big takeaway:** Motor speed and throttle state are way more predictive than some random loose connection.

---


## What each script does

- **`build_model.py`** — Pulls data from 6 parquet files, extracts 30-second windows around each fault, adds electrical features (voltage drop, resistance stuff), trains the XGBoost model, saves it and the training data.

- **`run_scenario.py`** — Takes the pre-trained model, filters to a specific scenario (high current, hot controller, whatever), samples down to 4,500 rows so the dashboard doesn't choke, computes SHAP values, and spins up the interactive dashboard. Also spits out some charts showing the label split.

- **`build_all_scenarios.py`** — Batch processes all 12 scenarios without launching dashboards—useful if you want to just compute everything in parallel.

- **`FS-3_xgBoostMCFault.py`** — The original model (just here for reference).

## Data files

- **`windows_dataset.parquet`** — All 24,851 fault windows from the 6 parquet files. Has all the telemetry signals, labels (0=normal, 1=fault warning), time_to_fault, and which file it came from.

- **`model.joblib`** — The trained XGBoost model Nathaniel gave me, plus the feature list. Works for all scenarios.

- **`explainer_*.joblib`** — Pre-computed SHAP explainers for each scenario (like `explainer_high_current.joblib`). Just load these instead of recomputing SHAP values every time.

- **`label_distribution_*.png`** — Charts showing the split between normal vs. fault-warning for each scenario.

---

## How I set this up

### Pulling the data out
Found all the Code 26 faults in the raw parquet files and grabbed a 40-second chunk around each one (30 seconds before, 10 seconds after). Then labeled them:
- **Label=1:** Last 5 seconds before the fault (warning zone)
- **Label=0:** Normal operation (30–10 seconds before)
- **Label=-1:** Transition and post-fault stuff (ignored)

### Features I added
Built 8 electrical features to test if the intermittent-resistance thing was actually happening:
- `voltage_drop_connection` = voltage lost between battery and MC
- `est_resistance_mohm` = resistance estimate using Ohm's law
- `dBusV_dt` = how fast the MC voltage is changing (catches sudden drops)
- `busV_std_05s` = voltage noise (catches flickering)
- `busV_min_05s` = lowest voltage in a 0.5s window
- `dBusC_dt` = rate of current change
- `busC_max_05s` = peak current in a 0.5s window
- `mc_power_W` = power at the motor controller (V × I)

### Training the model
Used 24,851 rows, 241 features (cut out GPS, IMU, suspension junk), ran XGBoost with 200 trees, max depth 5, learning rate 0.05. Set base_score to 0.5 to keep SHAP happy, and used class weighting to handle the imbalance.

## Testing different scenarios
I ran the model on 5 different scenarios (got 7 more I haven't looked at yet):

| Scenario | Filter | Why |
|----------|--------|-----|
| `all` | No filter | Overall: what predicts faults? |
| `high_current` | >150A | Does high current trigger it? |
| `contactor_stable` | No contactor openings | Is contactor bounce the problem? |
| `contactor_flicker` | Some contactor openings | How does contactor behavior play in? |
| `hot_controller` | >40°C | Does heat matter? |
| `early_in_session` | First half | Early on, before heating up? |
| `late_in_session` | Second half | Later when it's hot? |

For each scenario I:
1. Filtered the dataset
2. Sampled down to 4,500 rows (kept the class ratio balanced)
3. Computed SHAP values
4. Saved the explainer
5. Launched the dashboard

---

## What the model learned

### Top 3 predictors (all 5 scenarios)
These always showed up at the top, no matter what I was looking at:

| Rank | Feature | What it's measuring |
|------|---------|---------------------|
| #1 | `SME_TRQSPD_Speed` | Motor RPM—faults happen at certain speeds |
| #2 | `ETC_STATUS_HE2` | Throttle/brake position—matters almost as much as speed |
| #3 | `busV_min_05s` | Lowest voltage in a rolling 0.5s window—there's dipping happening, but... |

**This held true across everything:** whether current was high or low, controller was hot or cool, contactor was bouncing or stable. Always the same top 3.

### Features that ranked LOW (things that didn't matter)
Stuff I thought would show up but didn't:

| Feature | Rank | Why I thought it'd matter | What actually happened |
|---------|------|--------------------------|------------------------|
| `voltage_drop_connection` | #25 | Resistance = voltage drop, right? | Nope, ranks low |
| `est_resistance_mohm` | #21 | Ohm's law should control it | Nope, not predictive |
| `contactor_closed` | Outside top 15 | Should differ between contactor scenarios | Doesn't differ |
| Temperature features | Low | Should matter more when it's hot | Same low rank everywhere |

---

## Bottom line
The model basically says:
- **I can predict a fault if I know:** the motor speed, throttle position, and whether voltage is dipping
- **I can't predict a fault from:** how much resistance is there, whether the contactor's bouncing, how hot the controller is, or how late in the session we are

---

## Back to the original hypothesis
Nathaniel's (or Luca's?) theory was: *"loose cable/corroded terminal/bad contactor = resistance = voltage drop at high current = fault."*

What the data actually shows: *"Speed and throttle are the real predictors. Voltage dips show up but resistance as the main cause doesn't rank high enough."*

### What I think is happening
There's probably some resistance somewhere, but it only triggers the fault when:
1. Motor's spinning at a certain RPM range (typically >3,800–7,000)
2. Throttle's at a specific position (that ETC_STATUS_HE2 value)
3. At that combo, even a small resistance (20–50mΩ) makes the voltage sag enough to trip the ~52V threshold

**The real thing:** The model learned speed + throttle matter more than resistance itself. This feels like a **demand mismatch**: at high speed/throttle, the motor controller asks for way more current than the battery can actually deliver smoothly, voltage sags, and boom—if there's any resistance in the path it gets worse.

---

## Example label split (early in session)
```
Total rows: 4,500
Normal operation: 3,595 (79.9%)
Warning zone: 905 (20.1%)
```

So out of 4,500 windows, 80% looked normal (30–10 sec before fault) and 20% showed warning signs (last 5 sec before fault). That 4:1 ratio is pretty good for training.

---

## Using the dashboard
1. **Feature Importance** — See which signals matter for each scenario
2. **SHAP Summary** — Red = pushes toward fault, blue = pushes away
3. **SHAP Dependence** — Check if voltage drop actually has a threshold
4. **Individual Predictions** — Look at a row ~10 seconds before a fault fires to see the buildup
5. **What-If** — Tweak signals manually and see how predictions change

---

## If we need to dig into the hardware

### Step 1: Log the actual fault (15–30 min)
Before we take anything apart, just record what's happening when the fault fires.

- [ ] Capture: RPM, throttle (HE2), battery voltage, MC voltage, current
- [ ] Drive until fault happens
- [ ] Look at:
  - Does battery voltage drop? (battery cell sag, not a connector problem)
  - MC voltage drop while battery stays up? (connection resistance)
  - RPM jumps around? (speed sensor noise)
  - Throttle goes weird? (control/firmware thing)

### Step 2: If it looks like a connector problem
Only do this if Step 1 shows stable battery but MC voltage tanking.

- [ ] Battery terminals: corrosion, loose bolts, cracked lugs
- [ ] Main contactor: pitting/arcing, test it closes tight (<0.1Ω)
- [ ] MC terminals (B+/B−): tight bolts, heat damage
- [ ] Fuses/bus bars: heat marks, loose connections, measure resistance

### Step 3: If it looks like a control/firmware problem
Only if Step 1 shows stable voltage but sketchy RPM or throttle signals.

- [ ] Speed sensor: check the connector, look for noise in the raw RPM signal
- [ ] Throttle/ETC sensor: what's the HE2 value when the fault fires? connector look ok?
- [ ] MC firmware: what's the max power limit? can the battery actually deliver that?

---


---

## More scenarios to test
Got 7 more I haven't run yet:
- `very_high_current` (>300A)
- `high_torque` (>50%)
- `large_v_drop` (>5V)
- `autox_files` (just the fast runs)
- `rolling_resistance` (the rolling resistance test sessions)

---

## What the SME manual says (Code 26 = Under Voltage)
The official fault list says Code 26 can be caused by:
- Battery damaged/depleted
- **Battery resistance too high** ← what we've been thinking
- Battery disconnected
- Bad key-switch fuse
- External load drawing power

Basically: voltage drops below ~52V and the controller throws a fit. Goes away once you get the voltage back up.

---

## The numbers (stat breakdown)
Ran stats on all 240 features. Here's the weird stuff that jumped out:

| Feature | Normal | Fault | Change |
|---------|--------|-------|--------|
| `SME_TRQSPD_Speed` | 940 RPM | 1,824 RPM | 2x faster |
| `ETC_STATUS_HE2` | 586 | 902 | 54% higher |
| `SME_THROTL_TorqueDemand` | 2,225 | 7,564 | 3.4x higher |
| `voltage_drop_connection` | 21.3V | 1.6V | **drops to almost nothing** |
| `est_resistance_mohm` | 6,463 mΩ | 22.5 mΩ | **goes to nearly zero** |
| `SME_TEMP_BusCurrent` | 698A | 103A | drops 85% |
| `mc_power_W` | 31,446W | 9,860W | drops 69% |

**The weird part:** If resistance was the problem, faults should happen when resistance is HIGH and voltage drop is BIG. But the data shows the opposite—faults happen when both are LOW. This doesn't fit the "loose connection" theory. Instead it looks like the motor doing something (high speed, high torque demand) is what triggers it.

Full stats for all 240 features are in `aggregated_statistics.csv`.

---

## TL;DR

The model says faults are **speed and throttle dependent**, not just "voltage is low." The intermittent-resistance theory might be real, but it doesn't rank as the top predictor. The data actually shows the opposite of what you'd expect if loose connections were the problem.

The pattern: **high speed + high torque demand = fault conditions.** Even though the estimated resistance and voltage drop are actually LOW during these fault windows. That's backwards from the "loose cable" hypothesis.

Most likely what's happening:
1. At certain speeds/throttle combos, the motor controller asks for more power than the system can smoothly deliver
2. Voltage sags slightly
3. If there IS resistance somewhere (and there probably is), it makes things worse
4. Boom—under-voltage fault

**Next steps:**
1. Log actual sensor data when the fault fires (shows us if it's resistance, firmware, or sensor noise)
2. If it looks like resistance: check cables, terminals, contactor
3. If voltage is fine but speeds/throttle look weird: firmware/sensor issue