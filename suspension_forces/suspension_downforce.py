import pandas as pd
import polars as pl
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
import os

damper_params = {
    "FL": (5,    3,    2,    1.5),
    "FR": (5,    3.33, 1,    3.33),
    "BL": (7,    3,    3,    3),
    "BR": (7,    2.66, 3.33, 3),
}

preload_lb = {
    "FL": 0.0,
    "FR": 0.0,
    "BL": 0.0,
    "BR": 0.0,
}

spring_rate = 200  # lbs per in

##luca's code basically imported
highspeed_curves = pl.read_csv('12-12-highspeed.csv')
lowspeed_curves = pl.read_csv('12-12-lowspeed.csv')
nasty_curves = pl.concat([highspeed_curves, lowspeed_curves], how="horizontal")

settings = {"0-4.3 0-4.3": (0,4.3,0,4.3),
            "0-3 0-3": (0,3,0,3),
            "0-2 0-2": (0,2,0,2),
            "0-1 0-1": (0,1,0,1),
            "0-0 0-0": (0,0,0,0),
            "2-4.3 2-4.3": (2,4.3,2,4.3),
            "4-4.3 4-4.3": (4,4.3,4,4.3),
            "6-4.3 6-4.3": (6,4.3,6,4.3),
            "10-4.3 10-4.3": (10,4.3,10,4.3),
            "15-4.3 15-4.3": (15,4.3,15,4.3),
            "25-4.3 25-4.3": (25,4.3,25,4.3)}

curves = pl.DataFrame()
for key in settings.keys():
    tdf = nasty_curves.with_columns(
            (pl.col(key + " X").alias('velocity')),
            (pl.col(key + " Y").alias('force')))

    tdf = tdf.drop_nulls()
    tdf = tdf.sort('velocity')

    tdf = tdf.with_columns(
        pl.when(pl.col("force") < 0)
          .then(-pl.col("velocity").abs())
          .otherwise(pl.col("velocity").abs())
          .alias("velocity")
    )

    v_new = np.arange(0, 10.0 + 1e-9, 0.05)
    f_new = np.interp(v_new, tdf['velocity'], tdf['force'])

    tdf = pl.DataFrame({'velocity': v_new, 'force': f_new})

    tdf = tdf.with_columns(
            (pl.lit(float(settings[key][0])).alias('lsc')),
            (pl.lit(float(settings[key][1])).alias('hsc')),
            (pl.lit(float(settings[key][2])).alias('lsr')),
            (pl.lit(float(settings[key][3])).alias('hsr')))

    curves = pl.concat([curves, tdf["velocity", "force", "lsc", "hsc", "lsr", "hsr"]], how="vertical")

interperator = interpolate.NearestNDInterpolator(curves["velocity", "lsc", "hsc", "lsr", "hsr"], curves["force"])


def suspensionForce(position, velocity, params, preload):
    lsc = params[0]
    hsc = params[1]
    lsr = params[2]
    hsr = params[3]

    try:
      pos_in = position/25.4
      force_lb = pos_in * 200
      velocity = velocity / 25.4
      interp_result = interperator(np.array([velocity, lsc, hsc, lsr, hsr]))
      force_lb += interp_result.item() if hasattr(interp_result, 'item') else float(interp_result)
      force_lb += preload
    except:
      return None
    return force_lb

##changed ver of my code
files = ["/workspaces/Sims-Data/fs-data/FS-3/03162026/2_steeper_regen_curve.parquet"]
window = (10_000, 35_000)
out_root = "data"

def run(path):
    run_id = path.split("/")[-2]
    out = out_root + "/" + run_id
    os.makedirs(out, exist_ok=True)
    print("processing", run_id)

    all_data = pl.read_parquet(path)
    rd = all_data[['Time_ms',
                   'TPERIPH_BL_DATA_SUSTRAVEL',
                   'TPERIPH_BR_DATA_SUSTRAVEL',
                   'TPERIPH_FR_DATA_SUSTRAVEL',
                   'TPERIPH_FL_DATA_SUSTRAVEL']]

    rd = rd.filter(pl.col('Time_ms') > window[0]).filter(pl.col('Time_ms') < window[1])

    rd = rd.with_columns([
        pl.col("TPERIPH_BL_DATA_SUSTRAVEL").forward_fill(),
        pl.col("TPERIPH_BR_DATA_SUSTRAVEL").forward_fill(),
        pl.col("TPERIPH_FL_DATA_SUSTRAVEL").forward_fill(),
        pl.col("TPERIPH_FR_DATA_SUSTRAVEL").forward_fill(),
    ]).drop_nulls()

    rd = rd.with_columns(
        pl.col("TPERIPH_BL_DATA_SUSTRAVEL").rolling_mean(window_size=5).alias("TPERIPH_BL_DATA_SUSTRAVEL"),
        pl.col("TPERIPH_BR_DATA_SUSTRAVEL").rolling_mean(window_size=5).alias("TPERIPH_BR_DATA_SUSTRAVEL"),
        pl.col("TPERIPH_FL_DATA_SUSTRAVEL").rolling_mean(window_size=5).alias("TPERIPH_FL_DATA_SUSTRAVEL"),
        pl.col("TPERIPH_FR_DATA_SUSTRAVEL").rolling_mean(window_size=5).alias("TPERIPH_FR_DATA_SUSTRAVEL")
    )

    rd = rd.with_columns(
        (pl.col('TPERIPH_BL_DATA_SUSTRAVEL').diff() / (pl.col('Time_ms').diff()/1000)).alias('BL_SUSVELOCITY'),
        (pl.col('TPERIPH_BR_DATA_SUSTRAVEL').diff() / (pl.col('Time_ms').diff()/1000)).alias('BR_SUSVELOCITY'),
        (pl.col('TPERIPH_FL_DATA_SUSTRAVEL').diff() / (pl.col('Time_ms').diff()/1000)).alias('FL_SUSVELOCITY'),
        (pl.col('TPERIPH_FR_DATA_SUSTRAVEL').diff() / (pl.col('Time_ms').diff()/1000)).alias('FR_SUSVELOCITY'))

    bl_p = damper_params["BL"]
    br_p = damper_params["BR"]
    fl_p = damper_params["FL"]
    fr_p = damper_params["FR"]
    bl_pre = preload_lb["BL"]
    br_pre = preload_lb["BR"]
    fl_pre = preload_lb["FL"]
    fr_pre = preload_lb["FR"]

    rd = rd.with_columns([
        pl.struct(["TPERIPH_BL_DATA_SUSTRAVEL", "BL_SUSVELOCITY"])
          .map_elements(
              lambda row: suspensionForce(row["TPERIPH_BL_DATA_SUSTRAVEL"], row["BL_SUSVELOCITY"], bl_p, bl_pre),
              return_dtype=pl.Float64
          )
          .alias("BL_SUSFORCE"),

        pl.struct(["TPERIPH_BR_DATA_SUSTRAVEL", "BR_SUSVELOCITY"])
          .map_elements(
              lambda row: suspensionForce(row["TPERIPH_BR_DATA_SUSTRAVEL"], row["BR_SUSVELOCITY"], br_p, br_pre),
              return_dtype=pl.Float64
          )
          .alias("BR_SUSFORCE"),

        pl.struct(["TPERIPH_FL_DATA_SUSTRAVEL", "FL_SUSVELOCITY"])
          .map_elements(
              lambda row: suspensionForce(row["TPERIPH_FL_DATA_SUSTRAVEL"], row["FL_SUSVELOCITY"], fl_p, fl_pre),
              return_dtype=pl.Float64
          )
          .alias("FL_SUSFORCE"),

        pl.struct(["TPERIPH_FR_DATA_SUSTRAVEL", "FR_SUSVELOCITY"])
          .map_elements(
              lambda row: suspensionForce(row["TPERIPH_FR_DATA_SUSTRAVEL"], row["FR_SUSVELOCITY"], fr_p, fr_pre),
              return_dtype=pl.Float64
          )
          .alias("FR_SUSFORCE"),
    ])

    df = rd.to_pandas().dropna(subset=["FL_SUSFORCE", "FR_SUSFORCE", "BL_SUSFORCE", "BR_SUSFORCE"])

    df["FRONT_LB"] = df["FL_SUSFORCE"] + df["FR_SUSFORCE"]
    df["REAR_LB"]  = df["BL_SUSFORCE"] + df["BR_SUSFORCE"]
    df["TOTAL_LB"] = df["FRONT_LB"] + df["REAR_LB"]

    m  = df["TOTAL_LB"].mean()
    mx = df["TOTAL_LB"].max()
    mn = df["TOTAL_LB"].min()
    print("TOTAL lb mean:", round(m, 1), "max:", round(mx, 1), "min:", round(mn, 1))
    print("preload offset applied:", round(sum(preload_lb.values()), 1), "lb total")

    parked = df[(df["Time_ms"] >= 160_000) & (df["Time_ms"] <= 180_000)]
    if len(parked) > 0:
        print("parked total:", round(parked["TOTAL_LB"].mean(), 1), "lb (should ≈ vehicle weight)")

    t = df["Time_ms"] / 1000
    plt.figure(figsize=(12, 5))
    plt.plot(t, df["TOTAL_LB"], label="Total")
    plt.plot(t, df["FRONT_LB"], label="Front", alpha=0.7)
    plt.plot(t, df["REAR_LB"], label="Rear", alpha=0.7)
    plt.xlabel("Time [s]")
    plt.ylabel("Load [lb]")
    plt.title(run_id + " - vertical load (spring + damper + preload)")
    plt.legend()
    plt.grid()
    plt.savefig(out + "/load_vs_time.png", dpi=150)
    plt.close()
    return {"run": run_id, "mean": m, "max": mx, "min": mn}

results = []
for f in files:
    r = run(f)
    if r:
        results.append(r)

print("\nsummary:")
for r in results:
    print(r["run"], "mean:", round(r["mean"], 1))