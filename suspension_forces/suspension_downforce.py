import pandas as pd
import matplotlib.pyplot as plt
import os

files = ["/workspaces/Sims-Data/fs-data/FS-3/03162026/2_steeper_regen_curve.parquet"]
travel_cols = {
    "FL": "TPERIPH_FL_DATA_SUSTRAVEL",
    "FR": "TPERIPH_FR_DATA_SUSTRAVEL",
    "BL": "TPERIPH_BL_DATA_SUSTRAVEL",
    "BR": "TPERIPH_BR_DATA_SUSTRAVEL",
}
spring_rate = 200  # lbs per in
base = (40000, 45000)
window = (0, 150000)
out_root = "data"

def run(path):
    run_id = path.split("/")[-2]
    out = out_root + "/" + run_id
    os.makedirs(out, exist_ok=True)
    print("processing", run_id)

    df = pd.read_parquet(path)
    df = df[(df["Time_ms"] >= window[0]) & (df["Time_ms"] <= window[1])]

    corners = ["FL", "FR", "BL", "BR"]
    for c in corners:
        df[travel_cols[c]] = df[travel_cols[c]].ffill()
    df = df.dropna(subset=[travel_cols[c] for c in corners])

    base_df = df[(df["Time_ms"] >= base[0]) & (df["Time_ms"] <= base[1])]
    baseline = {}
    for c in corners:
        baseline[c] = base_df[travel_cols[c]].median()
    print("baseline:", baseline)

    for c in corners:
        df[c + "_LOAD_LB"] = (df[travel_cols[c]] - baseline[c]) / 25.4 * spring_rate

    df["FRONT_LB"] = df["FL_LOAD_LB"] + df["FR_LOAD_LB"]
    df["REAR_LB"] = df["BL_LOAD_LB"] + df["BR_LOAD_LB"]
    df["TOTAL_LB"] = df["FRONT_LB"] + df["REAR_LB"]

    m = df["TOTAL_LB"].mean()
    mx = df["TOTAL_LB"].max()
    mn = df["TOTAL_LB"].min()
    print("TOTAL lb mean:", round(m, 1), "max:", round(mx, 1), "min:", round(mn, 1))

    t = df["Time_ms"] / 1000
    plt.figure(figsize=(12, 5))
    plt.plot(t, df["TOTAL_LB"], label="Total")
    plt.plot(t, df["FRONT_LB"], label="Front", alpha=0.7)
    plt.plot(t, df["REAR_LB"], label="Rear", alpha=0.7)
    plt.axhline(0, color="k", linestyle="--")
    plt.xlabel("Time [s]")
    plt.ylabel("Load [lb]")
    plt.title(run_id)
    plt.legend()
    plt.grid()
    plt.savefig(out + "/downforce_vs_time.png", dpi=150)
    plt.close()

    df.to_csv(out + "/output.csv", index=False)
    return {"run": run_id, "mean": m, "max": mx, "min": mn}