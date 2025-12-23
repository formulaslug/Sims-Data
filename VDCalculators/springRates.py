#!/usr/bin/env python3
"""
suspension_tuner_with_dumpling.py

Uses FullVehicleSim.TireModel.dumpling.Tire to build tire lateral-capacity lookups
and then evaluates candidate front/rear spring rates to find which maximize steady-state
lateral acceleration for the vehicle.

Run from project root (so imports and params.json path work):
$ python suspension_tuner_with_dumpling.py

Output:
 - tire_grid_FxFy.csv
 - tire_gg_envelope.csv
 - tire_Fymax_vs_Fz.csv
 - spring_sweep_results.csv
 - plots in working directory

Dependencies:
 numpy, pandas, matplotlib
 The FullVehicleSim package (with dumpling Tire) accessible in PYTHONPATH or project root.
"""
import os
import sys
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# try to ensure parent (project root) is in path
PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Add parent to sys.path 
sys.path.append(PARENT) 

from FullVehicleSim.TireModel.dumpling import Tire

# -------------------- User / vehicle spec area --------------------
# These will be overwritten by params.json where available
VEHICLE = {
    "mass_kg": 309.0,           # total vehicle mass
    "g": 9.8067,
    "pressure": 82000,          # Pa
    "temperature": 20.0,        # degC
    "camber": 0.0,              # rad
    "velocityX": 10.0,          # m/s (used by Tire class)
    "track_front_m": 1.1,
    "track_rear_m": 1.1,
    "motion_ratio_front": 0.85,
    "motion_ratio_rear": 0.85,
    "sprung_mass_fraction": 0.95,  # approx
    "weight_fraction_front": 0.5,
    "h_cg_m": 0.25
}

# Tire grid settings (change for resolution / runtime tradeoff)
FZ_LIST = [200, 400, 600, 800, 1000]         # N
SLIP_ANGLES_DEG = np.linspace(-12, 12, 121)  # deg
SLIP_RATIOS = np.linspace(-0.25, 0.25, 101)  # sr

# Sweep ranges for spring tuning (N/m per corner)
# It's often convenient to think in N/mm; 1 N/mm = 1000 N/m
KS_FRONT_RANGE_NMM = np.linspace(5.0, 45.0, 17)   # N/mm per corner
KS_REAR_RANGE_NMM  = np.linspace(5.0, 45.0, 17)   # N/mm per corner

# lateral accel sweep to test if car can sustain
AY_TESTS = np.linspace(0.1, 2.0, 97) * VEHICLE["g"]  # 0.1g..2.0g

OUT_DIR = os.getcwd()

# -------------------- Utility functions --------------------

def load_params_json(path="FullVehicleSim/TireModel/params.json"):
    if not os.path.exists(path):
        print("params.json not found at", path, "— continuing with defaults.")
        return {}
    with open(path, "r") as f:
        return json.load(f)

def build_tire_grid_and_envelope(mech_params, magic_params,
                                 fz_list=FZ_LIST,
                                 slip_ratios=SLIP_RATIOS,
                                 slip_angles_deg=SLIP_ANGLES_DEG,
                                 velocityX=None,
                                 pressure=None,
                                 temperature=None,
                                 camber=None):
    """
    Evaluate Tire over grid and compute:
      - grid dataframe (Fx, Fy for each combo)
      - envelope points (polar outer hull) (ax, ay) as fraction of mg
      - Fy_max_vs_Fz table (max lateral force achievable for each Fz, using combined slip)
    """
    if velocityX is None:
        velocityX = VEHICLE["velocityX"]
    if pressure is None:
        pressure = VEHICLE["pressure"]
    if temperature is None:
        temperature = VEHICLE["temperature"]
    if camber is None:
        camber = VEHICLE["camber"]

    all_rows = []
    envelopes = {}
    for Fz in fz_list:
        points = []
        # Evaluate full grid
        for sr in slip_ratios:
            for sa_deg in slip_angles_deg:
                sa = math.radians(sa_deg)
                t = Tire(Fz, sr, sa, velocityX, pressure,
                         temperature, camber, mech_params, magic_params)
                Fx = t.getLongForce()
                Fy = t.getLateralForce()
                all_rows.append({
                    "Fz": Fz, "sr": sr, "sa_deg": sa_deg,
                    "Fx": Fx, "Fy": Fy,
                })
                # store for envelope
                ax = Fx / (VEHICLE["mass_kg"] * VEHICLE["g"])
                ay = Fy / (VEHICLE["mass_kg"] * VEHICLE["g"])
                points.append((ax, ay, Fx, Fy))
        pts = np.array([[p[0], p[1]] for p in points])
        # polar envelope (sample many angles and pick max projection)
        env = []
        thetas = np.linspace(0, 2*np.pi, 360, endpoint=False)
        for th in thetas:
            proj = pts[:,0] * math.cos(th) + pts[:,1] * math.sin(th)
            idx = np.nanargmax(proj)
            env.append(pts[idx])
        env = np.array(env)
        envelopes[Fz] = env

    df_grid = pd.DataFrame(all_rows)
    # compute Fy_max at each Fz (pure lateral or combined slip)
    Fymax_rows = []
    for Fz in fz_list:
        df_fz = df_grid[df_grid["Fz"] == Fz].copy()
        # max lateral force magnitude (use absolute)
        idx = df_fz["Fy"].abs().idxmax()
        Fymax = float(df_fz.loc[idx, "Fy"])
        Fymax_rows.append({"Fz": Fz, "Fy_max_N": Fymax})
    df_fymax = pd.DataFrame(Fymax_rows).sort_values("Fz")

    # save outputs
    df_grid.to_csv(os.path.join(OUT_DIR, "tire_grid_FxFy.csv"), index=False)
    env_list = []
    for Fz, arr in envelopes.items():
        for ax, ay in arr:
            env_list.append({"Fz": Fz, "ax_g": float(ax), "ay_g": float(ay)})
    pd.DataFrame(env_list).to_csv(os.path.join(OUT_DIR, "tire_gg_envelope.csv"), index=False)
    df_fymax.to_csv(os.path.join(OUT_DIR, "tire_Fymax_vs_Fz.csv"), index=False)

    print("Saved tire grid and envelope CSVs.")
    return df_grid, envelopes, df_fymax

def make_Fymax_interp(df_fymax):
    # linear interpolation in Fz for Fy_max
    from scipy.interpolate import interp1d
    Fz = df_fymax["Fz"].values
    Fy = df_fymax["Fy_max_N"].values
    return interp1d(Fz, Fy, bounds_error=False, fill_value=(Fy[0], Fy[-1]))

def spring_nm_per_mm_to_N_per_m(ks_nmm):
    # ks is provided per corner in N/mm -> convert to N/m
    return ks_nmm * 1000.0

def wheel_rate_from_spring_rate(ks_N_per_m, motion_ratio):
    # k_w = k_s * MR^2 where MR = wheel_disp / spring_disp
    return ks_N_per_m * (motion_ratio**2)

def axle_roll_stiffness_from_wheel_rate(k_wheel, track_m):
    # K_axle = k_w * track^2 (derived from k_axle_total = 2*k_w, K_axle = k_axle_total * t^2 / 2)
    # simplifies to K_axle = k_w * t^2
    return k_wheel * (track_m**2)

def compute_per_wheel_Fz_from_roll(Kf, Kr, ay, vehicle):
    # compute roll angle phi (rad)
    m = vehicle["mass_kg"]
    h = vehicle["h_cg_m"]
    M = m * ay * h
    K_total = Kf + Kr
    # avoid division by zero
    if K_total <= 0:
        phi = 0.0
    else:
        phi = M / K_total
    # axle load transfer from roll stiffness: deltaF = 2*K_axle*phi / t
    deltaF_f = 2.0 * Kf * phi / vehicle["track_front_m"]
    deltaF_r = 2.0 * Kr * phi / vehicle["track_rear_m"]
    # static vertical loads (sprung approx)
    W = m * vehicle["g"]
    Wf = W * vehicle.get("weight_fraction_front", 0.5)
    Wr = W * (1.0 - vehicle.get("weight_fraction_front", 0.5))
    Fz_f_static = Wf / 2.0
    Fz_r_static = Wr / 2.0
    # left/right
    Fz_fl = Fz_f_static + deltaF_f/2.0
    Fz_fr = Fz_f_static - deltaF_f/2.0
    Fz_rl = Fz_r_static + deltaF_r/2.0
    Fz_rr = Fz_r_static - deltaF_r/2.0
    # clamp to small positive
    Fz_fl = max(10.0, Fz_fl); Fz_fr = max(10.0, Fz_fr)
    Fz_rl = max(10.0, Fz_rl); Fz_rr = max(10.0, Fz_rr)
    return {"phi_rad": phi, "Fz_fl": Fz_fl, "Fz_fr": Fz_fr, "Fz_rl": Fz_rl, "Fz_rr": Fz_rr}

def evaluate_sustainable_ay_for_springs(ks_f_nmm, ks_r_nmm, Fy_interp, vehicle, motion_ratio_f, motion_ratio_r):
    """
    For given per-corner spring rates (N/mm), compute maximum ay (m/s^2) such that
    available lateral force >= required (m*ay).

    Uses Fy_interp(Fz) which returns Fy_max (N) for a given normal load.
    """
    ks_f_Npm = spring_nm_per_mm_to_N_per_m(ks_f_nmm)
    ks_r_Npm = spring_nm_per_mm_to_N_per_m(ks_r_nmm)
    k_w_f = wheel_rate_from_spring_rate(ks_f_Npm, motion_ratio_f)
    k_w_r = wheel_rate_from_spring_rate(ks_r_Npm, motion_ratio_r)
    # axle roll stiffness
    Kf = axle_roll_stiffness_from_wheel_rate(k_w_f, vehicle["track_front_m"])
    Kr = axle_roll_stiffness_from_wheel_rate(k_w_r, vehicle["track_rear_m"])

    # sweep AY and find highest ay where available >= required
    m = vehicle["mass_kg"]
    best_ay = 0.0
    for ay in AY_TESTS:
        loads = compute_per_wheel_Fz_from_roll(Kf, Kr, ay, vehicle)
        Fy_fl = float(Fy_interp(loads["Fz_fl"]))
        Fy_fr = float(Fy_interp(loads["Fz_fr"]))
        Fy_rl = float(Fy_interp(loads["Fz_rl"]))
        Fy_rr = float(Fy_interp(loads["Fz_rr"]))
        total_available = Fy_fl + Fy_fr + Fy_rl + Fy_rr
        required = m * ay
        if total_available + 1e-6 >= required:
            best_ay = ay
        else:
            # since AY_TESTS is increasing, once fails we can break (but might be conservative)
            break
    # also compute vertical natural freq front/rear (for driver comfort check)
    sprung_mass = vehicle["mass_kg"] * vehicle.get("sprung_mass_fraction", 0.95)
    m_corner = sprung_mass / 4.0
    fn_front = (1.0/(2*math.pi)) * math.sqrt(wheel_rate_from_spring_rate(ks_f_Npm, motion_ratio_f) / m_corner)
    fn_rear  = (1.0/(2*math.pi)) * math.sqrt(wheel_rate_from_spring_rate(ks_r_Npm, motion_ratio_r) / m_corner)
    return {"best_ay_g": best_ay / vehicle["g"], "best_ay_mps2": best_ay, "Kf": Kf, "Kr": Kr, "fn_front_hz": fn_front, "fn_rear_hz": fn_rear}

# -------------------- Main routine --------------------
def main():
    # load params.json if present
    params = load_params_json()
    mech = params.get("Mechanical-Parameters", {})
    magic = params.get("Magic-Parameters", {})

    # override VEHICLE values (safe merge)
    for k, v in params.get("Vehicle", {}).items():
        VEHICLE[k] = v

    print("Building tire grid & envelope using dumpling Tire (this may take a while)...")
    df_grid, envelopes, df_fymax = build_tire_grid_and_envelope(mech, magic,
                                                                fz_list=FZ_LIST,
                                                                slip_ratios=SLIP_RATIOS,
                                                                slip_angles_deg=SLIP_ANGLES_DEG,
                                                                velocityX=VEHICLE["velocityX"],
                                                                pressure=VEHICLE["pressure"],
                                                                temperature=VEHICLE["temperature"],
                                                                camber=VEHICLE["camber"])
    Fy_interp = make_Fymax_interp(df_fymax)

    # prepare spring sweep
    records = []
    print("Sweeping spring rates (this may take a while depending on grid size)...")
    for ks_f_nmm in tqdm(KS_FRONT_RANGE_NMM, desc="Front ks"):
        for ks_r_nmm in KS_REAR_RANGE_NMM:
            res = evaluate_sustainable_ay_for_springs(ks_f_nmm, ks_r_nmm, Fy_interp, VEHICLE,
                       VEHICLE["motion_ratio_front"], VEHICLE["motion_ratio_rear"])

            # store
            records.append({
                "ks_f_N_per_mm": ks_f_nmm,
                "ks_r_N_per_mm": ks_r_nmm,
                "best_ay_g": res["best_ay_g"],
                "best_ay_mps2": res["best_ay_mps2"],
                "Kf_Nm_per_rad": res["Kf"],
                "Kr_Nm_per_rad": res["Kr"],
                "fn_front_hz": res["fn_front_hz"],
                "fn_rear_hz": res["fn_rear_hz"]
            })

    df_results = pd.DataFrame(records)
    out_csv = os.path.join(OUT_DIR, "spring_sweep_results.csv")
    df_results.to_csv(out_csv, index=False)
    print("Saved spring sweep results to", out_csv)

    # find best results (max best_ay_g)
    best = df_results.sort_values("best_ay_g", ascending=False).head(10)
    print("Top 10 spring candidates by sustained lateral accel (g):")
    print(best[["ks_f_N_per_mm", "ks_r_N_per_mm", "best_ay_g", "fn_front_hz", "fn_rear_hz"]])

    # Plot heatmap of best_ay_g vs ks_f, ks_r
    pivot = df_results.pivot(index="ks_r_N_per_mm", columns="ks_f_N_per_mm", values="best_ay_g")
    plt.figure(figsize=(8,6))
    plt.title("Sustained lateral accel (g) vs front/rear spring rates")
    im = plt.imshow(pivot.values, origin="lower", aspect="auto",
                    extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()])
    plt.colorbar(im, label="sustained ay (g)")
    plt.xlabel("Front spring ks (N/mm)")
    plt.ylabel("Rear spring ks (N/mm)")
    plt.savefig(os.path.join(OUT_DIR, "spring_sweep_heatmap.png"), dpi=200)
    plt.close()
    print("Saved spring_sweep_heatmap.png")

if __name__ == "__main__":
    main()
