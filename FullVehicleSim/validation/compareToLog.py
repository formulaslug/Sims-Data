"""
Compare a sim run against the logged run that generated its control inputs.

    python FullVehicleSim/validation/compareToLog.py -s FullVehicleSim/simulation_output.parquet \
        -l ../fs-data/FS-3/03162026/2_steeper_regen_curve.parquet [--plot]

Logged channel conventions (FS-3 logger, verified on 2026-03-16 data against the GPS path):
  * VDM IMU is negated w.r.t. the sim's ISO frame (x fwd, y left, z up): a_x = -X_ACC, a_y = -Y_ACC (g), r = -Z_YAW_RATE (deg/s)
  * SME_TRQSPD_Torque is % of 180 Nm (100 = 180 Nm)
  * TMAIN_DATA_STEERING is steering-wheel degrees, + left, zero offset 10.5 deg
  * Speed reference = motor RPM through the gear ratio (GPS speed/position is unreliable)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from FullVehicleSim.paramLoader import Parameters  # noqa: E402

STEER_OFFSET_DEG = 10.5
TORQUE_PERCENT_TO_NM = 180.0 / 100.0


def loadLog(path, t) -> dict:
    log = pl.read_parquet(path)
    tLog = (log["Time_ms"] - log["Time_ms"].min()).to_numpy() * 1e-3

    def ch(name):
        s = log[name].cast(pl.Float64).to_numpy()
        m = ~np.isnan(s)
        return np.interp(t, tLog[m], s[m])

    return {
        "speed": ch("SME_TRQSPD_Speed") * 2 * np.pi / 60 / Parameters["gearRatio"] * Parameters["wheelRadius"],
        "motorTorque": ch("SME_TRQSPD_Torque") * TORQUE_PERCENT_TO_NM,
        "yawRate": -np.radians(ch("VDM_Z_AXIS_YAW_RATE")),
        "ay": -ch("VDM_Y_AXIS_ACCELERATION") * 9.81,
        "ax": -ch("VDM_X_AXIS_ACCELERATION") * 9.81,
        "steerWheel": np.radians(ch("TMAIN_DATA_STEERING") - STEER_OFFSET_DEG),
    }, float(tLog.max())


def smooth(a, n=25):
    return np.convolve(a, np.ones(n) / n, "same")


def errorStats(name, sim, real, mask):
    s, r = sim[mask], real[mask]
    gain = np.polyfit(r, s, 1)[0]
    print(f"  {name:14s} sim/real gain {gain:6.3f}   corr {np.corrcoef(s, r)[0, 1]:+.3f}   bias {np.mean(s - r):+8.3f}   rms {np.sqrt(np.mean((s - r) ** 2)):8.3f}")


def steadyStateFit(steerWheel, yawRate, speed, ay, t, lagSteps=3):
    """
    Steady-state handling: delta_sw = ratio * (L*r/V) + ratio*K*a_y  (+ offset), on quasi-steady cornering samples.
    -> effective steering ratio and understeer gradient K (road-wheel deg per g).
    """
    L = Parameters["wheelBase"]
    st, r, v, a = np.roll(smooth(steerWheel, 31), lagSteps), smooth(yawRate, 31), smooth(speed, 31), smooth(ay, 31)
    m = (v > 6) & (np.abs(np.gradient(st, t)) < 0.3) & (np.abs(np.gradient(r, t)) < 0.3) & (np.abs(r) > 0.1)
    A = np.c_[L * r[m] / v[m], a[m], np.ones(m.sum())]
    coef, *_ = np.linalg.lstsq(A, st[m], rcond=None)
    r2 = 1 - np.var(st[m] - A @ coef) / np.var(st[m])
    return coef[0], np.degrees(coef[1] / coef[0]) * 9.81, r2, int(m.sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", "-s", required=True)
    ap.add_argument("--log", "-l", required=True)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    sim = pl.read_parquet(args.sim)
    t = sim["time"].to_numpy()
    real, tEnd = loadLog(args.log, t)
    n = min(len(t), int(tEnd * Parameters["stepsPerSecond"]))
    t = t[:n]
    real = {k: v[:n] for k, v in real.items()}
    s = {k: sim[c].to_numpy()[:n] for k, c in [("speed", "longitudinalVelocity"), ("motorTorque", "motorTorque"),
                                                ("yawRate", "yawRate"), ("ay", "accelerationY"), ("ax", "accelerationX")]}
    dt = 1 / Parameters["stepsPerSecond"]

    print(f"distance: sim {np.sum(s['speed']) * dt:.0f} m, real {np.sum(real['speed']) * dt:.0f} m")
    moving = real["speed"] > 3
    print("sim vs real (real moving > 3 m/s):")
    errorStats("speed m/s", s["speed"], real["speed"], np.ones(n, bool))
    errorStats("torque Nm", s["motorTorque"], real["motorTorque"], moving)
    errorStats("yawRate rad/s", s["yawRate"], real["yawRate"], moving)
    errorStats("ay m/s2", smooth(s["ay"]), smooth(real["ay"]), moving)
    errorStats("ax m/s2", smooth(s["ax"]), smooth(real["ax"]), moving)
    print(f"  |ay| p99:  sim {np.percentile(np.abs(smooth(s['ay'])), 99):.1f}  real {np.percentile(np.abs(smooth(real['ay'])), 99):.1f} m/s2")

    print("steady-state handling fit (steering wheel = ratio*(L r / V) + ratio*K*a_y):")
    for name, r, v, a in [("real", real["yawRate"], real["speed"], real["ay"]), ("sim ", s["yawRate"], s["speed"], s["ay"])]:
        ratio, K, r2, npts = steadyStateFit(real["steerWheel"], r, v, a, t)
        print(f"  {name}: effective steering ratio {ratio:5.2f}   understeer gradient {K:+5.2f} deg/g   (R2 {r2:.2f}, n {npts})")
    print(f"  (params steeringRatio = {Parameters['steeringRatio']})")

    if args.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
        for i, (k, label) in enumerate([("speed", "speed m/s"), ("motorTorque", "motor torque Nm"), ("yawRate", "yaw rate rad/s"), ("ay", "a_y m/s2")]):
            ax[i].plot(t, real[k], lw=0.6, label="real")
            ax[i].plot(t, s[k], lw=0.6, label="sim")
            ax[i].set_ylabel(label)
            ax[i].legend()
        ax[-1].set_xlabel("time s")
        plt.tight_layout()
        plt.show()
