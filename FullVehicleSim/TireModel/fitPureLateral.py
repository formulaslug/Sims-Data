"""
Fit the Pacejka Magic Formula PURE LATERAL force (no slip ratio, with camber) to FSAE TTC Round 8 data.

    python FullVehicleSim/TireModel/fitPureLateral.py --tire front
    python FullVehicleSim/TireModel/fitPureLateral.py --tire rear --plot

Equations: H. B. Pacejka, Tire and Vehicle Dynamics, 3rd ed. (2012), Section 4.3.2, Eqs. 4.E19-4.E30 (pp. 177-180),
with camber (IA -4..4 deg), all zeta = 1 and all user scaling factors lambda = 1 (the sim applies its own lambda_muy
for the TTC-belt -> track friction difference, see Mech/steering.py).

Sign convention: TTC data is SAE (FZ negative = load, FY = -K*SA). The fit is done in Pacejka/ISO form,
Fy = +K*alpha, i.e. Fy_fit = -FY_sae and alpha = SA. Fz = -FZ_sae.
Camber gamma = IA as logged (SAE: + = top of the wheel leaning to the RIGHT). The fit gives Fy < 0 for gamma > 0
(camber thrust points the way the wheel leans). Mech/tireLoad.py converts each wheel's camber to this sign.

Output: TireModel/LateralFits/<name>.json, loaded by paramLoader via params.json5 ("frontTireFit"/"rearTireFit").
"""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import least_squares

HERE = Path(__file__).parent
TTC_DIR = HERE / "../../../fs-data/TireData"  # D:/Projects/FormulaSlug/fs-data/TireData

# FSAE TTC Round 8 run guide (RunGuide_Round8.pdf): run containing "initial 12 psi" + "10, 14 psi" cornering sweeps.
TIRES = {
    #         Hoosier part, size, compound, rim width (in), cornering run
    "front": dict(name="H43070LC0_16x6.0-10_6in", run=21, description="Hoosier 43070 16.0x6.0-10 LC0 on 6 in rim"),
    "rear":  dict(name="H43075LC0_16x7.5-10_7in", run=18, description="Hoosier 43075 16.0x7.5-10 LC0 on 7 in rim"),
}
NOMINAL_LOAD = 650.0      # N, Fz0 (TTC sweeps span 200-1200 N; ~static corner weight of this car)
NOMINAL_PRESSURE = 82737  # Pa, 12 psi (p_i0)

PARAM_NAMES = ["p_cy1", "p_dy1", "p_dy2", "p_dy3", "p_py3", "p_py4", "p_ey1", "p_ey2", "p_ey3", "p_ey4",
               "p_ky1", "p_ky2", "p_ky3", "p_ky4", "p_ky5", "p_ky6", "p_ky7", "p_py1", "p_py2",
               "p_hy1", "p_hy2", "p_vy1", "p_vy2", "p_vy3", "p_vy4"]
INITIAL = dict(p_cy1=1.4, p_dy1=2.5, p_dy2=-0.2, p_dy3=0.0, p_py3=0.0, p_py4=0.0, p_ey1=0.0, p_ey2=0.0, p_ey3=0.0, p_ey4=0.0,
               p_ky1=40.0, p_ky2=1.5, p_ky3=0.0, p_ky4=2.0, p_ky5=0.0, p_ky6=-1.0, p_ky7=0.0, p_py1=0.0, p_py2=0.0,
               p_hy1=0.0, p_hy2=0.0, p_vy1=0.0, p_vy2=0.0, p_vy3=0.0, p_vy4=0.0)


def corneringStiffness(p:dict, normalForce, pressure, camber=0.0, nominalLoad=NOMINAL_LOAD, nominalPressure=NOMINAL_PRESSURE):
    # 4.E25, N/rad
    dpi = (pressure - nominalPressure) / nominalPressure
    g = np.sin(camber)
    return (p["p_ky1"] * nominalLoad * (1 + p["p_py1"] * dpi) * (1 - p["p_ky3"] * np.abs(g))
            * np.sin(p["p_ky4"] * np.arctan(normalForce / nominalLoad / ((p["p_ky2"] + p["p_ky5"] * g**2) * (1 + p["p_py2"] * dpi)))))


def pureLateralForce(p:dict, slipAngle, normalForce, pressure, camber=0.0, nominalLoad=NOMINAL_LOAD, nominalPressure=NOMINAL_PRESSURE):
    """
    Pacejka Magic Formula pure lateral force, Eqs. 4.E19-4.E30 (pp. 179-180). ISO sign: Fy = +K*alpha.
    The sim (Mech/steering.py) keeps its own copy of these equations: keep them in sync.
    """
    dfz = (normalForce - nominalLoad) / nominalLoad                              # 4.E2a
    dpi = (pressure - nominalPressure) / nominalPressure                         # 4.E2b
    g = np.sin(camber)                                                           # 4.E4: gamma* = sin(gamma)
    Cy = p["p_cy1"]                                                              # 4.E21
    muy = (p["p_dy1"] + p["p_dy2"] * dfz) * (1 + p["p_py3"] * dpi + p["p_py4"] * dpi**2) * (1 - p["p_dy3"] * g**2)  # 4.E23
    Dy = muy * normalForce                                                       # 4.E22
    Kya = corneringStiffness(p, normalForce, pressure, camber, nominalLoad, nominalPressure)  # 4.E25
    By = Kya / (Cy * Dy + 1e-6)                                                  # 4.E26
    SVyg = normalForce * (p["p_vy3"] + p["p_vy4"] * dfz) * g                     # 4.E28
    SVy = normalForce * (p["p_vy1"] + p["p_vy2"] * dfz) + SVyg                   # 4.E29
    Kyg0 = normalForce * (p["p_ky6"] + p["p_ky7"] * dfz)                         # 4.E30 (camber stiffness)
    SHy = p["p_hy1"] + p["p_hy2"] * dfz + (Kyg0 * g - SVyg) / (Kya + 1e-6)      # 4.E27
    alphaY = np.tan(slipAngle) + SHy                                             # 4.E3 (alpha* = tan alpha), 4.E20
    Ey = np.minimum((p["p_ey1"] + p["p_ey2"] * dfz) * (1 - (p["p_ey3"] + p["p_ey4"] * g) * np.sign(alphaY)), 1.0)  # 4.E24 (<= 1)
    BA = By * alphaY
    return Dy * np.sin(Cy * np.arctan(BA - Ey * (BA - np.arctan(BA)))) + SVy     # 4.E19


def loadSweeps(run:int) -> pl.DataFrame:
    d = pl.read_csv(TTC_DIR / f"B1965raw{run}.dat", separator="\t", skip_rows=1, skip_rows_after_header=1)
    d = d.filter(
        (pl.col("IA").abs() < 4.5)      # camber sweeps 0, 2, 4 deg
        & (pl.col("V") > 20)            # rolling
        & (pl.col("FZ") < -100)         # loaded
        & (pl.col("SA").abs() < 13)
    )
    return d.select(
        pl.col("SA").radians().alias("alpha"),
        (-pl.col("FZ")).alias("Fz"),
        (pl.col("P") * 1000).alias("p"),   # kPa -> Pa
        pl.col("IA").radians().alias("gamma"),
        (-pl.col("FY")).alias("Fy"),       # SAE -> ISO sign
        pl.col("ET"),
    ).gather_every(25)  # ~10k points is plenty for 25 parameters


def fit(data:pl.DataFrame) -> tuple[dict, float]:
    a, fz, pr, ga, fy = (data[c].to_numpy() for c in ("alpha", "Fz", "p", "gamma", "Fy"))

    def residual(x):
        return (pureLateralForce(dict(zip(PARAM_NAMES, x)), a, fz, pr, ga) - fy) / fz  # normalised so all loads count equally

    x0 = np.array([INITIAL[k] for k in PARAM_NAMES])
    sol = least_squares(residual, x0, loss="soft_l1", f_scale=0.05, x_scale="jac", max_nfev=5000)
    params = dict(zip(PARAM_NAMES, sol.x.tolist()))
    rms = float(np.sqrt(np.mean((pureLateralForce(params, a, fz, pr, ga) - fy) ** 2)))
    return params, rms


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tire", choices=list(TIRES), required=True)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    tire = TIRES[args.tire]

    data = loadSweeps(tire["run"])
    params, rms = fit(data)

    out = {
        "description": tire["description"],
        "source": f"FSAE TTC Round 8, Calspan project 1965, run {tire['run']}, IA -4..4 deg, all pressures",
        "fit": "Pacejka 3rd ed. Eqs. 4.E19-4.E30, pure lateral, ISO sign (Fy = +K*alpha), gamma = IA (SAE sign)",
        "nominalLoad": NOMINAL_LOAD,
        "nominalPressure": NOMINAL_PRESSURE,
        "rmsErrorN": rms,
        **params,
    }
    outDir = HERE / "LateralFits"
    outDir.mkdir(exist_ok=True)
    (outDir / f"{tire['name']}.json").write_text(json.dumps(out, indent=4))

    print(f"{tire['description']}: {data.height} points, rms error {rms:.1f} N")
    sa = np.radians(np.linspace(-12, 12, 400))
    for fz in (300, 500, 650, 800, 1000):
        C = corneringStiffness(params, fz, NOMINAL_PRESSURE)
        peak = pureLateralForce(params, sa, fz, NOMINAL_PRESSURE).max()
        print(f"  Fz {fz:5.0f} N: C = {C:7.0f} N/rad ({np.radians(C):5.1f} N/deg)  peak Fy {peak:6.0f} N (mu {peak/fz:.2f})")
        for ia in (-2, 2):
            f = pureLateralForce(params, sa, fz, NOMINAL_PRESSURE, np.radians(ia))
            print(f"      IA {ia:+d}: Fy(SA=0) {pureLateralForce(params, 0.0, fz, NOMINAL_PRESSURE, np.radians(ia)):+5.0f} N   peak +{f.max():.0f} / {f.min():.0f} N")

    if args.plot:
        import matplotlib.pyplot as plt
        plt.scatter(np.degrees(data["alpha"]), data["Fy"], s=0.2, c=data["Fz"], cmap="viridis")
        for fz in (250, 450, 650, 850, 1100):
            plt.plot(np.degrees(sa), pureLateralForce(params, sa, fz, NOMINAL_PRESSURE), label=f"fit Fz={fz}")
        plt.xlabel("slip angle (deg)"); plt.ylabel("Fy (N, ISO)"); plt.legend(); plt.title(tire["description"])
        plt.show()
