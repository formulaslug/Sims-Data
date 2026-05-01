import numpy as np
import matplotlib.pyplot as plt
import json
import csv
import os

mass = 293.97 #FS-3 Comp weight + 160lb driver, DOES NOT INCLUDE AERO PACKAGE
weight = 2883.8457  # N
frontWD = 0.4632
rearWD = 1 - frontWD
leftWD = 0.492
CGHeight = 0.234  # meters
trackWidth = 1.325  # 1325 mm center-center track width
wheelBase = 1.59
RCFront = 0.0203
RCRear = 0.0493
#recalcualate RC's after onshape is done
#Right now rear is ~0.0426974 mm
motionRatioF = 1.006
motionRatioR = 1.004
multiplier = 0.00571015 #n/m -> lbf/in
masterAy= 1.7
masterV = 40
TRG = 0.01524409115   # target roll gradient in rad/g
print("TRG = ", TRG)

# No-aero baseline: zero aero downforce and drag across all components.
REF_V_MPS = 40.0
REF_V2 = REF_V_MPS**2
DF_FRONT_WING_C = 0.0
DF_REAR_WING_C = 0.0
DF_FLOOR_BODY_C = 0.0

DRAG_FRONT_WING_C = 0.0
DRAG_REAR_WING_C = 0.0
DRAG_FLOOR_BODY_C = 0.0

ROLL_CENTER_IN = np.array([0.0, -21.685, 2.879])
PITCH_CENTER_IN = np.array([0.0, 13.913, 5.970])

WHEEL_DIAMETER_IN = 16.1
WHEEL_CENTER_Z_IN = 4.9064
GROUND_Z_IN = WHEEL_CENTER_Z_IN - (WHEEL_DIAMETER_IN / 2.0)

BODY_POINTS_IN = {
    "front_wing": np.array([27.7744, -58.112, -1.1732]),
    "floor": np.array([26.6546, -2.9176, -1.2756]),
    "floor_2": np.array([26.6546, 28.7472, -1.2756]),
    "rear_wing_thing": np.array([-12.0, 47.0468, -1.2756]),
}

SWEEP_AY_G = np.linspace(-2.5, 2.5, 81)
SWEEP_AX_G = np.linspace(-2.0, 2.0, 81)
TRAVEL_SWEEP_AY_G = np.linspace(0.0, 2.5, 81)
###     Aero Loads (N)  ###


def aero_component_loads(v):
    front_wing = DF_FRONT_WING_C * v**2
    rear_wing = DF_REAR_WING_C * v**2
    floor_body_total = DF_FLOOR_BODY_C * v**2

    front_floor = 0.5 * floor_body_total
    rear_floor = 0.5 * floor_body_total

    front_aero = front_wing + front_floor
    rear_aero = rear_wing + rear_floor
    return front_aero, rear_aero, front_wing, rear_wing, floor_body_total


def aero_component_drag(v):
    front_drag = DRAG_FRONT_WING_C * v**2
    rear_drag = DRAG_REAR_WING_C * v**2
    floor_body_drag = DRAG_FLOOR_BODY_C * v**2
    total_drag = front_drag + rear_drag + floor_body_drag
    return front_drag, rear_drag, floor_body_drag, total_drag


def compute_spring_rates(v, ay, trg):
    frontAero, rearAero, _, _, _ = aero_component_loads(v)
    
    
    print("Aero Balance (front%) =", frontAero/2000)
    print("Compute_spring_rates is running")
  


    ###     Axle Weights    ###

    _, _, _, _, M_pitch_aero, frontAW, rearAW = aero_load_model(v, truncate_to_CG=True)

    ###     Roll Moments    ###

    frontRM = frontAW * (CGHeight - RCFront) * ay
    rearRM = rearAW * (CGHeight - RCRear) * ay

    ###     Finding Required Roll stiffness     ###

    frontRS = frontRM / trg
    rearRS = rearRM / trg

    ###     Moment from one side about CG       ###
    # roll = Φ
    # track width = t
    # Δz = Φ * t/2
    # wheel rate per side = kw
    # Force at each wheel:
    #     F = kw * Φ * t/2
    # Moment from one side:
    #     Mside = F * (t/2) = kw * Φ * (t/2)^2

    # (t/2)^2 term
    halfTrackSq = (trackWidth / 2)**2

    ###     Wheel Rates    ###
    # Kphi = 2 * kw * (t/2)^2
    # kw = Kphi / (2 * (t/2)^2)

    frontKW = frontRS / (2 * halfTrackSq)
    rearKW = rearRS / (2 * halfTrackSq)

    ###     Spring Rates (using motion ratios)     ###
    # kw = ks * MR^2
    # ks = kw / MR^2

    frontKS = frontKW / (motionRatioF**2)
    rearKS = rearKW / (motionRatioR**2)
    return frontKS, rearKS, frontRS, rearRS, frontKW, rearKW

def roll_and_pitch_gradients(v, ay, ax, frontKS, rearKS):
    """
    Returns:
    roll_gradient  (deg/g)
    pitch_gradient (deg/g)
    """

    # --- Convert springs to wheel rates ---
    frontKW = frontKS * motionRatioF**2
    rearKW  = rearKS  * motionRatioR**2

    # ================= ROLL =================
    halfTrackSq = (trackWidth / 2)**2
    frontRS = 2 * frontKW * halfTrackSq
    rearRS  = 2 * rearKW  * halfTrackSq
    Kphi = frontRS + rearRS

    # Aero
    frontAero, rearAero, _, _, _ = aero_component_loads(v)

    frontAW = frontWD * weight + frontAero
    rearAW  = rearWD  * weight + rearAero

    M_roll = (
        frontAW * (CGHeight - RCFront) +
        rearAW  * (CGHeight - RCRear)
    ) * ay

    phi = M_roll / Kphi
    roll_gradient = np.degrees(phi / ay)

    # ================= PITCH =================
    af = wheelBase * rearWD
    ar = wheelBase * frontWD

    PCHeight = 0.5 * (RCFront + RCRear)

    M_pitch = weight * (CGHeight - PCHeight) * ax

    Ktheta = frontKW * af**2 + rearKW * ar**2

    theta = M_pitch / Ktheta
    pitch_gradient = np.degrees(theta / ax)

    spring_travel_front = (frontAW / 2.0) / frontKW / motionRatioF
    spring_travel_rear = (rearAW / 2.0) / rearKW / motionRatioR

    return roll_gradient, pitch_gradient, spring_travel_front, spring_travel_rear

def aero_load_model(v, truncate_to_CG=True):
    """
    Computes aerodynamic loading and pitch moment.

    Parameters
    ----------
    v : float
        vehicle speed (m/s)

    truncate_to_CG : bool
        If True, aero force is assumed to act through CG (no pitch moment)
        If False, real center of pressure is used

    Returns
    -------
    frontAero : float   (N)
    rearAero  : float   (N)
    totalAero : float   (N)
    x_cop     : float   (m from front axle)
    M_pitch_aero : float (Nm about CG, positive = nose up)
    frontAxleLoad : float (N)
    rearAxleLoad  : float (N)
    """

    # ===============================
    # 1. COMPONENT AERO FORCES
    # ===============================
    frontAero, rearAero, frontWing, rearWing, floorTotal = aero_component_loads(v)

    totalAero = frontAero + rearAero

    # ===============================
    # 2. CG LOCATION FROM FRONT AXLE
    # ===============================
    x_cg = wheelBase * rearWD

    # ===============================
    # 3. TRUE CENTER OF PRESSURE
    # ===============================
    if totalAero > 0:
        x_cop_true = (rearAero * wheelBase) / totalAero
    else:
        x_cop_true = x_cg   # avoid divide by zero

    # ===============================
    # 4. OPTION: TRUNCATE COP TO CG
    # ===============================
    if truncate_to_CG:
        x_cop = x_cg
    else:
        x_cop = x_cop_true

    # ===============================
    # 5. PITCH MOMENT ABOUT CG
    # ===============================
    # positive moment = nose up
    M_pitch_aero = totalAero * (x_cop - x_cg)

    # ===============================
    # 6. VERTICAL LOAD DISTRIBUTION
    # ===============================
    # static weight
    frontStatic = frontWD * weight
    rearStatic  = rearWD  * weight

    # distribute aero vertically to axles
    # based on COP position along wheelbase
    frontAero_axle = totalAero * (wheelBase - x_cop) / wheelBase
    rearAero_axle  = totalAero * x_cop / wheelBase

    frontAxleLoad = frontStatic + frontAero_axle
    rearAxleLoad  = rearStatic  + rearAero_axle

    return (
        frontAero,
        rearAero,
        totalAero,
        x_cop,
        M_pitch_aero,
        frontAxleLoad,
        rearAxleLoad
    )

def mirror_across_x_axis(point):
    return np.array([point[0], -point[1], point[2]])

def make_sweep_points(include_mirror=True):
    points = dict(BODY_POINTS_IN)
    if include_mirror:
        for name, point in BODY_POINTS_IN.items():
            points[f"{name}_mirror"] = mirror_across_x_axis(point)
    return points

def rotate_about_x(point, center, angle_rad):
    translated = point - center
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    rotated = np.array([
        translated[0],
        c * translated[1] - s * translated[2],
        s * translated[1] + c * translated[2],
    ])
    return center + rotated

def rotate_about_y(point, center, angle_rad):
    translated = point - center
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    rotated = np.array([
        c * translated[0] - s * translated[2],
        translated[1],
        s * translated[0] + c * translated[2],
    ])
    return center + rotated

def transform_point(point, roll_rad, pitch_rad, order):
    if order == "roll_then_pitch":
        after_roll = rotate_about_x(point, ROLL_CENTER_IN, roll_rad)
        return rotate_about_y(after_roll, PITCH_CENTER_IN, pitch_rad)
    if order == "pitch_then_roll":
        after_pitch = rotate_about_y(point, PITCH_CENTER_IN, pitch_rad)
        return rotate_about_x(after_pitch, ROLL_CENTER_IN, roll_rad)
    raise ValueError(f"Unknown rotation order: {order}")

def sweep_cornering_ground_clearance(roll_gradient_deg_per_g, pitch_gradient_deg_per_g, spring_travel_front, spring_travel_rear):
    points = make_sweep_points(include_mirror=True)
    rows = []

    for ay_g in SWEEP_AY_G:
        roll_deg = roll_gradient_deg_per_g * ay_g
        roll_rad = np.deg2rad(roll_deg)

        for ax_g in SWEEP_AX_G:
            pitch_deg = pitch_gradient_deg_per_g * ax_g
            pitch_rad = np.deg2rad(pitch_deg)

            order_results = {}
            for order in ("roll_then_pitch", "pitch_then_roll"):
                transformed = {
                    name: transform_point(point, roll_rad, pitch_rad, order)
                    for name, point in points.items()
                }
                min_name = min(transformed, key=lambda name: transformed[name][2])
                min_z = float(transformed[min_name][2])
                clearance_in = min_z - GROUND_Z_IN
                order_results[order] = {
                    "min_name": min_name,
                    "min_z_in": min_z,
                    "clearance_in": clearance_in,
                    "touches_ground": clearance_in <= 0.0,
                }

            worst_order = min(order_results, key=lambda order: order_results[order]["clearance_in"])
            worst = order_results[worst_order]

            rows.append({
                "ay_g": float(ay_g),
                "ax_g": float(ax_g),
                "roll_deg": float(roll_deg),
                "pitch_deg": float(pitch_deg),
                "ground_z_in": float(GROUND_Z_IN),
                "worst_order": worst_order,
                "worst_point": worst["min_name"],
                "worst_min_z_in": worst["min_z_in"],
                "worst_clearance_in": worst["clearance_in"],
                "worst_clearance_mm": worst["clearance_in"] * 25.4,
                "touches_ground": worst["touches_ground"],
                "roll_then_pitch_clearance_in": order_results["roll_then_pitch"]["clearance_in"],
                "pitch_then_roll_clearance_in": order_results["pitch_then_roll"]["clearance_in"],
            })

    return rows

def spring_displacement_model(v, ay, ax, frontKS, rearKS, truncate_to_CG=True):
    """
    Computes spring compression at front and rear due to
    weight + aero + acceleration.

    Returns:
    front_spring_disp : meters
    rear_spring_disp  : meters
    """

    # --- Aero loads and axle loads ---
    _, _, _, _, M_pitch_aero, frontAW, rearAW = aero_load_model(v, truncate_to_CG)

    # --- Longitudinal pitch load transfer (simplified) ---
    PCHeight = 0.5 * (RCFront + RCRear)
    M_pitch_long = weight * (CGHeight - PCHeight) * ax

    M_pitch_total = M_pitch_long + M_pitch_aero

    af = wheelBase * rearWD
    ar = wheelBase * frontWD

    # Axle load changes from pitch
    deltaF_front = -M_pitch_total / wheelBase
    deltaF_rear  =  M_pitch_total / wheelBase

    frontAW += deltaF_front
    rearAW  += deltaF_rear

    # --- Per-wheel vertical load ---
    front_wheel_load = frontAW / 2
    rear_wheel_load  = rearAW / 2

    # --- Convert springs to wheel rates ---
    frontKW = frontKS * motionRatioF**2
    rearKW  = rearKS  * motionRatioR**2

    # --- Wheel displacement ---
    front_wheel_disp = front_wheel_load / frontKW
    rear_wheel_disp  = rear_wheel_load  / rearKW

    # --- Spring displacement ---
    front_spring_disp = front_wheel_disp / motionRatioF
    rear_spring_disp  = rear_wheel_disp  / motionRatioR

    return front_spring_disp, rear_spring_disp


def sweep_cornering_spring_travel(roll_gradient_deg_per_g, front_static_travel_m, rear_static_travel_m):
    rows = []

    for ay_g in TRAVEL_SWEEP_AY_G:
        roll_deg = roll_gradient_deg_per_g * ay_g
        roll_rad = np.deg2rad(roll_deg)

        front_delta_m = abs(roll_rad) * (trackWidth / 2.0) / motionRatioF
        rear_delta_m = abs(roll_rad) * (trackWidth / 2.0) / motionRatioR

        front_outside_m = front_static_travel_m + front_delta_m
        front_inside_m = max(0.0, front_static_travel_m - front_delta_m)
        rear_outside_m = rear_static_travel_m + rear_delta_m
        rear_inside_m = max(0.0, rear_static_travel_m - rear_delta_m)

        rows.append({
            "ay_g": float(ay_g),
            "roll_deg": float(roll_deg),
            "front_static_mm": float(front_static_travel_m * 1000.0),
            "rear_static_mm": float(rear_static_travel_m * 1000.0),
            "front_outside_mm": float(front_outside_m * 1000.0),
            "front_inside_mm": float(front_inside_m * 1000.0),
            "rear_outside_mm": float(rear_outside_m * 1000.0),
            "rear_inside_mm": float(rear_inside_m * 1000.0),
            "front_delta_mm": float(front_delta_m * 1000.0),
            "rear_delta_mm": float(rear_delta_m * 1000.0),
        })

    return rows


def plot_spring_travel_sweep(rows):
    ay_g = [row["ay_g"] for row in rows]
    front_outside = [row["front_outside_mm"] for row in rows]
    rear_outside = [row["rear_outside_mm"] for row in rows]
    front_inside = [row["front_inside_mm"] for row in rows]
    rear_inside = [row["rear_inside_mm"] for row in rows]

    plt.figure()
    plt.plot(ay_g, front_outside, label="Front outside")
    plt.plot(ay_g, rear_outside, label="Rear outside")
    plt.plot(ay_g, front_inside, linestyle="--", label="Front inside")
    plt.plot(ay_g, rear_inside, linestyle="--", label="Rear inside")
    plt.xlabel("Lateral Acceleration (g)")
    plt.ylabel("Spring Travel (mm)")
    plt.title("Spring Travel vs Lateral Acceleration")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_scrape_map(rows, output_path="cornering_scrape_map.png"):
    ax_vals = np.array([row["ax_g"] for row in rows], dtype=float)
    ay_vals = np.array([row["ay_g"] for row in rows], dtype=float)
    scrape = np.array([bool(row["touches_ground"]) for row in rows], dtype=bool)

    plt.figure(figsize=(8, 6))
    plt.scatter(ax_vals[~scrape], ay_vals[~scrape], s=22, c="blue", alpha=0.8, label="No scrape")
    plt.scatter(ax_vals[scrape], ay_vals[scrape], s=22, c="red", alpha=0.85, label="Scrapes")
    plt.xlabel("Longitudinal Acceleration (g)")
    plt.ylabel("Lateral Acceleration (g)")
    plt.title("Ground Scrape Map")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.show()







###     Graphs      ###
def spring_rates_vs_speed_fixed_g():
    speeds = np.linspace(10, 35, 30)
    ay = 2.2
    front_rates = []
    rear_rates = []
    for v in speeds:
        frontKS, rearKS, _, _ = compute_spring_rates(v, ay, TRG)
        front_rates.append(frontKS / 1000)  # N/mm
        rear_rates.append(rearKS / 1000)

    plt.plot(speeds, front_rates, label="Front")
    plt.plot(speeds, rear_rates, label="Rear")

    plt.xlabel("Speed (m/s)")
    plt.ylabel("Spring Rate (N/mm)")
    plt.title("Required Spring Rate vs Speed @ 1.7g")
    plt.legend()
    plt.grid(True)
    plt.show()


def roll_vs_lateral():
    g_vals = np.linspace(0.5, 2.5, 30)
    v = 30  # m/s

    roll_angles = []

    for ay in g_vals:
        _, _, frontRS, rearRS = compute_spring_rates(v, ay, TRG)
        Kphi_total = frontRS + rearRS
        phi = ay / TRG   # radians
        roll_angles.append(np.degrees(phi))

    plt.plot(g_vals, roll_angles)
    plt.xlabel("Lateral Acceleration (g)")
    plt.ylabel("Roll Angle (deg)")
    plt.title("Roll Angle vs Lateral G @ 30 m/s")
    plt.grid(True)
    plt.show()

def roll_angle_vs_speed_calc(speeds, ay, frontKS, rearKS):
    """
    Compute roll angle (deg) vs speed for fixed lateral g and fixed springs.

    speeds  : array-like, vehicle speed (m/s)
    ay      : lateral acceleration (g)
    frontKS : front spring rate (N/m)
    rearKS  : rear spring rate (N/m)

    Returns:
    roll_angles : numpy array of roll angle in degrees
    """

    # --- Convert springs to roll stiffness ---
    frontKW = frontKS * motionRatioF**2
    rearKW  = rearKS  * motionRatioR**2

    halfTrackSq = (trackWidth / 2)**2

    frontRS = 2 * frontKW * halfTrackSq
    rearRS  = 2 * rearKW  * halfTrackSq

    Kphi_total = frontRS + rearRS

    roll_angles = []

    for v in speeds:

        # --- Aero forces ---
        frontAero, rearAero, _, _, _ = aero_component_loads(v)

        # --- Axle loads ---
        frontAW = frontWD * weight + frontAero
        rearAW  = rearWD  * weight + rearAero

        # --- Roll moment ---
        frontRM = frontAW * (CGHeight - RCFront) * ay
        rearRM  = rearAW  * (CGHeight - RCRear)  * ay

        M_roll = frontRM + rearRM

        # --- Roll angle ---
        phi = M_roll / Kphi_total      # radians
        roll_angles.append(np.degrees(phi))

    return np.array(roll_angles)

def roll_angle_vs_speed_display(ay):
    speeds = np.linspace(10, 35, 40)
    

    frontKS = compute_spring_rates(masterV, masterAy, TRG)[0]   # N/m
    rearKS  = compute_spring_rates(masterV, masterAy, TRG)[1]   # N/m
    print(frontKS, rearKS)
    roll = roll_angle_vs_speed_calc(speeds, ay, frontKS, rearKS)
    print(frontKS, rearKS)

    plt.plot(speeds, roll)
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Roll Angle (deg)")
    plt.title(f"Roll Angle vs Speed @ {ay}g")
    plt.grid(True)
    plt.show()
#for roll angle vs speed at a set g, call the display one with a constant ay as g's
computed_spring_rates = compute_spring_rates(masterV,masterAy, TRG)
frontKS = computed_spring_rates[0]
rearKS = computed_spring_rates[1]
frontRS = computed_spring_rates[2]
rearRS = computed_spring_rates[3]
frontKW = computed_spring_rates[4]
rearKW = computed_spring_rates [5]

###     Print results     ###

print("Front Wheel Rate (kN/m):", frontKW/1000)
print("Rear Wheel Rate (kN/m):", rearKW/1000)
print("Front Spring Rate (N/m):", frontKS)
print("Rear Spring Rate (N/m):", rearKS)
print("Front Spring Rate (lbf/in):", frontKS * multiplier)
print("Rear Spring Rate (lbf/in):", rearKS * multiplier)
print()
roll_gradient, pitch_gradient, spring_travel_front, spring_travel_rear = roll_and_pitch_gradients(30, 1, 1, frontKS, rearKS)
print("roll gradient:", roll_gradient*0.0174533, "rads/g")
print(roll_gradient)
print("pitch gradient:", pitch_gradient*0.0174533, "rads/g")
print(pitch_gradient)

corner_sweep = sweep_cornering_ground_clearance(roll_gradient, pitch_gradient, spring_travel_front, spring_travel_rear)
out_csv = "cornering_ground_clearance_sweep.csv"
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(corner_sweep[0].keys()))
    writer.writeheader()
    writer.writerows(corner_sweep)

scrape_map_png = "cornering_scrape_map.png"
plot_scrape_map(corner_sweep, scrape_map_png)

travel_sweep = sweep_cornering_spring_travel(roll_gradient, spring_travel_front, spring_travel_rear)
travel_csv = "cornering_spring_travel_sweep.csv"
with open(travel_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(travel_sweep[0].keys()))
    writer.writeheader()
    writer.writerows(travel_sweep)

plot_spring_travel_sweep(travel_sweep)

worst_case = min(corner_sweep, key=lambda row: row["worst_clearance_in"])
touching_cases = [row for row in corner_sweep if row["touches_ground"]]
max_front_travel = max(travel_sweep, key=lambda row: row["front_outside_mm"])
max_rear_travel = max(travel_sweep, key=lambda row: row["rear_outside_mm"])

print()
print("Cornering sweep summary")
print("Track width used:", trackWidth, "m")
print("Sweep grid:", len(SWEEP_AY_G), "lateral points x", len(SWEEP_AX_G), "longitudinal points")
print("Ground plane z (in):", f"{GROUND_Z_IN:.4f}")
print("Worst-case clearance:", f"{worst_case['worst_clearance_in']:.4f} in", f"({worst_case['worst_clearance_mm']:.2f} mm)")
print("Worst-case load case:", f"ay={worst_case['ay_g']:.3f} g", f"ax={worst_case['ax_g']:.3f} g")
print("Worst-case point:", worst_case["worst_point"], "under", worst_case["worst_order"])
print("Touches ground:", "yes" if touching_cases else "no")
print("Saved sweep results to", os.path.abspath(out_csv))
print("Saved scrape map to", os.path.abspath(scrape_map_png))
print()
print("Spring travel sweep summary")
print("Saved spring travel results to", os.path.abspath(travel_csv))
print("Front static spring travel:", f"{spring_travel_front * 1000.0:.2f} mm")
print("Rear static spring travel:", f"{spring_travel_rear * 1000.0:.2f} mm")
print("Max front outside travel:", f"{max_front_travel['front_outside_mm']:.2f} mm", f"@ {max_front_travel['ay_g']:.2f} g")
print("Max rear outside travel:", f"{max_rear_travel['rear_outside_mm']:.2f} mm", f"@ {max_rear_travel['ay_g']:.2f} g")

output = {
    "frontKS": frontKS,
    "rearKS":  rearKS,
    "frontRS": frontRS,
    "rearRS": rearRS,
    "frontKW": frontKW,
    "rearKW": rearKW,
    "roll_gradient_deg_per_g": roll_gradient,
    "pitch_gradient_deg_per_g": pitch_gradient
}
with open("spring_rates_output.json", "w") as f:
    json.dump(output, f)
print("Saved spring rates to spring_rates_output.json")
