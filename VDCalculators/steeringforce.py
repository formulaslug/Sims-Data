"""
  - tireLoad   : Tuple of tire loads (FL, FR, RL, RR) in N
  - slipRatio  : Slip ratio (dimensionless)
  - speed      : Vehicle speed (m/s)
  - surfaceTemperature : Surface temperature (C)
  - tirePressure : Tire pressure (kPa)
  - alpha_deg  : Slip angle (degrees)
  - s_m        : Mechanical trail (m)
  - L_arm      : Steering arm length (m)
  - caster_deg : Caster angle (degrees)
  - SR         : Steering ratio (dimensionless)
  - r_sw       : Steering wheel radius (m)

"""

import math
from Mech.traction import calcCorneringStiffness
from Mech.tireLoad import calcLoadTransfer
from Mech.steering import calcSlipAngle
from paramLoader import Parameters, Magic


def compute_steering_forces(
    # Tyre — from team cornering stiffness calculator
    tireLoad: tuple,
    slipRatio: float,
    speed: float,
    surfaceTemperature: float,
    tirePressure: float,
    alpha_deg: float = 3.0,
    # Mz: float = ,       
    # Geometry
    s_m: float = 0.002598,
    L_arm: float = 0.06698,
    caster_deg: float = 7.23,
    # Steering ratio
    SR: float = 3.81,
    r_sw: float = 0.15,
) -> dict:
    """
    Steering force pipeline:

      C_alpha  = calcCorneringStiffness(...)
      Fy       = C_alpha * alpha
      M_axis   = Fy * s_m * cos(caster)
      F_rack   = M_axis / L_arm
      T_wheel  = M_axis / SR
      F_driver = T_wheel / r_sw

    """

    results = {}

    # Compute C_alpha from team calculator
    slip_tuple = (math.radians(alpha_deg), math.radians(alpha_deg))
    C_alpha, _ = calcCorneringStiffness(tireLoad, slip_tuple, slipRatio, speed, surfaceTemperature, tirePressure)
    results["C_alpha"] = C_alpha

    # STEP 1: Lateral tyre force (per tire)
    # Fy = C_alpha * alpha
    alpha_rad = math.radians(alpha_deg)
    Fy = C_alpha * alpha_rad
    results["alpha_deg"] = alpha_deg
    results["alpha_rad"] = alpha_rad
    results["Fy_N"] = Fy

    #pneumatic poopoo
    # t_p = Mz / Fy
    # results["Mz_Nm"] = Mz
    # results["t_p_m"] = t_p

    # STEP 2: Steering axis moment
    # M_axis = Fy * s_m * cos(caster)

    cos_caster = math.cos(math.radians(caster_deg))
    results["caster_deg"] = caster_deg
    results["cos_caster"] = cos_caster
    M_axis = Fy * s_m * cos_caster
    results["M_axis_Nm"] = M_axis

    # STEP 3: Rack force
    # F_rack = M_axis / L_arm
    F_rack = M_axis / L_arm
    results["F_rack_N"] = F_rack

    # STEP 4: Steering wheel torque
    # T_wheel = M_axis / SR
    T_wheel = M_axis / SR
    results["SR"] = SR
    results["T_wheel_Nm"] = T_wheel

    # STEP 5: Driver rim force
    # F_driver = T_wheel / r_sw
    F_driver_N = T_wheel / r_sw
    results["r_sw_m"] = r_sw
    results["F_driver_N"] = F_driver_N

    return results


def print_results(results: dict):
    print("\n" + "=" * 58)
    print("RESULTS")
    print("=" * 58)

    labels = {
        # "Mz_Nm":     "Self-Aligning Torque            [Nm]",  
        # "t_p_m":     "Pneumatic Trail                  [m]",  
        "C_alpha":     "Cornering Stiffness          [N/rad]",
        "alpha_deg":   "Slip Angle                     [deg]",
        "alpha_rad":   "Slip Angle                     [rad]",
        "Fy_N":        "Lateral Tyre Force               [N]",
        "caster_deg":  "Caster Angle                   [deg]",
        "cos_caster":  "cos(caster)                        ",
        "M_axis_Nm":   "Steering Axis Moment            [Nm]",
        "F_rack_N":    "Rack Force                       [N]",
        "SR":          "Steering Ratio                     ",
        "T_wheel_Nm":  "Steering Wheel Torque           [Nm]",
        "r_sw_m":      "Steering Wheel Radius            [m]",
        "F_driver_N":  "Driver Rim Force                 [N]",
    }

    for key, label in labels.items():
        if key in results:
            print(f"  {label:<45} {results[key]:.6g}")

    print("=" * 58 + "\n")


tireLoad = calcLoadTransfer(0, 9.81, 0)  # ax, ay, yawRate — 1g cornering

results = compute_steering_forces(
    tireLoad           = tireLoad,
    slipRatio          = 0.15,
    speed              = 20,
    surfaceTemperature = 80,
    tirePressure       = 40,
)
print_results(results)