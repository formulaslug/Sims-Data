import numpy as np
import json

IN_TO_M = 0.0254

mass = 293.97
g = 9.81
weight = mass * g

w_f = 0.4632
w_r = 1 - w_f

h_design_front = 0.040
h_design_rear = 0.040

# force per wheel
F_front = (weight * w_f) / 2
F_rear  = (weight * w_r) / 2

motionRatioF = 1.006
motionRatioR = 1.004

lowest_point_offset = 0.010

# Geometry points provided in inches as (x, y, z).
GEOMETRY_POINTS_IN = {
    "front_wing": (27.7744, -58.112, -1.1732),
    "floor": (26.6546, -2.9176, -1.2756),
    "floor_2": (26.6546, 28.7472, -1.2756),
    "rear_wing_thing": (-12.0, 47.0468, -1.2756),
}

WHEEL_DIAMETER_IN = 16.1
WHEEL_CENTER_Z_IN = 4.9064


def spring_displacement(force, k):
    return force / k


def wheel_spring_constant(k_spring, motion_ratio):
    return k_spring * motion_ratio**2


def wheel_displacement(x_spring, motion_ratio):
    return x_spring / motion_ratio


def static_ride_height(h_design, x_wheel):
    return h_design - x_wheel


def ground_plane_z_from_wheel(center_z_in, diameter_in):
    radius_in = diameter_in / 2.0
    return center_z_in - radius_in


def compute_point_clearances(points_in, ground_plane_z_in):
    clearances = {}
    for name, (_, _, z_in) in points_in.items():
        c_in = z_in - ground_plane_z_in
        clearances[name] = {
            "clearance_in": c_in,
            "clearance_mm": c_in * 25.4,
        }
    return clearances


def compute_ride_height(frontKS, rearKS,
                        h_des_f=h_design_front,
                        h_des_r=h_design_rear):

    # convert spring rate → wheel rate
    kwF = wheel_spring_constant(frontKS, motionRatioF)
    kwR = wheel_spring_constant(rearKS, motionRatioR)

    # wheel displacement from load
    xF = spring_displacement(F_front, kwF)
    xR = spring_displacement(F_rear, kwR)

    h_front = static_ride_height(h_des_f, xF)
    h_rear  = static_ride_height(h_des_r, xR)

    ground_clearance = min(h_front, h_rear) - lowest_point_offset

    return h_front, h_rear, ground_clearance


if __name__ == "__main__":

    with open("spring_rates_output.json") as f:
        rates = json.load(f)

    frontKS = rates["frontKS"]
    rearKS  = rates["rearKS"]

    print(f"Loaded front KS : {frontKS:.2f} N/m")
    print(f"Loaded rear  KS : {rearKS:.2f} N/m")

    h_front, h_rear, gc = compute_ride_height(frontKS, rearKS)

    print(f"\nFront static ride height : {h_front * 1000:.2f} mm")
    print(f"Rear  static ride height : {h_rear * 1000:.2f} mm")
    print(f"Ground clearance         : {gc * 1000:.2f} mm")

    # Absolute geometry-based clearances from tire contact plane.
    ground_plane_in = ground_plane_z_from_wheel(WHEEL_CENTER_Z_IN, WHEEL_DIAMETER_IN)
    point_clearances = compute_point_clearances(GEOMETRY_POINTS_IN, ground_plane_in)
    min_name = min(point_clearances, key=lambda k: point_clearances[k]["clearance_in"])
    min_clearance_in = point_clearances[min_name]["clearance_in"]

    print("\nGeometry-based clearances from ground plane:")
    print(f"Wheel ground plane z     : {ground_plane_in:.4f} in")
    for name, vals in point_clearances.items():
        print(
            f"  {name:16s}: {vals['clearance_in']:.4f} in ({vals['clearance_mm']:.2f} mm)"
        )
    print(
        f"Minimum point clearance  : {min_name} = {min_clearance_in:.4f} in "
        f"({min_clearance_in * 25.4:.2f} mm)"
    )
