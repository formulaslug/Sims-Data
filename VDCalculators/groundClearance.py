import numpy as np
import json

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


def spring_displacement(force, k):
    return force / k


def wheel_spring_constant(k_spring, motion_ratio):
    return k_spring * motion_ratio**2


def wheel_displacement(x_spring, motion_ratio):
    return x_spring / motion_ratio


def static_ride_height(h_design, x_wheel):
    return h_design - x_wheel


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
