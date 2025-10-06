import dumpling as tire
import matplotlib.pyplot as plt
import numpy as np
import json

# ========== INPUT PARAMETERS (MODIFY HERE) ==========
TIRE_TEMP = 100          # Tire temperature (C)
CAMBER = 0 #1 * 3.1415/180 # Camber angle (rad)
PRESSURE = 80           # Tire pressure (kPa)
VELOCITY = 40           # Velocity (km/h)
SLIP_ANGLE = 6          # Slip angle (degrees)
SLIP_RATIO = 0       # Slip ratio

# Normal force range for plotting
NF_MIN = 200           # Minimum normal force (N)
NF_MAX = 3000          # Maximum normal force (N)
NF_POINTS = 50         # Number of points for normal force

# Slip angle range for lateral force calculation
SA_MIN = 0             # Minimum slip angle (degrees)
SA_MAX = 12            # Maximum slip angle (degrees)
SA_POINTS = 50         # Number of points for slip angle


filename = 'params-LC0-6.json'
with open(filename, 'r') as file:
    params = json.load(file)

def calculate_forces_vs_normal_force():
    """Calculate lateral force vs normal force for fixed slip angle"""
    normal_forces = np.linspace(NF_MIN, NF_MAX, NF_POINTS)
    lateral_forces = []

    for nf in normal_forces:
        runTire = tire.Tire(
            nf, SLIP_RATIO, SLIP_ANGLE, VELOCITY,
            TIRE_TEMP, PRESSURE, CAMBER,
            params["Mechanical-Parameters"],
            params["Magic-Parameters"]
        )
        lateral_force = runTire.getLateralForce()
        lateral_forces.append(lateral_force)

    print(normal_forces)
    print(lateral_forces)

    return normal_forces, lateral_forces

def calculate_forces_vs_slip_angle():
    """Calculate lateral force vs slip angle for different normal forces"""
    slip_angles = np.linspace(SA_MIN, SA_MAX, SA_POINTS)

    # Multiple normal force levels for comparison
    nf_levels = [300, 500, 700, 900, 1100, 1300, 1500]

    results = {}
    for nf in nf_levels:
        lateral_forces = []
        for sa in slip_angles:
            runTire = tire.Tire(
                nf, SLIP_RATIO, sa, VELOCITY,
                TIRE_TEMP, PRESSURE, CAMBER,
                params["Mechanical-Parameters"],
                params["Magic-Parameters"]
            )
            lateral_force = runTire.getLateralForce()
            lateral_forces.append(lateral_force)
        results[nf] = lateral_forces

    return slip_angles, results

def plot_nf_vs_lateral_force():
    """Create 2D plot of Normal Force vs Lateral Force"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Lateral Force vs Normal Force (fixed slip angle)
    nf_values, lat_forces = calculate_forces_vs_normal_force()
    ax1.plot(nf_values, lat_forces, 'b-', linewidth=2, label=f'SA = {SLIP_ANGLE}°')
    ax1.set_xlabel('Normal Force (N)')
    ax1.set_ylabel('Lateral Force (N)')
    titleString = f'Lateral Force vs Normal Force\n'
    titleString += filename
    ax1.set_title(titleString)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Lateral Force vs Slip Angle (multiple normal forces)
    sa_values, nf_results = calculate_forces_vs_slip_angle()
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'purple']

    for i, (nf, lat_forces) in enumerate(nf_results.items()):
        ax2.plot(sa_values, lat_forces, color=colors[i], linewidth=2, label=f'NF = {nf} N')

    ax2.set_xlabel('Slip Angle (degrees)')
    ax2.set_ylabel('Lateral Force (N)')
    ax2.set_title(f'Lateral Force vs Slip Angle\nTemp: {TIRE_TEMP}°C, Pressure: {PRESSURE} kPa, Camber: {CAMBER} rad')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    return fig

def print_parameters():
    """Print current parameter settings"""
    print("=" * 50)
    print("CURRENT TIRE PARAMETERS:")
    print("=" * 50)
    print(f"Tire Temperature: {TIRE_TEMP}°C")
    print(f"Camber Angle: {CAMBER}°")
    print(f"Tire Pressure: {PRESSURE} psi")
    print(f"Velocity: {VELOCITY} km/h")
    print(f"Slip Ratio: {SLIP_RATIO}")
    print(f"Slip Angle (Plot 1): {SLIP_ANGLE}°")
    print("-" * 50)
    print(f"Normal Force Range: {NF_MIN} - {NF_MAX} N")
    print(f"Slip Angle Range: {SA_MIN} - {SA_MAX}°")
    print("=" * 50)
    print("\nTo modify parameters, edit the values at the top of this file.")
    print("=" * 50)

if __name__ == "__main__":
    print_parameters()

    fig = plot_nf_vs_lateral_force()
    plt.title(filename)
    plt.show()

    # Print some sample calculations
    # print(f"\nSample calculation at NF=500N, SA={SLIP_ANGLE}°:")
    # sample_tire = tire.Tire(
    #     500, SLIP_RATIO, SLIP_ANGLE, VELOCITY,
    #     TIRE_TEMP, PRESSURE, CAMBER,
    #     params["Mechanical-Parameters"],
    #     params["Magic-Parameters"]
    # )
    # print(f"Lateral Force: {sample_tire.getLateralForce():.2f} N")
