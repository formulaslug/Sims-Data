import dumpling as tire
import numpy as np
import json

# ========== INPUT PARAMETERS (MODIFY HERE) ==========
# Variable ranges for optimization
NF_MIN = 200            # Minimum normal force (N)
NF_MAX = 3000           # Maximum normal force (N)
NF_POINTS = 20          # Number of points for normal force

SA_MIN = 0              # Minimum slip angle (degrees)
SA_MAX = 6             # Maximum slip angle (degrees)
SA_POINTS = 20          # Number of points for slip angle

TEMP_MIN = 80           # Minimum tire temperature (C)
TEMP_MAX = 105          # Maximum tire temperature (C)
TEMP_POINTS = 10        # Number of points for temperature

VEL_MIN = 20            # Minimum velocity (km/h)
VEL_MAX = 40            # Maximum velocity (km/h)
VEL_POINTS = 5         # Number of points for velocity

CAMBER_MIN = 0         # Minimum camber angle (rad) (0 degrees)
CAMBER_MAX = 0.02      # Maximum camber angle (rad) (~5 degrees)
CAMBER_POINTS = 3      # Number of points for camber

PRESSURE_MIN = 80       # Minimum tire pressure (kPa)
PRESSURE_MAX = 110      # Maximum tire pressure (kPa)
PRESSURE_POINTS = 10    # Number of points for pressure

SR_MIN = 0.0           # Minimum slip ratio
SR_MAX = 0.05           # Maximum slip ratio
SR_POINTS = 2          # Number of points for slip ratio

with open('params-LC0-6.json', 'r') as file:
    params = json.load(file)

def find_maximum_lateral_force():
    """Find the maximum lateral force within the given variable ranges"""
    normal_forces = np.linspace(NF_MIN, NF_MAX, NF_POINTS)
    slip_angles = np.linspace(SA_MIN, SA_MAX, SA_POINTS)
    temperatures = np.linspace(TEMP_MIN, TEMP_MAX, TEMP_POINTS)
    velocities = np.linspace(VEL_MIN, VEL_MAX, VEL_POINTS)
    cambers = np.linspace(CAMBER_MIN, CAMBER_MAX, CAMBER_POINTS)
    pressures = np.linspace(PRESSURE_MIN, PRESSURE_MAX, PRESSURE_POINTS)
    slip_ratios = np.linspace(SR_MIN, SR_MAX, SR_POINTS)

    max_lateral_force = 0
    optimal_params = {}

    total_combinations = (NF_POINTS * SA_POINTS * TEMP_POINTS *
                         VEL_POINTS * CAMBER_POINTS * PRESSURE_POINTS * SR_POINTS)

    print(f"Searching for maximum lateral force...")
    print(f"Variable Ranges:")
    print(f"  Normal Force: {NF_MIN} - {NF_MAX} N ({NF_POINTS} points)")
    print(f"  Slip Angle: {SA_MIN} - {SA_MAX}° ({SA_POINTS} points)")
    print(f"  Temperature: {TEMP_MIN} - {TEMP_MAX}°C ({TEMP_POINTS} points)")
    print(f"  Velocity: {VEL_MIN} - {VEL_MAX} km/h ({VEL_POINTS} points)")
    print(f"  Camber: {CAMBER_MIN:.3f} - {CAMBER_MAX:.3f} rad ({CAMBER_POINTS} points)")
    print(f"  Pressure: {PRESSURE_MIN} - {PRESSURE_MAX} kPa ({PRESSURE_POINTS} points)")
    print(f"  Slip Ratio: {SR_MIN} - {SR_MAX} ({SR_POINTS} points)")
    print(f"Total combinations: {total_combinations:,}")
    print("-" * 50)

    combination_count = 0

    for nf in normal_forces:
        for sa in slip_angles:
            for temp in temperatures:
                for vel in velocities:
                    for camber in cambers:
                        for pressure in pressures:
                            for sr in slip_ratios:
                                combination_count += 1

                                if combination_count % 10000 == 0:
                                    print(f"Progress: {combination_count:,}/{total_combinations:,} ({100*combination_count/total_combinations:.1f}%)")

                                runTire = tire.Tire(
                                    nf, sr, sa, vel,
                                    temp, pressure, camber,
                                    params["Mechanical-Parameters"],
                                    params["Magic-Parameters"]
                                )
                                lateral_force = abs(runTire.getLateralForce())

                                if lateral_force > max_lateral_force:
                                    max_lateral_force = lateral_force
                                    optimal_params = {
                                        'normal_force': nf,
                                        'slip_angle': sa,
                                        'temperature': temp,
                                        'velocity': vel,
                                        'camber': camber,
                                        'pressure': pressure,
                                        'slip_ratio': sr
                                    }

    return max_lateral_force, optimal_params

def find_max_for_fixed_normal_force(normal_force):
    """Find maximum lateral force for a specific normal force across all other variables"""
    slip_angles = np.linspace(SA_MIN, SA_MAX, SA_POINTS)
    temperatures = np.linspace(TEMP_MIN, TEMP_MAX, TEMP_POINTS)
    velocities = np.linspace(VEL_MIN, VEL_MAX, VEL_POINTS)
    cambers = np.linspace(CAMBER_MIN, CAMBER_MAX, CAMBER_POINTS)
    pressures = np.linspace(PRESSURE_MIN, PRESSURE_MAX, PRESSURE_POINTS)
    slip_ratios = np.linspace(SR_MIN, SR_MAX, SR_POINTS)

    max_lateral_force = 0
    optimal_params = {}

    for sa in slip_angles:
        for temp in temperatures:
            for vel in velocities:
                for camber in cambers:
                    for pressure in pressures:
                        for sr in slip_ratios:
                            runTire = tire.Tire(
                                normal_force, sr, sa, vel,
                                temp, pressure, camber,
                                params["Mechanical-Parameters"],
                                params["Magic-Parameters"]
                            )
                            lateral_force = abs(runTire.getLateralForce())

                            if lateral_force > max_lateral_force:
                                max_lateral_force = lateral_force
                                optimal_params = {
                                    'slip_angle': sa,
                                    'temperature': temp,
                                    'velocity': vel,
                                    'camber': camber,
                                    'pressure': pressure,
                                    'slip_ratio': sr
                                }

    return max_lateral_force, optimal_params

def print_parameters():
    """Print current parameter settings"""
    print("=" * 50)
    print("TIRE MAXIMUM LATERAL FORCE FINDER")
    print("=" * 50)
    print(f"Search Ranges:")
    print(f"  Normal Force: {NF_MIN} - {NF_MAX} N ({NF_POINTS} points)")
    print(f"  Slip Angle: {SA_MIN} - {SA_MAX}° ({SA_POINTS} points)")
    print(f"  Temperature: {TEMP_MIN} - {TEMP_MAX}°C ({TEMP_POINTS} points)")
    print(f"  Velocity: {VEL_MIN} - {VEL_MAX} km/h ({VEL_POINTS} points)")
    print(f"  Camber: {CAMBER_MIN:.3f} - {CAMBER_MAX:.3f} rad ({CAMBER_POINTS} points)")
    print(f"  Pressure: {PRESSURE_MIN} - {PRESSURE_MAX} kPa ({PRESSURE_POINTS} points)")
    print(f"  Slip Ratio: {SR_MIN} - {SR_MAX} ({SR_POINTS} points)")

    total_combinations = (NF_POINTS * SA_POINTS * TEMP_POINTS *
                         VEL_POINTS * CAMBER_POINTS * PRESSURE_POINTS * SR_POINTS)
    print(f"Total combinations: {total_combinations:,}")
    print("=" * 50)

if __name__ == "__main__":
    print_parameters()

    # Find overall maximum
    max_force, optimal_params = find_maximum_lateral_force()

    print(f"\nMAXIMUM LATERAL FORCE RESULTS:")
    print(f"Maximum Lateral Force: {max_force:.2f} N")
    print(f"Optimal Parameters:")
    print(f"  Normal Force: {optimal_params['normal_force']:.2f} N")
    print(f"  Slip Angle: {optimal_params['slip_angle']:.2f}°")
    print(f"  Temperature: {optimal_params['temperature']:.1f}°C")
    print(f"  Velocity: {optimal_params['velocity']:.1f} km/h")
    print(f"  Camber: {optimal_params['camber']:.3f} rad ({optimal_params['camber']*180/np.pi:.1f}°)")
    print(f"  Pressure: {optimal_params['pressure']:.1f} kPa")
    print(f"  Slip Ratio: {optimal_params['slip_ratio']:.3f}")
    print("=" * 50)

    # Show maximum for several fixed normal force levels
    print(f"\nMaximum lateral force at specific normal force levels:")
    nf_levels = [500, 1000, 1500, 2000, 2500]

    for nf in nf_levels:
        if NF_MIN <= nf <= NF_MAX:
            max_lat, opt_params = find_max_for_fixed_normal_force(nf)
            print(f"NF = {nf:4.0f} N: Max Lat Force = {max_lat:6.2f} N")
            print(f"    SA = {opt_params['slip_angle']:5.2f}°, T = {opt_params['temperature']:4.1f}°C, V = {opt_params['velocity']:4.1f} km/h")
            print(f"    Camber = {opt_params['camber']:6.3f} rad, P = {opt_params['pressure']:5.1f} kPa, SR = {opt_params['slip_ratio']:5.3f}")

    print("=" * 50)
    print("To modify search parameters, edit the values at the top of this file.")
