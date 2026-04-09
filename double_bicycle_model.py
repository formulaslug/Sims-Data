"""
Double Bicycle Yaw Rate Model for Formula Slug

2DOF model (lateral velocity + yaw rate) for basic vehicle dynamics.
Based on Rajamani's bicycle model.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
import matplotlib.pyplot as plt


@dataclass
class VehicleParameters:
    """Vehicle parameters from 2025 FSAE Design Spec Sheet (Car #216)"""

    # Geometry (meters) - from 2025 FSAE Design EV Spec Sheet
    wheelbase: float = 1.589989  # 1589.989 mm
    Lf: float = 0.8535  # Calculated from 46.32% front weight distribution
    Lr: float = 0.7365  # Calculated from weight distribution
    track_width_front: float = 1.234008  # 1234.008 mm
    track_width_rear: float = 1.186  # 1186 mm
    track_width: float = 1.234008  # Using front track for compatibility

    # Mass properties (with driver)
    mass: float = 300.0  # Vehicle + driver (from Formula Slug FS-4 specs)
    mass_without_driver: float = 232.0
    cg_height: float = 0.3048  # 304.8 mm
    yaw_inertia: float = 658.09  # Not in spec sheet, using previous value

    # Tire model (stiffness values not in spec sheet, using estimates)
    # Tires: Front - Hoosier 16.0x6.0-10 LC0, Rear - Hoosier 16.0x7.5-10 LC0
    C_alpha: float = 180000.0  # Cornering stiffness estimate
    C_alpha_front: float = 180000.0
    C_alpha_rear: float = 180000.0
    mu: float = 1.733
    longitudinal_inertia: float = 0.0

    # Additional spec sheet info (for reference)
    # Wheel rates: Front 30.647 N/mm, Rear 35.025 N/mm
    # Roll rates: Front 407.25 Nm/deg, Rear 429.93 Nm/deg
    # Natural frequency: Front 3.55 Hz, Rear 3.53 Hz

    def __post_init__(self):
        tolerance = 0.0001
        wheelbase_check = self.Lf + self.Lr
        assert abs(wheelbase_check - self.wheelbase) < tolerance, \
            f"Wheelbase math is wrong: {self.Lf} + {self.Lr} = {wheelbase_check} != {self.wheelbase}"
        assert self.mass > 0, "Mass has to be positive"
        assert self.yaw_inertia > 0, "Yaw inertia has to be positive"
        assert self.C_alpha > 0, "Tire stiffness has to be positive"


@dataclass
class TireModel:
    stiffness: float
    max_slip_angle: float = 0.3
    saturation_method: str = "linear"
    friction_coeff: float = 1.733  # Peak friction coefficient

    def lateral_force(self, slip_angle: float, vertical_load: float) -> float:
        if self.saturation_method == "linear":
            return self.stiffness * slip_angle

        elif self.saturation_method == "nonlinear":
            # tanh saturation at high slip angles
            # Saturates at F_max = mu * Fz (friction limit)
            F_linear = self.stiffness * slip_angle
            F_max = self.friction_coeff * vertical_load
            return F_max * np.tanh(F_linear / F_max)

        return 0.0


class DoubleBicycleModel:
    """2DOF bicycle model: v_y (lateral velocity) and r (yaw rate)"""

    def __init__(self, params: VehicleParameters, tire_model: str = "linear"):
        self.params = params
        self.tire_front = TireModel(params.C_alpha_front, saturation_method=tire_model, friction_coeff=params.mu)
        self.tire_rear = TireModel(params.C_alpha_rear, saturation_method=tire_model, friction_coeff=params.mu)

        self.state = np.array([0.0, 0.0])
        self.time_history = []
        self.state_history = []
        self.input_history = []
    
    def get_slip_angles(self, v_y: float, r: float, v_x: float, delta: float) \
            -> Tuple[float, float]:
        """Calculate front and rear slip angles"""
        if abs(v_x) < 0.1:
            return delta, 0.0

        alpha_f = delta - np.arctan2(v_y + self.params.Lf * r, v_x)
        alpha_r = -np.arctan2(v_y - self.params.Lr * r, v_x)

        return alpha_f, alpha_r
    
    def dynamics(self, state: np.ndarray, v_x: float, delta: float,
                 ax: float = 0.0) -> np.ndarray:
        """Compute state derivatives [dv_y/dt, dr/dt]"""
        v_y, r = state

        alpha_f, alpha_r = self.get_slip_angles(v_y, r, v_x, delta)

        # Static weight distribution (no load transfer)
        # Front axle load: weight farther back (at distance Lr from front)
        # Rear axle load: weight farther forward (at distance Lf from rear)
        Fz_f = self.params.mass * 9.81 * self.params.Lr / self.params.wheelbase / 2.0
        Fz_r = self.params.mass * 9.81 * self.params.Lf / self.params.wheelbase / 2.0

        Fy_f = self.tire_front.lateral_force(alpha_f, Fz_f)
        Fy_r = self.tire_rear.lateral_force(alpha_r, Fz_r)

        # Account for both tires per axle
        Fy_f_total = 2.0 * Fy_f
        Fy_r_total = 2.0 * Fy_r

        # Lateral and yaw accelerations
        a_y = (Fy_f_total * np.cos(delta) + Fy_r_total) / self.params.mass + v_x * r
        dv_y = a_y

        M_yaw = self.params.Lf * Fy_f_total * np.cos(delta) - self.params.Lr * Fy_r_total
        dr = M_yaw / self.params.yaw_inertia

        return np.array([dv_y, dr])
    
    def integrate_step(self, v_x: float, delta: float, dt: float,
                      ax: float = 0.0, method: str = "rk4") -> None:
        """Integrate one timestep using euler, rk2, or rk4"""
        if method == "euler":
            k1 = self.dynamics(self.state, v_x, delta, ax)
            self.state = self.state + dt * k1

        elif method == "rk2":
            k1 = self.dynamics(self.state, v_x, delta, ax)
            k2 = self.dynamics(self.state + 0.5*dt*k1, v_x, delta, ax)
            self.state = self.state + dt * k2

        elif method == "rk4":
            k1 = self.dynamics(self.state, v_x, delta, ax)
            k2 = self.dynamics(self.state + 0.5*dt*k1, v_x, delta, ax)
            k3 = self.dynamics(self.state + 0.5*dt*k2, v_x, delta, ax)
            k4 = self.dynamics(self.state + dt*k3, v_x, delta, ax)
            self.state = self.state + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

        else:
            raise ValueError(f"Unknown integration method: {method}")
    
    def simulate(self, v_x: float, steering_inputs: List[float],
                dt: float = 0.01, method: str = "rk4") -> Tuple[np.ndarray, np.ndarray]:
        """Run simulation with given steering input sequence"""
        self.state = np.array([0.0, 0.0])
        self.time_history = [0.0]
        self.state_history = [self.state.copy()]
        self.input_history = [(0.0, v_x)]

        for i, delta in enumerate(steering_inputs):
            t = (i + 1) * dt
            self.integrate_step(v_x, delta, dt, ax=0.0, method=method)

            self.time_history.append(t)
            self.state_history.append(self.state.copy())
            self.input_history.append((delta, v_x))

        return np.array(self.time_history), np.array(self.state_history)

    def reset(self):
        self.state = np.array([0.0, 0.0])
        self.time_history = []
        self.state_history = []
        self.input_history = []


def validate_against_telemetry(csv_path: str, model: DoubleBicycleModel,
                              sample_window: int = 1000) -> dict:
    """Load telemetry and extract stats for model comparison"""
    try:
        import pandas as pd
    except ImportError:
        print("Pandas required. Install with: pip install pandas")
        return {}

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        print(f"Failed to load {csv_path}: {e}")
        return {}

    results = {
        'samples_loaded': len(df),
        'columns_found': [],
        'validation_data': {}
    }

    if 'VDM_Z_AXIS_YAW_RATE' in df.columns:
        results['columns_found'].append('yaw_rate')
        yaw_telem = df['VDM_Z_AXIS_YAW_RATE'].dropna()

        if len(yaw_telem) > 0:
            results['validation_data']['yaw_rate_min'] = yaw_telem.min()
            results['validation_data']['yaw_rate_max'] = yaw_telem.max()
            results['validation_data']['yaw_rate_mean'] = yaw_telem.mean()
            results['validation_data']['yaw_rate_std'] = yaw_telem.std()

    if 'VDM_Y_AXIS_ACCELERATION' in df.columns:
        results['columns_found'].append('lateral_accel')
        lat_accel = df['VDM_Y_AXIS_ACCELERATION'].dropna()

        if len(lat_accel) > 0:
            results['validation_data']['lat_accel_min'] = lat_accel.min()
            results['validation_data']['lat_accel_max'] = lat_accel.max()
            results['validation_data']['lat_accel_mean'] = lat_accel.mean()
            results['validation_data']['lat_accel_max_g'] = lat_accel.max() / 9.81

    if 'SME_TRQSPD_Speed' in df.columns:
        results['columns_found'].append('speed')
        speed = df['SME_TRQSPD_Speed'].dropna()

        if len(speed) > 0:
            results['validation_data']['speed_min'] = speed.min()
            results['validation_data']['speed_max'] = speed.max()
            results['validation_data']['speed_mean'] = speed.mean()

    return results


def plot_response(model: DoubleBicycleModel, title: str = "Model Response"):
    if not model.time_history:
        print("No data. Run simulate() first.")
        return

    time = np.array(model.time_history)
    states = np.array(model.state_history)
    steering = [inp[0] for inp in model.input_history]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    axes[0].plot(time, np.rad2deg(steering), 'b-', linewidth=2)
    axes[0].set_ylabel('Steering Angle (°)', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Steering Input')

    axes[1].plot(time, states[:, 0], 'g-', linewidth=2, label='Lateral Velocity')
    axes[1].set_ylabel('Lateral Velocity (m/s)', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_title('Lateral Velocity')

    axes[2].plot(time, np.rad2deg(states[:, 1]), 'r-', linewidth=2, label='Yaw Rate')
    axes[2].set_ylabel('Yaw Rate (°/s)', fontsize=11)
    axes[2].set_xlabel('Time (s)', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].set_title('Yaw Rate')

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("\n--- FORMULA SLUG - DOUBLE BICYCLE YAW RATE MODEL ---\n")

    params = VehicleParameters()
    print("Vehicle Parameters:")
    print(f"  Wheelbase: {params.wheelbase:.3f} m (Lf={params.Lf:.3f}, Lr={params.Lr:.3f})")
    print(f"  Track width: {params.track_width:.3f} m")
    print(f"  Mass: {params.mass:.1f} kg")
    print(f"  Yaw inertia: {params.yaw_inertia:.1f} kg·m²")
    print(f"  Tire stiffness: {params.C_alpha:.0f} N/rad")

    model = DoubleBicycleModel(params, tire_model="linear")

    # Test 1: Step steer
    print("\nTEST 1: Step Steer")
    v_x = 10.0
    duration = 3.0
    dt = 0.01
    num_steps = int(duration / dt)

    steering_step = np.concatenate([
        np.zeros(int(0.2 / dt)),
        np.full(num_steps - int(0.2 / dt), np.deg2rad(5.7))
    ])
    steering_step = steering_step[:num_steps]

    time_sim, states_sim = model.simulate(v_x, steering_step, dt=dt, method="rk4")

    print(f"\nSpeed: {v_x:.1f} m/s ({v_x*3.6:.1f} km/h)")
    print(f"Duration: {duration:.2f} s")
    print(f"Results at end:")
    print(f"  Lateral velocity: {states_sim[-1, 0]:.3f} m/s")
    print(f"  Yaw rate: {np.rad2deg(states_sim[-1, 1]):.2f} °/s")
    print(f"  G-force: {v_x * states_sim[-1, 1] / 9.81:.3f}g")

    fig1 = plot_response(model, "Test 1: Step Steering")
    fig1.savefig('/Users/brianlee/vscode_projects/formula_slug/fs_yawratemodel/test1_step_steer.png', dpi=150)
    print("Saved to test1_step_steer.png")

    # Test 2: Ramp steer
    print("\nTEST 2: Ramp Steer")
    model.reset()
    v_x = 8.0
    duration = 5.0
    num_steps = int(duration / dt)

    steering_ramp = np.concatenate([
        np.linspace(0, np.deg2rad(10), int(1.0 / dt)),
        np.full(num_steps - int(1.0 / dt), np.deg2rad(10))
    ])
    steering_ramp = steering_ramp[:num_steps]

    time_sim, states_sim = model.simulate(v_x, steering_ramp, dt=dt, method="rk4")

    print(f"\nSpeed: {v_x:.1f} m/s ({v_x*3.6:.1f} km/h)")
    print(f"Ramp rate: {np.rad2deg(10):.1f}°/s")
    print(f"Results at end:")
    print(f"  Lateral velocity: {states_sim[-1, 0]:.3f} m/s")
    print(f"  Yaw rate: {np.rad2deg(states_sim[-1, 1]):.2f} °/s")
    print(f"  Steady-state G: {v_x * states_sim[-1, 1] / 9.81:.3f}g")

    fig2 = plot_response(model, "Test 2: Ramp Steering")
    fig2.savefig('/Users/brianlee/vscode_projects/formula_slug/fs_yawratemodel/test2_ramp_steer.png', dpi=150)
    print("Saved to test2_ramp_steer.png")

    # Test 3: Lane change
    print("\nTEST 3: Double Lane Change")
    model.reset()
    v_x = 10.0
    duration = 4.0
    num_steps = int(duration / dt)

    freq = 0.5
    steering_dlc = np.deg2rad(10) * np.sin(2 * np.pi * freq * np.linspace(0, duration, num_steps))

    time_sim, states_sim = model.simulate(v_x, steering_dlc, dt=dt, method="rk4")

    print(f"\nSpeed: {v_x:.1f} m/s ({v_x*3.6:.1f} km/h)")
    print(f"Frequency: {freq:.1f} Hz (±10° amplitude)")
    print(f"Results:")
    print(f"  Max lateral velocity: {np.max(np.abs(states_sim[:, 0])):.3f} m/s")
    print(f"  Max yaw rate: {np.rad2deg(np.max(np.abs(states_sim[:, 1]))):.2f} °/s")
    print(f"  Max G-force: {v_x * np.max(np.abs(states_sim[:, 1])) / 9.81:.3f}g")

    fig3 = plot_response(model, "Test 3: Double Lane Change")
    fig3.savefig('/Users/brianlee/vscode_projects/formula_slug/fs_yawratemodel/test3_double_lanechange.png', dpi=150)
    print("Saved to test3_double_lanechange.png")

    print("\n--- Done! ---")

    # Telemetry validation
    print("\nTELEMETRY VALIDATION")

    try:
        import pandas as pd

        print("\nLoading telemetry from 08102025Endurance1_FirstHalf.csv...")
        telemetry_file = '/Users/brianlee/vscode_projects/formula_slug/workshops/data_for_dwshop/08102025Endurance1_FirstHalf.csv'
    
        df = pd.read_csv(telemetry_file, low_memory=False)
        print(f"Loaded {len(df)} samples")

        columns_needed = [
            'VDM_X_AXIS_ACCELERATION',
            'VDM_Y_AXIS_ACCELERATION',
            'VDM_Z_AXIS_YAW_RATE',
            'SME_TRQSPD_Speed',
            'TPERIPH_FL_DATA_WHEELSPEED',
            'TPERIPH_FR_DATA_WHEELSPEED',
            'TPERIPH_BL_DATA_WHEELSPEED',
            'TPERIPH_BR_DATA_WHEELSPEED',
        ]

        available_cols = [col for col in columns_needed if col in df.columns]
        print(f"Found {len(available_cols)}/{len(columns_needed)} expected columns")

        if 'VDM_Z_AXIS_YAW_RATE' not in df.columns:
            print("  x VDM_Z_AXIS_YAW_RATE not found")
        else:
            print("  + VDM_Z_AXIS_YAW_RATE found")

        if 'VDM_Y_AXIS_ACCELERATION' not in df.columns:
            print("  x VDM_Y_AXIS_ACCELERATION not found")
        else:
            print("  + VDM_Y_AXIS_ACCELERATION found")

        if 'SME_TRQSPD_Speed' not in df.columns:
            print("  x SME_TRQSPD_Speed not found")
        else:
            print("  + SME_TRQSPD_Speed found")

        if 'VDM_Z_AXIS_YAW_RATE' in df.columns:
            valid_yaw = df[df['VDM_Z_AXIS_YAW_RATE'].notna()]['VDM_Z_AXIS_YAW_RATE']

            if len(valid_yaw) > 0:
                print(f"\nYaw rate statistics from telemetry:")
                print(f"  Min: {valid_yaw.min():.1f} °/s ({np.deg2rad(valid_yaw.min()):.3f} rad/s)")
                print(f"  Max: {valid_yaw.max():.1f} °/s ({np.deg2rad(valid_yaw.max()):.3f} rad/s)")
                print(f"  Mean: {valid_yaw.mean():.1f} °/s ({np.deg2rad(valid_yaw.mean()):.3f} rad/s)")
                print(f"  Std: {valid_yaw.std():.1f} °/s ({np.deg2rad(valid_yaw.std()):.3f} rad/s)")

                high_yaw_mask = np.abs(valid_yaw) > 10.0  # 10 °/s threshold for turning
                if high_yaw_mask.any():
                    high_idx = np.where(high_yaw_mask)[0][0]
                    print(f"\nFound turning maneuver at sample {high_idx}")
                    yaw_val = valid_yaw.iloc[high_idx]
                    print(f"  Yaw rate: {yaw_val:.1f} °/s ({np.deg2rad(yaw_val):.3f} rad/s)")

        print("\nTelemetry validation framework ready")

    except ImportError:
        print("Pandas not installed - can't load CSV")
        print("Install with: pip install pandas")
    except Exception as e:
        print(f"Error loading telemetry: {e}")

    plt.show()
