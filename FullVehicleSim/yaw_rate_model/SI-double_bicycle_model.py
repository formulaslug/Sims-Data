"""
Double Bicycle Yaw Rate Model for Formula Slug

2DOF model (lateral velocity + yaw rate) for basic vehicle dynamics.
Based on Rajamani's bicycle model.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List
import matplotlib.pyplot as plt
from pathlib import Path
import unyt as u

HERE = Path(__file__).parent

# --- Unit checking ---
CHECK_UNITS = True   # True: values include units and are checked (debugging). False: plain floats (full speed)

def unit(value, units):
    """Attaches units to a value, or returns a plain SI float when CHECK_UNITS is off."""
    quantity = value * u.Unit(units)
    if CHECK_UNITS:
        return quantity
    si = quantity.in_mks().v
    return float(si) if np.ndim(si) == 0 else si

RAD = u.rad if CHECK_UNITS else 1.0   # used to add or remove radians in rotational math

# The state array is [v_y, r]. These are the units of each slot
STATE_UNITS      = [u.Unit("m/s"),    u.Unit("rad/s")]      # [lateral velocity, yaw rate]
DERIVATIVE_UNITS = [u.Unit("m/s**2"), u.Unit("rad/s**2")]   # [dv_y/dt, dr/dt], what dynamics() returns

# Custom unit handling
u.define_unit("gee", 9.80665 * u.m / u.s**2)

# attach_units() turns plain numbers into unit values (like get() in the full sim).
# strip_units() converts each value to its expected unit, then returns plain numbers (like put()).
if CHECK_UNITS:
    def attach_units(values, units):
        return [u.unyt_quantity(v, unit) for v, unit in zip(values, units)]

    def strip_units(values, units):
        return np.array([v.to_value(unit) for v, unit in zip(values, units)])
else:
    def attach_units(values, units):
        return values

    def strip_units(values, units):
        return np.array(values)

def with_units(value, units):
    """For printing/plotting: gives a unit value in either mode (attaches units only if missing)."""
    return value if isinstance(value, u.unyt_array) else value * u.Unit(units)


@dataclass
class VehicleParameters:
    """Vehicle parameters from 2025 FSAE Design Spec Sheet (Car #216)"""

    # Geometry (meters) - from 2025 FSAE Design EV Spec Sheet
    wheelbase: float = field(default_factory=lambda: unit(1.589989, "m"))  # 1589.989 mm
    Lf: float = field(default_factory=lambda: unit(0.8535, "m"))  # Calculated from 46.32% front weight distribution
    Lr: float = field(default_factory=lambda: unit(0.7365, "m"))  # Calculated from weight distribution
    track_width_front: float = field(default_factory=lambda: unit(1.234008, "m"))  # 1234.008 mm
    track_width_rear: float = field(default_factory=lambda: unit(1.186, "m"))  # 1186 mm
    track_width: float = field(default_factory=lambda: unit(1.234008, "m"))  # Using front track for compatibility

    # Mass properties (with driver)
    mass: float = field(default_factory=lambda: unit(300.0, "kg"))  # Vehicle + driver (from Formula Slug FS-4 specs)
    mass_without_driver: float = field(default_factory=lambda: unit(232.0, "kg"))
    cg_height: float = field(default_factory=lambda: unit(0.3048, "m"))  # 304.8 mm
    yaw_inertia: float = field(default_factory=lambda: unit(658.09, "kg*m**2"))  # Not in spec sheet, using previous value
    
    # Tire model (stiffness values not in spec sheet, using estimates)
    # Tires: Front - Hoosier 16.0x6.0-10 LC0, Rear - Hoosier 16.0x7.5-10 LC0
    C_alpha: float = field(default_factory=lambda: unit(180000.0, "N/rad"))  # Cornering stiffness estimate
    C_alpha_front: float = field(default_factory=lambda: unit(180000.0, "N/rad"))
    C_alpha_rear: float = field(default_factory=lambda: unit(180000.0, "N/rad"))
    mu: float = 1.733
    longitudinal_inertia: float = field(default_factory=lambda: unit(0.0, "kg*m**2"))

    # Additional spec sheet info (for reference)
    # Wheel rates: Front 30.647 N/mm, Rear 35.025 N/mm
    # Roll rates: Front 407.25 Nm/deg, Rear 429.93 Nm/deg
    # Natural frequency: Front 3.55 Hz, Rear 3.53 Hz

    def __post_init__(self):
        tolerance = unit(0.0001, "m")
        wheelbase_check = self.Lf + self.Lr
        assert abs(wheelbase_check - self.wheelbase) < tolerance, \
            f"Wheelbase math is wrong: {self.Lf} + {self.Lr} = {wheelbase_check} != {self.wheelbase}"
        assert self.mass > 0, "Mass has to be positive"
        assert self.yaw_inertia > 0, "Yaw inertia has to be positive"
        assert self.C_alpha > 0, "Tire stiffness has to be positive"


@dataclass
class TireModel:
    stiffness: float
    max_slip_angle: float = field(default_factory=lambda: unit(0.3, "rad"))
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

        return unit(0.0, "N")


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
    
    def get_slip_angles(self, v_y: u.unyt_quantity, r: u.unyt_quantity, v_x: u.unyt_quantity, delta: u.unyt_quantity) \
            -> Tuple[u.unyt_quantity, u.unyt_quantity]:
        """Calculate front and rear slip angles"""
        if abs(v_x) < unit(0.1, "m/s"):
            return delta, 0.0 * RAD

        alpha_f = delta - np.arctan2(v_y + self.params.Lf * r / RAD, v_x) * RAD
        alpha_r = -np.arctan2(v_y - self.params.Lr * r / RAD, v_x) * RAD

        return alpha_f, alpha_r
    
    def dynamics(self, 
                 state: u.unyt_quantity, 
                 v_x: u.unyt_quantity,
                 delta: u.unyt_quantity,
                 ax: u.unyt_quantity = unit(0.0, "m/s**2")) -> np.ndarray:
        """Compute state derivatives [dv_y/dt, dr/dt]"""
        v_y, r = attach_units(state, STATE_UNITS)

        alpha_f, alpha_r = self.get_slip_angles(v_y, r, v_x, delta)

        # Static weight distribution (no load transfer)
        # Front axle load: weight farther back (at distance Lr from front)
        # Rear axle load: weight farther forward (at distance Lf from rear)

        g = unit(9.81, "m/s**2")
        Fz_f = self.params.mass * g * self.params.Lr / self.params.wheelbase / 2.0
        Fz_r = self.params.mass * g * self.params.Lf / self.params.wheelbase / 2.0

        Fy_f = self.tire_front.lateral_force(alpha_f, Fz_f)
        Fy_r = self.tire_rear.lateral_force(alpha_r, Fz_r)

        # Account for both tires per axle
        Fy_f_total = 2.0 * Fy_f
        Fy_r_total = 2.0 * Fy_r

        # Lateral and yaw accelerations
        a_y = (Fy_f_total * np.cos(delta) + Fy_r_total) / self.params.mass + v_x * r / RAD
        dv_y = a_y

        M_yaw = self.params.Lf * Fy_f_total - self.params.Lr * Fy_r_total
        dr = M_yaw / self.params.yaw_inertia * RAD

        return strip_units([dv_y, dr], DERIVATIVE_UNITS)
    
    def integrate_step(self, v_x: u.unyt_quantity, delta: u.unyt_quantity, dt: u.unyt_quantity,
                      ax: u.unyt_quantity = unit(0.0, "m/s**2"), method: str = "rk4") -> None:
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
    
    def simulate(self, v_x: u.unyt_quantity, steering_inputs: List[u.unyt_quantity],
                dt: u.unyt_quantity = unit(0.01, "s"), method: str = "rk4") -> Tuple[np.ndarray, np.ndarray]:
        """Run simulation with given steering input sequence"""

        dt_s = float(with_units(dt, "s").to_value("s"))
        v_x_si = float(with_units(v_x, "m/s").to_value("m/s"))
        
        self.state = np.array([0.0, 0.0])
        self.time_history = [0.0]
        self.state_history = [self.state.copy()]
        self.input_history = [(0.0, v_x_si)]

        for i, delta in enumerate(steering_inputs):
            t = (i + 1) * dt_s
            self.integrate_step(v_x, delta, dt_s, ax=unit(0.0, "m/s**2"), method=method)

            self.time_history.append(t)
            self.state_history.append(self.state.copy())
            self.input_history.append((float(with_units(delta, "rad").to_value("rad")), v_x_si))

        return np.array(self.time_history), np.array(self.state_history)

    def reset(self):
        self.state = np.array([0.0, 0.0])
        self.time_history = []
        self.state_history = []
        self.input_history = []


# def validate_against_telemetry(csv_path: str, model: DoubleBicycleModel,
#                               sample_window: int = 1000) -> dict:
#     """Load telemetry and extract stats for model comparison"""
#     try:
#         import pandas as pd
#     except ImportError:
#         print("Pandas required. Install with: pip install pandas")
#         return {}

#     try:
#         df = pd.read_csv(csv_path, low_memory=False)
#     except Exception as e:
#         print(f"Failed to load {csv_path}: {e}")
#         return {}

#     results = {
#         'samples_loaded': len(df),
#         'columns_found': [],
#         'validation_data': {}
#     }

#     if 'VDM_Z_AXIS_YAW_RATE' in df.columns:
#         results['columns_found'].append('yaw_rate')
#         yaw_telem = df['VDM_Z_AXIS_YAW_RATE'].dropna()

#         if len(yaw_telem) > 0:
#             results['validation_data']['yaw_rate_min'] = yaw_telem.min()
#             results['validation_data']['yaw_rate_max'] = yaw_telem.max()
#             results['validation_data']['yaw_rate_mean'] = yaw_telem.mean()
#             results['validation_data']['yaw_rate_std'] = yaw_telem.std()

#     if 'VDM_Y_AXIS_ACCELERATION' in df.columns:
#         results['columns_found'].append('lateral_accel')
#         lat_accel = df['VDM_Y_AXIS_ACCELERATION'].dropna()

#         if len(lat_accel) > 0:
#             results['validation_data']['lat_accel_min'] = lat_accel.min()
#             results['validation_data']['lat_accel_max'] = lat_accel.max()
#             results['validation_data']['lat_accel_mean'] = lat_accel.mean()
#             results['validation_data']['lat_accel_max_g'] = lat_accel.max() / 9.81

#     if 'SME_TRQSPD_Speed' in df.columns:
#         results['columns_found'].append('speed')
#         speed = df['SME_TRQSPD_Speed'].dropna()

#         if len(speed) > 0:
#             results['validation_data']['speed_min'] = speed.min()
#             results['validation_data']['speed_max'] = speed.max()
#             results['validation_data']['speed_mean'] = speed.mean()

#     return results


def plot_response(model: DoubleBicycleModel, title: str = "Model Response"):
    if not model.time_history:
        print("No data. Run simulate() first.")
        return

    time = np.array(model.time_history)
    states = np.array(model.state_history)
    steering = [inp[0] for inp in model.input_history]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    axes[0].plot(time, (np.array(steering) * u.Unit("rad")).to_value("deg"), 'b-', linewidth=2)
    axes[0].set_ylabel('Steering Angle (°)', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Steering Input')

    axes[1].plot(time, states[:, 0], 'g-', linewidth=2, label='Lateral Velocity')
    axes[1].set_ylabel('Lateral Velocity (m/s)', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_title('Lateral Velocity')

    axes[2].plot(time, (states[:, 1] * u.Unit("rad/s")).to_value("deg/s"), 'r-', linewidth=2, label='Yaw Rate')
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
    print(f"  Wheelbase: {with_units(params.wheelbase, 'm'):.3f} "
          f"(Lf={with_units(params.Lf, 'm'):.3f}, Lr={with_units(params.Lr, 'm'):.3f})")
    print(f"  Track width: {with_units(params.track_width, 'm'):.3f}")
    print(f"  Mass: {with_units(params.mass, 'kg'):.1f}")
    print(f"  Yaw inertia: {with_units(params.yaw_inertia, 'kg*m**2'):.1f}")
    print(f"  Tire stiffness: {with_units(params.C_alpha, 'N/rad'):.0f}")

    model = DoubleBicycleModel(params, tire_model="linear")

    # Test 1: Step steer
    print("\nTEST 1: Step Steer")
    v_x = unit(10.0, "m/s")
    duration = unit(3.0, "s")
    dt = unit(0.01, "s")
    num_steps = int(round(float(duration / dt)))
    n_delay = int(round(float(unit(0.2, "s") / dt)))

    steering_step = np.concatenate([
        np.zeros(n_delay),
        np.full(num_steps - n_delay, 5.7)
    ])
    steering_step = unit(steering_step[:num_steps], "deg")

    time_sim, states_sim = model.simulate(v_x, steering_step, dt=dt, method="rk4")

    speed = with_units(v_x, "m/s")
    v_y_end = with_units(states_sim[-1, 0], "m/s")
    r_end = with_units(states_sim[-1, 1], "rad/s")
    print(f"\nSpeed: {speed:.1f} ({speed.to('km/hr'):.1f})")
    print(f"Duration: {with_units(duration, 's'):.2f}")
    print(f"Results at end:")
    print(f"  Lateral velocity: {v_y_end:.3f}")
    print(f"  Yaw rate: {r_end.to('deg/s'):.2f}")
    print(f"  G-force: {(speed * r_end / u.rad).to('gee'):.3f}")

    fig1 = plot_response(model, "Test 1: Step Steering")
    fig1.savefig(HERE / "test1_step_steer.png", dpi=150)
    print("Saved to test1_step_steer.png")

    # Test 2: Ramp steer
    print("\nTEST 2: Ramp Steer")
    model.reset()
    v_x = unit(8.0, "m/s")
    duration = unit(5.0, "s")
    num_steps = int(round(float(duration / dt)))
    ramp_time = unit(1.0, "s")
    max_steer = unit(10.0, "deg")
    max_steer_deg = with_units(max_steer, "rad").to_value("deg")
    n_ramp = int(round(float(ramp_time / dt)))

    steering_ramp = np.concatenate([
        np.linspace(0, max_steer_deg, n_ramp),
        np.full(num_steps - n_ramp, max_steer_deg)
    ])
    steering_ramp = unit(steering_ramp[:num_steps], "deg")

    time_sim, states_sim = model.simulate(v_x, steering_ramp, dt=dt, method="rk4")

    
    speed = with_units(v_x, "m/s")
    ramp_rate = with_units(max_steer, "rad") / with_units(ramp_time, "s")
    v_y_end = with_units(states_sim[-1, 0], "m/s")
    r_end = with_units(states_sim[-1, 1], "rad/s")
    print(f"\nSpeed: {speed:.1f} ({speed.to('km/hr'):.1f})")
    print(f"Ramp rate: {ramp_rate.to('deg/s'):.1f}")
    print(f"Results at end:")
    print(f"  Lateral velocity: {v_y_end:.3f}")
    print(f"  Yaw rate: {r_end.to('deg/s'):.2f}")
    print(f"  Steady-state G: {(speed * r_end / u.rad).to('gee'):.3f}")

    fig2 = plot_response(model, "Test 2: Ramp Steering")
    fig2.savefig(HERE / "test2_ramp_steer.png", dpi=150)
    print("Saved to test2_ramp_steer.png")

    # Test 3: Lane change
    print("\nTEST 3: Double Lane Change")
    model.reset()
    v_x = unit(10.0, "m/s")
    duration = unit(4.0, "s")
    num_steps = int(round(float(duration / dt)))

    freq = unit(0.5, "Hz")
    amplitude = unit(10.0, "deg")
    t = np.linspace(0 * duration, duration, num_steps)
    steering_dlc = amplitude * np.sin(2 * np.pi * freq * t)

    time_sim, states_sim = model.simulate(v_x, steering_dlc, dt=dt, method="rk4")

    speed = with_units(v_x, "m/s")
    max_v_y = with_units(np.max(np.abs(states_sim[:, 0])), "m/s")
    max_r = with_units(np.max(np.abs(states_sim[:, 1])), "rad/s")
    print(f"\nSpeed: {speed:.1f} ({speed.to('km/hr'):.1f})")
    print(f"Frequency: {with_units(freq, 'Hz'):.1f} (±{with_units(amplitude, 'rad').to('deg'):.0f} amplitude)")
    print(f"Results:")
    print(f"  Max lateral velocity: {max_v_y:.3f}")
    print(f"  Max yaw rate: {max_r.to('deg/s'):.2f}")
    print(f"  Max G-force: {(speed * max_r / u.rad).to('gee'):.3f}")

    fig3 = plot_response(model, "Test 3: Double Lane Change")
    fig3.savefig(HERE / "test3_double_lanechange.png", dpi=150)
    print("Saved to test3_double_lanechange.png")

    print("\n--- Done! ---")

    # # Telemetry validation
    # print("\nTELEMETRY VALIDATION")

    # try:
    #     import pandas as pd

    #     print("\nLoading telemetry from 08102025Endurance1_FirstHalf.csv...")
    #     telemetry_file = '/Users/brianlee/vscode_projects/formula_slug/workshops/data_for_dwshop/08102025Endurance1_FirstHalf.csv'
    
    #     df = pd.read_csv(telemetry_file, low_memory=False)
    #     print(f"Loaded {len(df)} samples")

    #     columns_needed = [
    #         'VDM_X_AXIS_ACCELERATION',
    #         'VDM_Y_AXIS_ACCELERATION',
    #         'VDM_Z_AXIS_YAW_RATE',
    #         'SME_TRQSPD_Speed',
    #         'TPERIPH_FL_DATA_WHEELSPEED',
    #         'TPERIPH_FR_DATA_WHEELSPEED',
    #         'TPERIPH_BL_DATA_WHEELSPEED',
    #         'TPERIPH_BR_DATA_WHEELSPEED',
    #     ]

    #     available_cols = [col for col in columns_needed if col in df.columns]
    #     print(f"Found {len(available_cols)}/{len(columns_needed)} expected columns")

    #     if 'VDM_Z_AXIS_YAW_RATE' not in df.columns:
    #         print("  x VDM_Z_AXIS_YAW_RATE not found")
    #     else:
    #         print("  + VDM_Z_AXIS_YAW_RATE found")

    #     if 'VDM_Y_AXIS_ACCELERATION' not in df.columns:
    #         print("  x VDM_Y_AXIS_ACCELERATION not found")
    #     else:
    #         print("  + VDM_Y_AXIS_ACCELERATION found")

    #     if 'SME_TRQSPD_Speed' not in df.columns:
    #         print("  x SME_TRQSPD_Speed not found")
    #     else:
    #         print("  + SME_TRQSPD_Speed found")

    #     if 'VDM_Z_AXIS_YAW_RATE' in df.columns:
    #         valid_yaw = df[df['VDM_Z_AXIS_YAW_RATE'].notna()]['VDM_Z_AXIS_YAW_RATE']

    #         if len(valid_yaw) > 0:
    #             print(f"\nYaw rate statistics from telemetry:")
    #             print(f"  Min: {valid_yaw.min():.1f} °/s ({np.deg2rad(valid_yaw.min()):.3f} rad/s)")
    #             print(f"  Max: {valid_yaw.max():.1f} °/s ({np.deg2rad(valid_yaw.max()):.3f} rad/s)")
    #             print(f"  Mean: {valid_yaw.mean():.1f} °/s ({np.deg2rad(valid_yaw.mean()):.3f} rad/s)")
    #             print(f"  Std: {valid_yaw.std():.1f} °/s ({np.deg2rad(valid_yaw.std()):.3f} rad/s)")

    #             high_yaw_mask = np.abs(valid_yaw) > 10.0  # 10 °/s threshold for turning
    #             if high_yaw_mask.any():
    #                 high_idx = np.where(high_yaw_mask)[0][0]
    #                 print(f"\nFound turning maneuver at sample {high_idx}")
    #                 yaw_val = valid_yaw.iloc[high_idx]
    #                 print(f"  Yaw rate: {yaw_val:.1f} °/s ({np.deg2rad(yaw_val):.3f} rad/s)")

    #     print("\nTelemetry validation framework ready")

    # except ImportError:
    #     print("Pandas not installed - can't load CSV")
    #     print("Install with: pip install pandas")
    # except Exception as e:
    #     print(f"Error loading telemetry: {e}")

    plt.show()
