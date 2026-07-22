"""
Double Bicycle Yaw Rate Model for Formula Slug

2DOF model (lateral velocity + yaw rate) for basic vehicle dynamics.
Based on Rajamani's bicycle model.
"""

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List
import matplotlib.pyplot as plt
import sys
from paramLoader import *
import math

@dataclass
class VehicleParameters:
    """Vehicle parameters from 2025 FSAE Design Spec Sheet (Car #216)"""

    # Geometry (meters) - from 2025 FSAE Design EV Spec Sheet
    wheelbase: float = 1.589989  # 1589.989 mm
    Lf: float = 0.8535  # Calculated from 46.32% front weight distribution
    Lr: float = 0.7365  # Calculated from weight distribution
    track_width_front: float = 1.234008  # 1234.008 mm
    track_width_rear: float = 1.186  # 1186 m
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

class TireModel:

    def __init__(self, params: VehicleParameters, temperature, mechanicalParams, magicParams, axle, slipAngle, pressure=12, camber=0):
        
        self.magic = magicParams
        self.mechanical = mechanicalParams
        self.params = params

        if axle == "front":
            self.normalForce = self.params.mass * 9.81 * self.params.Lr / self.params.wheelbase / 2.0
        elif axle == "rear":
            self.normalForce = self.params.mass * 9.81 * self.params.Lf / self.params.wheelbase / 2.0

        self.slipAngle = slipAngle
        self.tirePressure = pressure
        self.tireTemperature = temperature
        self.actPressure = pressure # Actual PSI
        self.camber = camber # Radians




        #if(lat):
        self.normDeltaLoadLat = self.normalizeLoadLat()
        self.normDeltaPressureLat = self.normalizePressureLat()
        #if(long):
        self.normDeltaLoadLong = self.normalizeLoadLong()
        self.normDeltaPressureLong = self.normalizePressureLong()

        self.normalForce = self.getNormalLoad(self.normalForce)

    def getNormalLoad(self, inputNormalForce):
        # Neglecting last force
        # I intentionally neglect the last Fx and Fy because that would involve a large rewrite of this.
        sqrt_term = math.sqrt(9.81 * self.mechanical["unloaded-radius"])
        term1 = (1 + self.magic["q_v2"] * abs(self.magic["Omega"]) * self.mechanical["unloaded-radius"]/sqrt_term - self.magic["q_Fcx"] - self.magic["q_Fcy"])
        # We assume the deflection is 1 because idk how to do that
        term2 = (self.magic["q_Fz1"] + self.magic["q_Fz2"] * self.camber**2) / self.mechanical["unloaded-radius"]
        term3 = (1 + self.magic["P_pFz1"] * self.normDeltaPressureLong) * inputNormalForce

        return term1 * term2 * term3

    ##### ********************************
    ##### LATERAL SLIP FUNCTION
    ##### ********************************

    def getLateralForce(self, worldArray:NDArray[np.float64], step:int):

        self.velocityX = worldArray[step, varVelX]

        Alphas = self.magic["lambda_alphastar"] * self.slipAngle * math.copysign(1, self.velocityX)
        Byk = self.magic["r_by1"]# + self.magic["r_by4"] * math.sin(self.camber) ** 2) * math.cos(math.atan(self.magic["r_by2"] * (Alphas - self.magic["r_by3"]))) * self.magic["lambda_yk"]
        Cyk = self.magic["r_cy1"]
        Eyk = self.magic["r_ey1"] + self.magic["r_ey2"] * self.normDeltaLoadLat
        Shyk = self.magic["r_hy1"] + self.magic["r_hy2"] * self.normDeltaLoadLat
        
        # Use Slip Ratio = (Wheel RPM - GPS Speed) / GPS Speed
        rpm = worldArray[step, varWheelRPM]
        angular_speed = (2 * math.pi * Parameters['wheelRadius'] * rpm) / 60
        longitudinal_speed = worldArray[step, varSpeed]

        if longitudinal_speed == 0: self.slipRatio = 0
        else: self.slipRatio = (angular_speed - longitudinal_speed) / longitudinal_speed

        Ks = self.slipRatio + Shyk
        BykKs = Byk * Ks
        BykShyk = Byk * Shyk
        Gykappa = math.cos(Cyk * math.atan(BykKs - Eyk * (BykKs - math.atan(BykKs))))
        Gykappazero =  math.cos(Cyk * math.atan(BykShyk - Eyk * (BykShyk - math.atan(BykShyk))))


        Dvyk = self.mechanical["friction-coeff-lat"] * self.normalForce * (self.magic["r_vy1"] + self.magic["r_vy2"] * self.normDeltaLoadLat + self.magic["r_vy3"] * math.sin(self.camber)) * math.cos(math.atan(self.magic["r_vy4"] * math.sin(Alphas)))  * self.magic["zeta_2"]
        Svyk = Dvyk * math.sin(self.magic["r_vy5"] * math.atan(self.magic["r_vy6"] * self.slipRatio)) * self.magic["lambda_vyk"]

        #print(Byk, Cyk, Eyk, Shyk)

        return Gykappa/Gykappazero * self.getLateralForcePure() #+ Svyk # + self.magic["Svyk"]

    def getLateralForcePure(self):
        Alphas = self.magic["lambda_alphastarypure"] * self.slipAngle * math.copysign(1,self.velocityX)

        loadDependentPeak = self.magic["loadA"] * self.normalForce * self.normalForce + self.magic["loadB"] * self.normalForce + self.magic["loadC"]

        self.Cy = self.magic["p_cy1"]
        self.Dy = loadDependentPeak * self.getLateralCoefficientOfFriction() * self.normalForce * (self.magic["tempYAPure"] * self.tireTemperature ** 2 + self.magic["tempYBPure"] * self.tireTemperature + self.magic["tempYCPure"])
        self.By = self.magic["By_pure"]
        self.Ey = self.getLateralE(Alphas)

        Svy = self.magic["Svy"]
        return self.stdCurveSine(self.By, self.Cy, self.Dy, self.Ey, self.slipRatio) + Svy
    
    def getLateralCoefficientOfFriction(self):
        return (self.magic["p_dy1"] + self.magic["p_dy2"] * self.normDeltaLoadLat) * (1 + self.magic["p_py3"] * self.normDeltaPressureLat + self.magic["p_py4"] * self.normDeltaPressureLat ** 2) * (1 - self.magic["p_dy3"] * math.sin(self.camber) ** 2) * self.magic["lambda_coeffscalary"]
    
    def getLateralE(self, Alphas):
        term1 = (self.magic["p_ey1"] + self.magic["p_ey2"] * self.normDeltaLoadLat)
        term2 = (1 + self.magic["p_ey5"] * math.sin(self.camber) ** 2 - (self.magic["p_ey3"] + self.magic["p_ey4"] * math.sin(self.camber)) * Alphas)
        return term1 * term2 * self.magic["lambda_ey"]


    ##### ********************************
    ##### Standard Functioms
    ##### ********************************

    def stdCurveSine(self, Bx, Cx, Dx, Ex, slip):
        BxSlip = Bx * slip
        return Dx * math.sin( Cx * math.atan( BxSlip - Ex * (BxSlip - math.atan(BxSlip) ) ) )

    def normalizeLoadLong(self):
        return (self.normalForce - self.magic["lambda_loadscalarlong"] * self.normalForce) / (self.magic["lambda_loadscalarlong"] * self.normalForce)

    def normalizeLoadLat(self):
        return (self.normalForce - self.magic["lambda_loadscalarlat"] * self.normalForce) / (self.magic["lambda_loadscalarlat"] * self.normalForce)

    def normalizePressureLong(self):
        # Only long because lat doesn't use it
        return (self.tirePressure - self.magic["lambda_pressurescalarlong"] * self.tirePressure) / (self.magic["lambda_pressurescalarlong"] * self.tirePressure)

    def normalizePressureLat(self):
        # Only long because lat doesn't use it
        return (self.tirePressure - self.magic["lambda_pressurescalarlat"] * self.tirePressure) / (self.magic["lambda_pressurescalarlat"] * self.tirePressure)

    def updateParams(self, normalForce=-1, slipRatio=-1, velocityX=-1):
        if normalForce != -1:
            self.normalForce = normalForce
        if slipRatio != -1:
            self.slipRatio = slipRatio
        if velocityX != -1:
            self.velocityX = velocityX

class DoubleBicycleModel:
    """2DOF bicycle model: v_y (lateral velocity) and r (yaw rate)"""

    def __init__(self, params: VehicleParameters):
        
        self.params = params

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
    
    def rackMovement(self, wheelInput: float): #returns the amount of L-R displacement (in mm) of the steering rack, with the right direction as "positive"

        rackShift: float = Parameters['rackRatio']*wheelInput # wheelInput
        return rackShift

    def calculateAckermann(self, wheelInput: float): #calculates the steer angles of both wheels

        l1Left = (0.5*(Parameters["tw"]-Parameters["l_rack"])) - self.rackMovement(wheelInput) #l1 is the instantaneous parallel distance from the rack knuckle to steering axis (KPA). 
        l1Right = (0.5*(Parameters["tw"]-Parameters["l_rack"])) + self.rackMovement(wheelInput)
        l_nought = (0.5*(Parameters["tw"]-Parameters["l_rack"]))
        beta_nought = self.betaTrigSolver(l_nought) #used to find the initial "beta" geometry to determine the real steer angle at the wheels

        beta_L = self.betaTrigSolver(l1Left) - beta_nought #additionally, because there is a static "beta" (simply just arm geometry), we must find the difference to find the actual wheel angles
        beta_R = self.betaTrigSolver(l1Right) - beta_nought
        
        return beta_L, beta_R
        #return beta_nought, betaTrigSolver(l1Left), betaTrigSolver(l1Right)
        
    def betaTrigSolver(self, l1): #a separate function to solve the big bad trig equation
        l2 = np.sqrt((l1**2) + (Parameters["d"]**2)) #l2 is the instantaneous direct distance from rack knuckle to steering axis (KPA)
        atan = np.arctan(Parameters["d"]/l1) #first term of the "beta" equation

        num = (Parameters["l_arm"]**2) + (l2**2) - (Parameters["l_rod"]**2) #just simplifying the calculation of the second term 
        denom = 2*Parameters["l_arm"]*l2
        frac = num/denom
        acos = np.arccos(frac)
        beta = (np.pi/2) - atan - acos
        return beta
        #return frac

    def dynamics(self, state: NDArray[np.float64], v_x: float, delta: float,
                 worldArray: NDArray[np.float64], step:int, ax: float = 0.0) -> NDArray[np.float64]:
        """Compute state derivatives [dv_y/dt, dr/dt]"""
        v_y, r = state

        alpha_f, alpha_r = self.get_slip_angles(v_y, r, v_x, delta) # old way of getting slip angles

        delta_fl, delta_fr = self.calculateAckermann(wheelInput=delta)

        self.tire_fl = TireModel(
            temperature=Parameters['ambientTemperature'], 
            mechanicalParams=Parameters, 
            magicParams=Magic, 
            slipAngle=delta_fl,
            axle="front",
            params=params
        )

        self.tire_fr = TireModel( 
            temperature=Parameters['ambientTemperature'], 
            mechanicalParams=Parameters, 
            magicParams=Magic, 
            slipAngle=delta_fr,
            axle="front",
            params=params
        )

        self.tire_rear = TireModel( 
            temperature=Parameters['ambientTemperature'], 
            mechanicalParams=Parameters, 
            magicParams=Magic, 
            slipAngle=alpha_r,
            axle="rear",
            params=params
        )

        Fy_fl = -self.tire_fl.getLateralForce(worldArray, step)
        Fy_fr = -self.tire_fr.getLateralForce(worldArray, step)
        Fy_r = -self.tire_rear.getLateralForce(worldArray, step)

        # Account for both tires per axle
        Fy_f_total = Fy_fl + Fy_fr
        Fy_r_total = 2.0 * Fy_r

        # Lateral and yaw accelerations
        a_y = (Fy_f_total * np.cos(delta) + Fy_r_total) / self.params.mass + v_x * r
        dv_y = a_y

        M_yaw = self.params.Lf * Fy_f_total - self.params.Lr * Fy_r_total
        dr = M_yaw / self.params.yaw_inertia

        return np.array([dv_y, dr])
    
    def integrate_step(self, v_x: float, delta: float, dt: float, worldArray: NDArray[np.float64], step:int,
                      ax: float = 0.0, method: str = "rk4") -> None:
        """Integrate one timestep using euler, rk2, or rk4"""
        if method == "euler":
            k1 = self.dynamics(state=self.state, v_x=v_x, delta=delta, ax=ax, worldArray=worldArray, step=step)
            self.state = self.state + dt * k1

        elif method == "rk2":
            k1 = self.dynamics(state=self.state, v_x=v_x, delta=delta, ax=ax, worldArray=worldArray, step=step)
            k2 = self.dynamics(state=self.state + 0.5*dt*k1, v_x=v_x, delta=delta, ax=ax, worldArray=worldArray, step=step)
            self.state = self.state + dt * k2

        elif method == "rk4":
            k1 = self.dynamics(state=self.state, v_x=v_x, delta=delta, ax=ax, worldArray=worldArray, step=step)
            k2 = self.dynamics(state=self.state + 0.5*dt*k1, v_x=v_x, delta=delta, ax=ax, worldArray=worldArray, step=step)
            k3 = self.dynamics(state=self.state + 0.5*dt*k2, v_x=v_x, delta=delta, ax=ax, worldArray=worldArray, step=step)
            k4 = self.dynamics(state=self.state + dt*k3, v_x=v_x, delta=delta, ax=ax,worldArray=worldArray, step=step)
            self.state = self.state + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

        else:
            raise ValueError(f"Unknown integration method: {method}")
    
    """
    
    v_x = longitudinal velocty (forward speed of vehicle (m/s))
    steering_inputs = time series of steering angles 
    
    """
    
    def simulate(self, v_x: float, steering_inputs: List[float], worldArray: NDArray[np.float64], step: int,
                dt: float = 0.01, method: str = "rk4") -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Run simulation with given steering input sequence"""
        self.state = np.array([0.0, 0.0])
        self.time_history = [0.0]
        self.state_history = [self.state.copy()]
        self.input_history = [(0.0, v_x)]

        for i, delta in enumerate(steering_inputs):
            t = (i + 1) * dt
            self.integrate_step(v_x, delta, dt, ax=0.0, method=method, worldArray=worldArray, step=step)

            self.time_history.append(t)
            self.state_history.append(self.state.copy())
            self.input_history.append((delta, v_x))

        return np.array(self.time_history, dtype=np.float64), np.array(self.state_history, dtype=np.float64)

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

# globally create these so calcYawRate() can use it.
params = VehicleParameters()
model = DoubleBicycleModel(params=params)

def calcYawRate(worldArray:NDArray[np.float64], step: int) -> tuple[np.float64, np.float64]:
    
    """
    Calculate the Yaw Rate

    :param worldArray: World State Array
    :param step: Current step index
    :return: (Lateral Velocity, Yaw Rate)
    """

    dt = 1 / Parameters["stepsPerSecond"]

    # load model's previous state so we can use it in integrate_step later
    model.state = np.array([
        worldArray[step-1, varLateralVelocty], # v_y
        worldArray[step-1, varYawRate] # r
    ])

    # find steering angle for integrating the step
    delta = worldArray[step, varSteerAngle]

    # create a new state with new v_x and dt
    v_x = worldArray[step, varSpeed]
    model.integrate_step(v_x=v_x, delta=delta, dt=dt, worldArray=worldArray, step=step)

    # extract yaw rate and lateral velocity
    lateral_velocity = model.state[0]
    yaw_rate = model.state[1]

    return lateral_velocity, yaw_rate

if __name__ == "__main__":

    print("\n--- FORMULA SLUG - DOUBLE BICYCLE YAW RATE MODEL ---\n")

    params = VehicleParameters()
    print("Vehicle Parameters:")
    print(f"  Wheelbase: {params.wheelbase:.3f} m (Lf={params.Lf:.3f}, Lr={params.Lr:.3f})")
    print(f"  Track width: {params.track_width:.3f} m")
    print(f"  Mass: {params.mass:.1f} kg")
    print(f"  Yaw inertia: {params.yaw_inertia:.1f} kg·m²")
    print(f"  Tire stiffness: {params.C_alpha:.0f} N/rad")

    model = DoubleBicycleModel(params=params)

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

    time_sim, states_sim = model.simulate(v_x, steering_step, dt=dt, method="rk4", )

    print(f"\nSpeed: {v_x:.1f} m/s ({v_x*3.6:.1f} km/h)")
    print(f"Duration: {duration:.2f} s")
    print(f"Results at end:")
    print(f"  Lateral velocity: {states_sim[-1, 0]:.3f} m/s")
    print(f"  Yaw rate: {np.rad2deg(states_sim[-1, 1]):.2f} °/s")
    print(f"  G-force: {v_x * states_sim[-1, 1] / 9.81:.3f}g")

    fig1 = plot_response(model, "Test 1: Step Steering")
    # fig1.savefig('/Users/brianlee/vscode_projects/formula_slug/fs_yawratemodel/test1_step_steer.png', dpi=150)
    # print("Saved to test1_step_steer.png")

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
    # fig2.savefig('/Users/brianlee/vscode_projects/formula_slug/fs_yawratemodel/test2_ramp_steer.png', dpi=150)
    # print("Saved to test2_ramp_steer.png")

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
    # fig3.savefig('/Users/brianlee/vscode_projects/formula_slug/fs_yawratemodel/test3_double_lanechange.png', dpi=150)
    # print("Saved to test3_double_lanechange.png")

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
