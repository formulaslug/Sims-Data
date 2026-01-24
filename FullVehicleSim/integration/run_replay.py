import time as pytime

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from state import VehicleState
from engine import stepState
from integration.data_source_csv import CSVControlsSource



def generate_straight_track_cones(length=100.0, half_width=2.0, spacing=5.0):
    """
    Generates a straight "corridor" track from x=0..length with cones on y=±half_width.
    Returns list of dicts: {"x":..., "y":..., "type":"left"/"right"}.
    """
    cones = []
    x = 0.0
    while x <= length:
        cones.append({"x": float(x), "y": float(+half_width), "type": "left"})
        cones.append({"x": float(x), "y": float(-half_width), "type": "right"})
        x += spacing
    return cones


def generate_circular_track_cones(
    center=(0.0, 0.0),
    radius=35.0,
    track_width=6.0,
    cone_spacing=10.0,
):
    """
    Generates cones around a circular track (inner/outer boundaries).
    Returns list of dicts: {"x":..., "y":..., "type":"left"/"right"}.
    """
    cx, cy = center
    r_inner = radius - track_width / 2.0
    r_outer = radius + track_width / 2.0

    circumference = 2.0 * np.pi * radius
    n = max(12, int(circumference / cone_spacing))

    cones = []
    for i in range(n):
        theta = 2.0 * np.pi * i / n

        xi = cx + r_inner * np.cos(theta)
        yi = cy + r_inner * np.sin(theta)
        cones.append({"x": float(xi), "y": float(yi), "type": "left"})

        xo = cx + r_outer * np.cos(theta)
        yo = cy + r_outer * np.sin(theta)
        cones.append({"x": float(xo), "y": float(yo), "type": "right"})

    return cones


def run(csv_path: str, steps_per_second=100, sim_duration=6.0, max_radius=100.0):
    dt = 1.0 / steps_per_second

    curr = VehicleState(
        stepSize=dt,
        position=np.asarray([0, 0, 0], dtype=np.float32),
        speed=0,
        acceleration=0,
        heading=np.asarray([1, 0, 0], dtype=np.float32),
        charge=50,
        lastCurrent=0,
        throttle=0,
        brakes=0,
        yawRate=0,
        steerAngle=0,
        brakeTemperature=150,
        timeSinceLastSteer=0,
        initSpeed=0,
    )

    src = CSVControlsSource(csv_path)

    vehicle_states = [curr]
    time_col = [0.0]

    t = 0.0
    timeSinceLastSteer = 0.0
    initSpeed = 0.0
    prev_steer = None

    start = pytime.time()

    for _ in range(int(sim_duration * steps_per_second)):
        t += dt
        timeSinceLastSteer += dt

        controls = src.get_controls(t)  # [throttle, brake, steer]

        # Steering-change bookkeeping (same idea as your main.py)
        steer = controls[2]
        if prev_steer is None:
            prev_steer = steer
        if steer != prev_steer:
            timeSinceLastSteer = 0.0
            initSpeed = max(curr.speed, 5.0)  # stability clamp
            prev_steer = steer

        curr = stepState(curr, controls, dt, timeSinceLastSteer, initSpeed)
        vehicle_states.append(curr)
        time_col.append(t)

        # Hard cap for visualization bounds
        if np.linalg.norm(curr.position[:2]) >= max_radius:
            print(f"Stopping early: exceeded {max_radius}m radius")
            break

    print("Replay runtime (s):", pytime.time() - start)

    # Build dataframe (same columns you used)
    columns = [
        "posX", "posY", "velX", "velY", "speed", "acceleration",
        "headingX", "headingY", "yawRate", "steerAngle", "throttle",
        "brakes", "drag", "resistiveForces", "motorForce", "netForce",
        "torque", "motorTorque", "maxTraction", "maxTractionTorqueAtWheel",
        "cooledBrakeTemperature", "wheelRPM", "wheelRotationsHZ",
        "rpm", "motorRotationsHZ", "charge", "voltage", "current",
        "power", "maxPower", "stepSize", "timeSinceLastSteer",
    ]

    data_rows = [s.logProperties() for s in vehicle_states]
    df = pl.DataFrame(data_rows, schema=columns, orient="row").with_columns(
        pl.Series("time", time_col, dtype=pl.Float64)
    )
    # car dynamics debugging info 
    print("posY min/max:", float(df["posY"].min()), float(df["posY"].max()))
    print("velY min/max:", float(df["velY"].min()), float(df["velY"].max()))
    print("headingY min/max:", float(df["headingY"].min()), float(df["headingY"].max()))


    # Debug prints (do this BEFORE plotting)
    print("Final pos:", curr.position)
    print("Final heading:", curr.heading)
    print("Yaw rate range:", float(df["yawRate"].min()), float(df["yawRate"].max()))
    print("Steer range:", float(df["steerAngle"].min()), float(df["steerAngle"].max()))

    # -----------------------
    # 3D Track Visualization
    # -----------------------
    x = df["posX"].to_list()
    y = df["posY"].to_list()
    z = [0.0] * len(x)

    # Choose one:
    #cones = generate_straight_track_cones(length=100.0, half_width=2.0, spacing=5.0)
    cones = generate_circular_track_cones(center=(0, 0), radius=35.0, track_width=6.0, cone_spacing=10.0)

    cone_x = [c["x"] for c in cones]
    cone_y = [c["y"] for c in cones]

    left_idx = [i for i, c in enumerate(cones) if c.get("type") == "left"]
    right_idx = [i for i, c in enumerate(cones) if c.get("type") == "right"]

    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection="3d")

    ax3d.plot(x, y, z, linewidth=2)
    if x and y:
        ax3d.scatter([x[0]], [y[0]], [0.0], s=120)     # start
        ax3d.scatter([x[-1]], [y[-1]], [0.0], s=120)   # end

    if left_idx:
        ax3d.scatter(
            [cone_x[i] for i in left_idx],
            [cone_y[i] for i in left_idx],
            [0.0] * len(left_idx),
            marker="^",
            s=60,
        )
    if right_idx:
        ax3d.scatter(
            [cone_x[i] for i in right_idx],
            [cone_y[i] for i in right_idx],
            [0.0] * len(right_idx),
            marker="^",
            s=60,
        )

    ax3d.set_title("3D View: Trajectory + Cones (Z=0 ground)")
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    # 2d plot 
    fig2d = plt.figure(figsize=(8,5))
    plt.plot(x, y, linewidth=2, label="Trajectory")
    plt.scatter(cone_x, cone_y, marker="^", s=60, label="Cones")

    # show corridor bounds for straight track
    plt.axhline(+2.0, linestyle="--", linewidth=1)
    plt.axhline(-2.0, linestyle="--", linewidth=1)

    plt.axis("equal")
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    plt.grid(True)
    plt.legend()
    plt.title("Top-down View: Trajectory + Cones")

    # Bounds tuned for the straight 0..100m corridor.
    ax3d.set_xlim(-50, 50)
    ax3d.set_ylim(-50, 50)
    ax3d.set_zlim(0, 1)

    # -----------------------
    # Car dynamics plots
    # -----------------------
    time_list = df["time"].to_list()
    current = df["current"].to_list()
    speed = df["speed"].to_list()
    voltage = df["voltage"].to_list()
    yaw_rate = df["yawRate"].to_list()
    steer_angle = df["steerAngle"].to_list()
    throttle = df["throttle"].to_list()
    brakes = df["brakes"].to_list()

    fig2 = plt.figure()
    ax1 = plt.subplot(2, 3, 1); ax1.set_title("Current vs Time"); ax1.plot(time_list, current)
    ax2 = plt.subplot(2, 3, 2); ax2.set_title("Speed vs Time"); ax2.plot(time_list, speed)
    ax3 = plt.subplot(2, 3, 3); ax3.set_title("Voltage vs Time"); ax3.plot(time_list, voltage)
    ax4 = plt.subplot(2, 3, 4); ax4.set_title("YawRate vs Time"); ax4.plot(time_list, yaw_rate)
    ax5 = plt.subplot(2, 3, 5); ax5.set_title("SteerAngle vs Time"); ax5.plot(time_list, steer_angle)
    ax6 = plt.subplot(2, 3, 6); ax6.set_title("Throttle/Brakes vs Time"); ax6.plot(time_list, throttle); ax6.plot(time_list, brakes)

    plt.tight_layout()
    print("z unique:", set(z))
    print("position z min/max:", float(df["posY"].min()), float(df["posY"].max()))  # ignore, no posZ in df
    print("curr.position:", curr.position)
    cx, cy = 0.0, 0.0
    r = np.sqrt((np.array(x) - cx)**2 + (np.array(y) - cy)**2)
    print("radius min/max:", r.min(), r.max())
    print("radius mean:", r.mean())
    plt.show()

    return df


if __name__ == "__main__":
    run("integration/constant_turn.csv", steps_per_second=100, sim_duration=10.0, max_radius=100.0)
