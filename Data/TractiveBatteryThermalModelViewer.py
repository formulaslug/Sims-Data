# TractiveBatteryThermalModelViewer.py

import argparse
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import polars as pl

from fs4BatteryThermalModelingEx import thermal_ode_solve_ivp

def load_current_from_parquet(path: str, column: str) -> NDArray[np.float64]:
    """
    Load a single current column from a Parquet file as a NumPy array.
    """
    df = pl.read_parquet(path, columns=[column])
    return df[column].to_numpy()


def run_thermal_model(
    current_draw: NDArray[np.float64],
    t_end: float,
    initial_temp: float,
):
    """
    Run the thermal ODE solver over [0, t_end] for the given current profile.
    """
    t_span = (0.0, t_end)
    t_eval = np.linspace(t_span[0], t_span[1], len(current_draw), dtype=np.float64)
    return thermal_ode_solve_ivp(
        current_draw=current_draw,
        t_span=t_span,
        initial_temp=initial_temp,
        t_eval=t_eval,
    )


def plot_temperature(solution) -> None:
    """
    Plot temperature vs time from a solve_ivp solution.
    """
    t = solution.t
    T = solution.y[0]

    plt.figure()
    plt.plot(t, T, label="Cell temperature")
    plt.xlabel("Time [s]")
    plt.ylabel("Temperature [°C]")
    plt.title("Tractive Battery Thermal Model")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View tractive battery thermal model as a function of current draw."
    )
    parser.add_argument(
        "--path_parquet",
        required=True,
        help="Path to the Parquet file containing current data.",
    )
    parser.add_argument(
        "--column-name",
        default="SME_TEMP_BusCurrent",
        help="Name of the current column in the Parquet file "
             "(default: SME_TEMP_BusCurrent).",
    )
    parser.add_argument(
        "--t-end",
        type=float,
        default=60.0,
        help="End time in seconds for the simulation (default: 60).",
    )
    parser.add_argument(
        "--initial-temp",
        type=float,
        default=22.0,
        help="Initial temperature in °C (default: 22).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    current = load_current_from_parquet(args.path_parquet, args.column_name)

    solution = run_thermal_model(current, args.t_end, args.initial_temp)

    if not solution.success:
        raise RuntimeError(f"ODE solver failed: {solution.message}")

    plot_temperature(solution)


if __name__ == "__main__":
    main()
