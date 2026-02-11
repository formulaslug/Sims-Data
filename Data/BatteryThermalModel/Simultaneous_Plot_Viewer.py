# read current draw from parquet file
from fs4BatteryThermalModelingEx import thermal_ode_solve_ivp
import argparse
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Tuple
from scipy.integrate import solve_ivp

def load_avg_temp_from_parquet(path: str) -> NDArray[np.float64]:
    """
    Load average temperature from all ACC_SEG*_TEMPS_CELL* columns in a Parquet file.
    
    Parameters
    ----------
    path : str
        Path to the Parquet file containing temperature data.
    
    Returns
    -------
    NDArray[np.float64]
        Array of average temperatures across all cells.
    """
    columns_list = [
        "ACC_SEG0_TEMPS_CELL0", "ACC_SEG0_TEMPS_CELL1", "ACC_SEG0_TEMPS_CELL2", 
        "ACC_SEG0_TEMPS_CELL3", "ACC_SEG0_TEMPS_CELL4", "ACC_SEG0_TEMPS_CELL5",
        "ACC_SEG1_TEMPS_CELL0", "ACC_SEG1_TEMPS_CELL1", "ACC_SEG1_TEMPS_CELL2", 
        "ACC_SEG1_TEMPS_CELL3", "ACC_SEG1_TEMPS_CELL4", "ACC_SEG1_TEMPS_CELL5",
        "ACC_SEG2_TEMPS_CELL0", "ACC_SEG2_TEMPS_CELL1", "ACC_SEG2_TEMPS_CELL2", 
        "ACC_SEG2_TEMPS_CELL3", "ACC_SEG2_TEMPS_CELL4", "ACC_SEG2_TEMPS_CELL5",
        "ACC_SEG3_TEMPS_CELL0", "ACC_SEG3_TEMPS_CELL1", "ACC_SEG3_TEMPS_CELL2", 
        "ACC_SEG3_TEMPS_CELL3", "ACC_SEG3_TEMPS_CELL4", "ACC_SEG3_TEMPS_CELL5",
        "ACC_SEG4_TEMPS_CELL0", "ACC_SEG4_TEMPS_CELL1", "ACC_SEG4_TEMPS_CELL2", 
        "ACC_SEG4_TEMPS_CELL3", "ACC_SEG4_TEMPS_CELL4", "ACC_SEG4_TEMPS_CELL5"
    ]
    
    temps_df = pl.read_parquet(path, columns=columns_list)
    # take the mean of all the columns
    avg_temp_df = temps_df.select(pl.mean_horizontal(pl.col(columns_list)).alias("avg"))
    return avg_temp_df["avg"].to_numpy()

def run_thermal_model_from_parquet(
    path: str,
    current_column: str = "SME_TEMP_BusCurrent",
    t_span: Tuple[float, float] = (0, 10),
    initial_temp: float = 30,
    t_eval: NDArray[np.float64] | None = None,
):
    """
    Load current data from a Parquet file and run the thermal ODE model.
    
    Parameters
    ----------
    path : str
        Path to the Parquet file containing current data.
    current_column : str, optional
        Name of the current column in the Parquet file (default: "SME_TEMP_BusCurrent").
    t_span : Tuple[float, float], optional
        Integration time span (t0, tf) in seconds (default: (0, 10)).
    initial_temp : float, optional
        Initial temperature in °C (default: 30).
    t_eval : NDArray[np.float64] | None, optional
        Times at which to evaluate the solution. If None, uses 100 points over t_span.
    
    Returns
    -------
    OdeResult
        Solution from thermal_ode_solve_ivp containing temperature predictions.
    """
    df = pl.read_parquet(path)
    current_draw = df[current_column].to_numpy()
    
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 100, dtype=np.float64)
    
    return thermal_ode_solve_ivp(current_draw, t_span, initial_temp, t_eval)

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compare thermal model predictions with measured temperatures from parquet data."
    )
    parser.add_argument(
        "parquet_path",
        type=str,
        help="Path to the Parquet file containing temperature and current data.",
    )
    return parser.parse_args()


def plot_temperature_comparison(
    solution,
    av_temp_array: NDArray[np.float64],
    t_span: Tuple[float, float] = (0, 10),
) -> None:
    """
    Plot comparison between thermal model prediction and measured temperatures.
    
    Parameters
    ----------
    solution : OdeResult
        Solution from thermal_ode_solve_ivp containing model predictions.
    av_temp_array : NDArray[np.float64]
        Array of measured average temperatures.
    t_span : Tuple[float, float], optional
        Time span (t0, tf) in seconds for the actual data (default: (0, 10)).
    """
    # Create time array for av_temp_array (assuming same time span as model)
    t_actual = np.linspace(t_span[0], t_span[1], len(av_temp_array))
    
    # Plot the temperature vs time in the same plot window
    plt.figure()  # Explicitly create a single figure
    plt.plot(solution.t, solution.y[0], label="Model prediction", linewidth=2)
    plt.plot(t_actual, av_temp_array, label="Measured temperature", linewidth=2, alpha=0.7)
    plt.xlabel("Time [s]")
    plt.ylabel("Temperature [°C]")
    plt.title("Thermal Model Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    """
    Main function to load data, run thermal model, and plot comparison.
    """
    args = parse_args()
    parquet_path = args.parquet_path
    
    # Load average temperature data from parquet
    av_temp_array = load_avg_temp_from_parquet(parquet_path)
    
    # Run thermal model
    sol = run_thermal_model_from_parquet(parquet_path)
    
    # Check if solution was successful
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")
    
    # Plot the comparison
    plot_temperature_comparison(sol, av_temp_array, t_span=(0, 10))

# run the main function: python Simultaneous_Plot_Viewer.py "/Users/gautham/Documents/fs-data/FS-3/08102025/08102025Endurance1_FirstHalf.parquet"
if __name__ == "__main__":
    main()
