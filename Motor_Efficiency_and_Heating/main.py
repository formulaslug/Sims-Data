import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.interpolate import LinearNDInterpolator
from pathlib import Path
import pickle

def load_wpd(path: Path) -> tuple[list[int], pd.DataFrame]:
    
    """
    Load and parse WebPlotDigitizer file

    Returns (numpy list of coordinates, list of bounds that match the numpy list)
    """

    bounds = np.array([])
    coordinates = np.empty((0, 2))
    df = pd.read_csv(path)
    X, Y = 0, 1

    while Y < len(df.columns):

        current_bound = float(df.columns[X])
        current_coordinates_series = df[1:][[df.columns[X], df.columns[Y]]].dropna().to_numpy().astype(float)

        coordinates = np.concatenate([coordinates, current_coordinates_series])
        
        for _ in range(len(current_coordinates_series)):

            bounds = np.append(bounds, current_bound)

        X += 2
        Y += 2

    return coordinates, bounds

def create_function(coordinates: np.ndarray, bounds: list[int], BELOW_70: float = 60.0) -> tuple[LinearNDInterpolator, float, float]:
    """
    Creates a function from the the return variables from load_wpd() using LinearNDInterpolator

    For the bottom and left edge where it transitions from 70 to 90 we add an 80 bound above/right of the 70 bound, because the
    original graph shows it as all green so the interpolator knows that the space between the 70 and 90 bound is actually between 80-90%
    """

    x_max = np.max(coordinates[:, 0])
    y_max = np.max(coordinates[:, 1])

    # find scale since x and y are two different datasets
    scale = np.array([x_max, y_max])

    p70 = coordinates[bounds == 70.0]
    p80 = coordinates[bounds == 80.0]

    # finding the nearest 80% point from each 70% point
    distances = []
    for p in p70:

        current_distances_from_p80 = (p80 - p) / scale
        x = current_distances_from_p80[:, 0]
        y = current_distances_from_p80[:, 1]
        nearest = np.min(np.sqrt(x**2 + y**2))
        distances.append(nearest)

    distances = np.array(distances)

    # 70 points with no 80 point nearby
    distance_threshold = 0.03
    no_80_nearby = p70[distances > distance_threshold]

    # shift each copy just inside the green: up on the bottom edge, right on the left edge
    on_bottom = (no_80_nearby[:, 1] < 20) & (no_80_nearby[:, 0] > 260)
    shift = np.where(on_bottom[:, None], [0.0, 0.5], [5.0, 0.0])
    new80 = no_80_nearby + shift

    coordinates = np.vstack([coordinates, new80])
    bounds = np.append(bounds, np.full(len(new80), 80.0))

    # adding 60 bound along x and y axis
    left   = np.column_stack([np.zeros(50), np.linspace(0, y_max, 50)])
    bottom = np.column_stack([np.linspace(0, x_max, 150), np.zeros(150)])
    edge = np.vstack([left, bottom])

    coordinates = np.vstack([coordinates, edge])
    bounds = np.append(bounds, np.full(len(edge), BELOW_70))

    loop = coordinates[bounds == 95]
    centre = loop.mean(axis=0)
    peak = 95.52 # on chart

    # adding smaller bounds for the peak
    for f in (0.67, 0.33):
        coordinates = np.vstack([coordinates, centre + f * (loop - centre)])
        bounds = np.append(bounds, np.full(len(loop), peak - f * (peak - 95)))
    coordinates = np.vstack([coordinates, centre])
    bounds = np.append(bounds, peak)

    efficiency = LinearNDInterpolator(coordinates, bounds, rescale=True)

    return efficiency, x_max, y_max

def plot_wpd(
    efficiency: LinearNDInterpolator,
    x_max: float = 7500,
    y_max: float = 200,
    BOUND_TO_COLOR: dict = {
        60: "#000080",
        70: "#1c52f7",
        80: "#b9ff3c",
        90: "#f97408",
        91: "#fd4e02",
        92: "#fb2a00",
        93: "#eb0509",
        94: "#c00002",
        95: "#920100"
    }
) -> None:

    """Plot WebPlotDigitizer csvs so they look like the given image"""

    S, T = np.meshgrid(np.linspace(0, x_max, 376), np.linspace(0, y_max, 201))
    E = efficiency(S, T)

    bounds = sorted(BOUND_TO_COLOR) + [96]
    colors = [BOUND_TO_COLOR[bound] for bound in sorted(BOUND_TO_COLOR)]

    fig, ax = plt.subplots(figsize=(10, 5))
    cf = ax.contourf(S, T, E, levels=bounds, colors=colors)
    fig.colorbar(cf, ax=ax, ticks=bounds, label="Efficiency (%)")
    ax.set(xlim=(0, x_max), ylim=(0, y_max),
           xlabel="Rotation speed (rpm)", ylabel="Torque (Nm)")
    plt.show()

    return efficiency
    
if __name__ == "__main__":
    
    path = 'wpd_dataset.csv'

    coordinates, bounds = load_wpd(path=path)

    efficiency, x_max, y_max = create_function(coordinates=coordinates, bounds=bounds)
    with open("interpolator.pickle", "wb") as f:
        pickle.dump(efficiency, f)

    plot_wpd(efficiency=efficiency, x_max=x_max, y_max=y_max)
