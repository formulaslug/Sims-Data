import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from .fs_signals import lat, lon, speed, LAP_COL


# ---------------------------------------------------------
# GPS LAP PLOTTING
# ---------------------------------------------------------

def plot_gps_laps(
    df: pl.DataFrame,
    lap_col: str = LAP_COL,
    color_by_speed: bool = False,
) -> None:

    if df.height == 0:
        print("[plot_gps_laps] df is empty, nothing to plot.")
        return

    max_val = df[lap_col].max()
    if max_val is None or max_val == 0:
        print("[plot_gps_laps] No valid laps, skipping plot.")
        return

    max_lap = int(max_val)

    plt.figure(figsize=(8, 6))

    for lap_num in range(1, max_lap + 1):
        d = df.filter(pl.col(lap_col) == lap_num)

        if color_by_speed:
            plt.scatter(
                d[lon],
                d[lat],
                c=d[speed],
                cmap="viridis",
                s=4,
                label=f"Lap {lap_num}"
            )
        else:
            plt.plot(d[lon], d[lat], label=f"Lap {lap_num}")

    plt.axis("scaled")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Blue Max GPS Laps")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# PLACEHOLDER CONSISTENCY FUNCTIONS (to prevent import errors)
# ---------------------------------------------------------

def plot_consistency_std(df, lap_col=LAP_COL):
    """
    Placeholder so imports don't crash.
    """
    print("[plot_consistency_std] Placeholder — not implemented yet.")
    return


def line_and_speed_std(df, lap_col=LAP_COL):
    """
    Placeholder — returns dummy arrays so main code can continue.
    """
    print("[line_and_speed_std] Placeholder — not implemented yet.")

    s = np.linspace(0, 1, 10)
    ystd = np.zeros_like(s)
    vstd = np.zeros_like(s)
    return s, ystd, vstd


def segment_consistency_report(s_center, y_std, v_std):
    """
    Placeholder summary for each segment.
    """
    print("[segment_consistency_report] Placeholder — not implemented yet.")

    report = []
    for i in range(len(s_center)):
        report.append((
            f"Seg{i+1}",
            float(y_std[i]),
            float(v_std[i])
        ))
    return report


def print_driver_feedback(report):
    """
    Placeholder driver feedback — doesn't do real analysis.
    """
    print("[print_driver_feedback] Placeholder — not implemented yet.")
    for seg, y, v in report:
        print(f"{seg}: Stay consistent here.")
