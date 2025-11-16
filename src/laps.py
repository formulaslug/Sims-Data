import polars as pl
import numpy as np

from .fs_signals import lat, lon, LAP_COL, blueMaxGPS_Square, t # import the columns i used in fs signals 

def _inside_box(lat_arr, lon_arr, box): 
    (lon_min, lat_max), (lon_max, lat_min) = box
    return (
        (lat_arr >= lat_min) & (lat_arr <= lat_max) &
        (lon_arr >= lon_min) & (lon_arr <= lon_max)
    )
def add_lap_column(
        df: pl.DataFrame, 
        gps_box=blueMaxGPS_Square,
        lap_col: str = LAP_COL,
        min_lap_time: float = 20.0,
) -> pl.DataFrame:
    
    pdf = df.to_pandas() # convert for easier looping

    lat_arr = pdf[lat].to_numpy()
    lon_arr = pdf[lon].to_numpy()
    time_arr = pdf[t].to_numpy()

    in_box = _inside_box(lat_arr, lon_arr, gps_box)

    laps = np.zeros(len(pdf), dtype=int)
    current_lap = 0
    last_cross_time = None

    for i, (inside, time_s) in enumerate(zip(in_box, time_arr)): 
        if inside: 
            if last_cross_time is None or (time_s - last_cross_time) > min_lap_time:
                current_lap += 1
                last_cross_time = time_s
        laps[i] = current_lap

    pdf[lap_col] = laps 

    # drop everything before the first full lap (lap 1)
    pdf = pdf[pdf[lap_col] > 0].reset_index(drop=True)

    return pl.from_pandas(pdf) # convert back to polars 

def lap_stats(
    df: pl.DataFrame,
    lap_col: str = LAP_COL,
) -> pl.DataFrame:
    """
    Compute simple per-lap stats: start time, end time, and lap time.
    """
    return (
        df.group_by(lap_col)
        .agg([
            pl.col(t).min().alias("t_start"),
            pl.col(t).max().alias("t_end"),
        ])
        .with_columns(
            (pl.col("t_end") - pl.col("t_start")).alias("LapTime")
        )
        .sort(lap_col)
    )
