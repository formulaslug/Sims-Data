import polars as pl
import numpy as np

from .fs_signals import t

def _infer_time_column(df: pl.DataFrame) -> pl.Series:
    """
    Infer the time column name in the given DataFrame.

    Args:
        df (pl.DataFrame): The input DataFrame.
    
    """
    if t in df.columns:
        return df[t]

    if "VDM_UTC_TIME_SECONDS" in df.columns:
        utc = df ["VDM_UTC_TIME_SECONDS"]
        # using last year as reference for day of year
        time_s = (utc - utc.min()) * 60.0
        return time_s.alias(t)
    
    n = df.height
    dt - 60.0 / 5035.0 # using last year's comments
    arr = np.arange(n) * dt 
    return pl.Series(name=t, values=arr)

def load_session(path: str) -> pl.DataFrame: 
    """
    Load a Parquet file and ensure it has a time column.

    """
    df = pl.read_parquet(path)

    time_col = _infer_time_column(df)
    
    if t in df.columns:
        df = df.drop(t)
    
    df = df.insert_column(0, time_col)
    return df

def trim_time(df: pl.DataFrame, t_min: float | None = None,
              t_max: float | None = None) -> pl.DataFrame:
    """
    Return a view of df between t_min and t_max (in seconds).
    """
    out = df
    if t_min is not None:
        out = out.filter(pl.col(t) > t_min)
    if t_max is not None:
        out = out.filter(pl.col(t) < t_max)
    return out
