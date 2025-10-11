import polars as pl
import cantools.database as db
import can.io as ci
from typing import Dict, Any

def process_reader(reader:ci.CanutilsLogReader, dbc, window_ms: int = 10):
    """
    Read CAN messages from `reader`, decode using `dbc`, group into windows of `window_ms` milliseconds,
    and write a Parquet file at `output_path`.

    - Messages within the same `window_ms` (default 10 ms) are grouped into one row.
    - If multiple values for the same signal arrive in the window, the latest (by timestamp) is used.
    - Missing/unknown signal values are written as null.
    """

    # Map window_start (float seconds) -> dict[col_name -> (timestamp, value)]
    windows: Dict[float, Dict[str, tuple]] = {}
    all_signals = set()

    for msg in reader:
        # try a few common timestamp attribute names
        ts = msg.timestamp
        id = msg.arbitration_id
        if id == 128:
            continue

        # compute window start time in seconds
        bin_index = int((ts * 1000) // window_ms)
        window_start = (bin_index * window_ms) / 1000.0

        if window_start not in windows:
            windows[window_start] = {}

        # Try to decode message using DBC
        try:
            # cantools Database has get_message_by_frame_id
            msg_def = dbc.get_message_by_frame_id(msg.arbitration_id)
        except Exception:
            msg_def = None

        if msg_def is None:
            # Unknown message id: skip
            continue

        try:
            decoded = msg_def.decode(msg.data)
        except Exception:
            # decoding failed; skip this message
            continue

        for sig_name, val in decoded.items():
            # build a unique column name: MessageName.SignalName (fallback to arbitration id if no name)
            col_name = f"{sig_name}" if getattr(msg_def, "name", None) else f"{msg.arbitration_id:X}.{sig_name}"
            all_signals.add(col_name)

            existing = windows[window_start].get(col_name)
            # Keep the latest value in the window (compare timestamps)
            if (existing is None) or (ts >= existing[0]):
                windows[window_start][col_name] = (ts, val)

    # Build rows sorted by window_start
    rows = []
    for window_start in sorted(windows.keys()):
        row: Dict[str, Any] = {"timestamp": window_start}
        win = windows[window_start]
        for sig in all_signals:
            if sig in win:
                row[sig] = win[sig][1]
            else:
                row[sig] = None
        rows.append(row)

    if not rows:
        print("No rows decoded; no Parquet written.")
        return

    df = pl.from_dicts(rows)
    # Write parquet; polars will use sensible defaults for nulls
    return df

dbcpath = "../fs-3/CANbus.dbc"
logfile = "C:/Users/Goob/Downloads/wheels_nathaniel_inv_test_w_fault.log"
logfile2 = "C:/Users/Goob/Downloads/no_wheels_nathaniel_inv_test.log"
out_parquet = "can_messages_10ms.parquet"

dbc = db.load_file(dbcpath)

f = open(logfile2, "r")
reader = ci.CanutilsLogReader(f)
df = process_reader(reader, dbc)
df = df.with_columns( #type: ignore
    pl.col("timestamp") - pl.col("timestamp").min()
)
df.write_parquet("C:/Projects/FormulaSlug/fs-data/FS-3/10082025/fixed_no_wheels_nathaniel_inv_test.parquet")

f = open(logfile, "r")
reader = ci.CanutilsLogReader(f)
df = process_reader(reader, dbc)
df = df.with_columns( #type: ignore
    pl.col("timestamp") - pl.col("timestamp").min()
)
df.write_parquet("C:/Projects/FormulaSlug/fs-data/FS-3/10082025/fixed_wheels_nathaniel_inv_test_w_fault.parquet")
