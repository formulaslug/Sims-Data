'''
Inventories every column that appears in each car's parquet logs in fs-data, for issue #80.
The mapping files are built from what was actually logged, not only from the DBCs, because
logs were decoded with different DBCs over time and some carry logger or hand-added columns.
Author: Lauwrence Tuy

Usage:
    python collect_columns.py --fs-data ../../../../fs-data

Writes, per car, into ./inventory/:
    <car>_columns.csv  one row per column: how many files have it, which DBCs define it,
                       dtypes seen, and the first and last folder it appears in
    <car>_eras.csv     files grouped by identical column set, so era changes are visible
Only parquet schemas are read, never the data.
'''

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq


##### SETTINGS


## Folder this script lives in, and the formula-slug folder four levels up
    # column_mapping -> FSLib -> Data -> Sims-Data -> formula-slug
    # The DBC repos and fs-data are expected to sit side by side inside formula-slug
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3] # formula-slug/

CARS = {"fs2": "FS-2", "fs3": "FS-3", "fs4": "FS-4"} # Short car name used in output file names -> that car's folder inside fs-data

## Which DBC files define each car's signals, relative to formula-slug
DBCS = {
    "fs2": ["fs-2/mainCANbus.dbc", "fs-2/aemCAN.dbc", "fs-2/ChargeStationCAN.dbc"],
    "fs3": ["fs-3/CANbus.dbc"],
    "fs4": ["fs-4-firmware/CANbus.dbc"],
}

SG_RE = re.compile(r"^\s+SG_\s+(\w+)", re.M) # Captures just a signal's name, e.g. " SG_ GLV_Voltage : ..." -> GLV_Voltage


##### STEP 1: SIGNAL NAMES FROM EACH CAR'S DBCs


def dbc_signals() -> dict[str, set[str]]:
    '''
    Collects the signal names each car's DBCs define. Used later to mark which DBC, if any,
    defines each logged column.

    Returns
    ----------
    {"fs2": {signal names}, "fs3": {...}, "fs4": {...}}. A missing DBC file is skipped,
    leaving that car's set smaller.
    '''
    out = {}
    for car, files in DBCS.items():
        names = set()
        for f in files:
            p = ROOT / f
            if p.exists():
                names |= set(SG_RE.findall(p.read_text(errors="replace")))
        out[car] = names
    return out


##### STEP 2: READ EACH PARQUET'S COLUMN LIST


def scan(car_dir: Path):
    '''
    Reads the column list of every parquet in one car's folder. pq.read_schema reads only the
    file's footer, so this is fast even for large logs.

    Parameters
    ----------
    car_dir
        The car's folder inside fs-data, e.g. fs-data/FS-3

    Returns
    ----------
    per_file
        {file path: {column name: type}}
    errors
        [(file path, error)] for files that couldn't be read. These are reported instead of
        stopping the whole run.
    '''
    files = sorted(p for p in car_dir.rglob("*") if p.suffix in {".pq", ".parquet"} and p.is_file()) # Every .pq / .parquet under the folder, including subfolders, in path order
    per_file, errors = {}, []
    for p in files:
        try:
            schema = pq.read_schema(p)
            per_file[p] = {f.name: str(f.type) for f in schema}
        except Exception as e:  # noqa: BLE001 - report and keep going
            errors.append((p, repr(e)[:120]))
    return per_file, errors


##### STEP 3: WRITE THE TWO CSVs FOR ONE CAR


def write_inventory(car: str, per_file: dict, base: Path, sigs: dict, out_dir: Path) -> None:
    '''
    Writes <car>_columns.csv and <car>_eras.csv.

    Parameters
    ----------
    car
        Short car name ("fs2", "fs3" or "fs4"), used in the file names
    per_file
        The column lists from scan
    base
        The car's folder inside fs-data. Paths in the CSVs are relative to it
    sigs
        The DBC signal names from dbc_signals
    out_dir
        Folder to write the CSVs into
    '''
    rel = lambda p: str(p.relative_to(base)) # A file's path relative to the car's folder
    folder = lambda p: str(p.parent.relative_to(base)) # A file's folder relative to the car's folder

    ## Tally each column across all of this car's files:
        # How many files have it
        # Every type it was stored as
        # The folders it appears in (in scan order, which is path order)
    cols = defaultdict(lambda: {"n": 0, "dtypes": set(), "folders": []})
    for p, schema in per_file.items():
        for name, dtype in schema.items():
            c = cols[name]
            c["n"] += 1
            c["dtypes"].add(dtype)
            if folder(p) not in c["folders"]:
                c["folders"].append(folder(p))

    ## <car>_columns.csv: one row per column
        # Rows are sorted by how many files have the column (most first), then by name
        # first_folder / last_folder follow path order, not date order
    with open(out_dir / f"{car}_columns.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["column", "n_files", "of_files", "in_fs2_dbc", "in_fs3_dbc", "in_fs4_dbc",
                    "dtypes", "n_folders", "first_folder", "last_folder"])
        for name in sorted(cols, key=lambda n: (-cols[n]["n"], n)):
            c = cols[name]
            w.writerow([name, c["n"], len(per_file), *(int(name in sigs[k]) for k in ("fs2", "fs3", "fs4")),
                        "|".join(sorted(c["dtypes"])), len(c["folders"]), c["folders"][0], c["folders"][-1]])

    ## Group files that have exactly the same set of columns. Each group is an "era"
        # Groups are numbered in order of their first file's path, which is not necessarily date order
    eras = defaultdict(list)
    for p, schema in per_file.items():
        eras[frozenset(schema)].append(p)
    ordered = sorted(eras.items(), key=lambda kv: min(rel(p) for p in kv[1]))
    all_cols = set().union(*eras) if eras else set() # Every column seen in any of this car's files, used to list what each era is missing

    ## <car>_eras.csv: one row per group
        # missing_vs_union lists the columns the group lacks, or just a count if there are over 40
    with open(out_dir / f"{car}_eras.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["era", "n_files", "n_columns", "folders", "missing_vs_union", "example_file"])
        for i, (colset, paths) in enumerate(ordered, 1):
            folders = sorted({folder(p) for p in paths})
            missing = sorted(all_cols - colset)
            w.writerow([i, len(paths), len(colset), "|".join(folders),
                        "|".join(missing) if len(missing) <= 40 else f"{len(missing)} columns",
                        rel(sorted(paths)[0])])
    print(f"{car}: {len(per_file)} files, {len(cols)} distinct columns, {len(eras)} distinct column sets")


##### COMMAND LINE


def main() -> None:
    ## Options: where fs-data is (default formula-slug/fs-data) and where to write the CSVs
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fs-data", type=Path, default=ROOT / "fs-data")
    ap.add_argument("--out", type=Path, default=HERE / "inventory")
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True)

    ## Read the DBC signal names once, then scan and write each car in turn
    sigs = dbc_signals()
    for car, dirname in CARS.items():
        base = args.fs_data / dirname
        per_file, errors = scan(base)
        write_inventory(car, per_file, base, sigs, args.out)
        for p, e in errors:
            print(f"  unreadable: {p.relative_to(args.fs_data)}: {e}")


if __name__ == "__main__":
    main()
