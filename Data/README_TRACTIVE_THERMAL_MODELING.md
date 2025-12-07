# Run Thermal Modeling
Todo

### Run Tests

```bash
python -m unittest discover -s test
```

#### Plot the curve
```bash
python ./TractiveBatteryThermalModelViewer.py -h
usage: TractiveBatteryThermalModelViewer.py [-h] --path_parquet PATH_PARQUET [--column-name COLUMN_NAME] [--t-end T_END] [--initial-temp INITIAL_TEMP]

View tractive battery thermal model as a function of current draw.

options:
  -h, --help            show this help message and exit
  --path_parquet PATH_PARQUET
                        Path to the Parquet file containing current data.
  --column-name COLUMN_NAME
                        Name of the current column in the Parquet file (default: SME_TEMP_BusCurrent).
  --t-end T_END         End time in seconds for the simulation (default: 60).
  --initial-temp INITIAL_TEMP
                        Initial temperature in °C (default: 22).
```

Example Command :

```
python TractiveBatteryThermalModelViewer.py --path_parquet <file_name>
```
