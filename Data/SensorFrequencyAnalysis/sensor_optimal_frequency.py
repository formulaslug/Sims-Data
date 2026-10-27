import polars as pl
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

DATA_DIRS = [
    '/Users/aanyajain/Documents/GitHub/fs-data/FS-3/01112026',
    '/Users/aanyajain/Documents/GitHub/fs-data/FS-3/01172026',
    '/Users/aanyajain/Documents/GitHub/fs-data/FS-3/03162026',
    '/Users/aanyajain/Documents/GitHub/fs-data/FS-3/08172025',
    '/Users/aanyajain/Documents/GitHub/fs-data/FS-3/11222025',
    '/Users/aanyajain/Documents/GitHub/fs-data/FS-3/08102025',
]
OUT_DIR = 'graphs'
os.makedirs(OUT_DIR, exist_ok=True)

SENSORS = {
    'Brake Pressure Front':  ['TMAIN_DATA_BRAKES_F'],
    'Brake Pressure Rear':   ['TMAIN_DATA_BRAKES_R'],
    'Suspension Travel FL':  ['TPERIPH_FL_DATA_SUSTRAVEL'],
    'Suspension Travel FR':  ['TPERIPH_FR_DATA_SUSTRAVEL'],
    'Suspension Travel BL':  ['TPERIPH_BL_DATA_SUSTRAVEL'],
    'Suspension Travel BR':  ['TPERIPH_BR_DATA_SUSTRAVEL'],
    'Tire Temp FL':          ['TPERIPH_FL_TIRETEMP_1','TPERIPH_FL_TIRETEMP_2','TPERIPH_FL_TIRETEMP_3','TPERIPH_FL_TIRETEMP_4'],
    'Tire Temp FR':          ['TPERIPH_FR_TIRETEMP_1','TPERIPH_FR_TIRETEMP_2','TPERIPH_FR_TIRETEMP_3','TPERIPH_FR_TIRETEMP_4'],
    'Tire Temp BL':          ['TPERIPH_BL_TIRETEMP_1','TPERIPH_BL_TIRETEMP_2','TPERIPH_BL_TIRETEMP_3','TPERIPH_BL_TIRETEMP_4'],
    'Tire Temp BR':          ['TPERIPH_BR_TIRETEMP_1','TPERIPH_BR_TIRETEMP_2','TPERIPH_BR_TIRETEMP_3','TPERIPH_BR_TIRETEMP_4'],
    'Brake Temp':            ['IZZE_BRAKETEMP_S1_CH1','IZZE_BRAKETEMP_S1_CH2','IZZE_BRAKETEMP_S1_CH3','IZZE_BRAKETEMP_S1_CH4'],
    'APPS':                  ['ETC_STATUS_PEDAL_TRAVEL'],
    'Wheel Speed FL':        ['TPERIPH_FL_DATA_WHEELSPEED'],
    'Wheel Speed FR':        ['TPERIPH_FR_DATA_WHEELSPEED'],
    'Wheel Speed BL':        ['TPERIPH_BL_DATA_WHEELSPEED'],
    'Wheel Speed BR':        ['TPERIPH_BR_DATA_WHEELSPEED'],
    'Strain Gauge FL':       ['TPERIPH_FL_DATA_STRAIN'],
    'Strain Gauge FR':       ['TPERIPH_FR_DATA_STRAIN'],
    'Strain Gauge BL':       ['TPERIPH_BL_DATA_STRAIN'],
    'Strain Gauge BR':       ['TPERIPH_BR_DATA_STRAIN'],
    'Battery Voltage':       ['ACC_POWER_PACK_VOLTAGE'],
    'Battery Current':       ['ACC_POWER_CURRENT'],
    'Battery Tray Temp':     ['ACC_TRAY_TEMPS_BUSBAR','ACC_TRAY_TEMPS_PACK_FUSE','ACC_TRAY_TEMPS_COWLING'],
    'Cell Temp':             ['ACC_SEG0_TEMPS_CELL0','ACC_SEG0_TEMPS_CELL1','ACC_SEG1_TEMPS_CELL0','ACC_SEG1_TEMPS_CELL1'],
    'Cell Voltage':          ['ACC_SEG0_VOLTS_CELL0','ACC_SEG0_VOLTS_CELL1','ACC_SEG1_VOLTS_CELL0','ACC_SEG1_VOLTS_CELL1'],
}

all_files = []
for d in DATA_DIRS:
    for f in sorted(os.listdir(d)):
        if f.endswith('.parquet'):
            all_files.append(os.path.join(d, f))

print(f"Found {len(all_files)} files")

best_results = {}

for sensor_name, cols in SENSORS.items():
    best_freq = 0.0
    best_amp = 0.0
    best_file = None
    best_fft = None
    best_freq_axis = None

    for path in all_files:
        try:
            df = pl.read_parquet(path)
            df = df.with_columns([
                pl.col(c).cast(pl.Float64) for c in df.columns
                if df[c].dtype in [pl.Float32, pl.Float64]
            ])
            valid_cols = [c for c in cols if c in df.columns]
            if not valid_cols:
                continue
            time_ms = df['Time_ms'].to_numpy()
            diffs = np.diff(time_ms)
            diffs = diffs[diffs > 0]
            if len(diffs) == 0:
                continue
            dt_ms = np.median(diffs)
            fs_hz = 1000.0 / dt_ms
            data = df.select(valid_cols).mean_horizontal().drop_nulls().to_numpy().astype(float)
            if len(data) < 64:
                continue
            data -= np.mean(data)
            N = len(data)
            fft = np.abs(np.fft.rfft(data)) / N
            freq = np.fft.rfftfreq(N, d=1.0/fs_hz)
            fft[0] = 0
            peak_idx = np.argmax(fft)
            peak_freq = freq[peak_idx]
            peak_amp = fft[peak_idx]
            if peak_amp > best_amp:
                best_amp = peak_amp
                best_freq = peak_freq
                best_file = os.path.basename(path)
                best_fft = fft
                best_freq_axis = freq
        except Exception:
            continue

    if best_fft is None:
        print(f"  SKIP {sensor_name} — no data")
        continue

    best_results[sensor_name] = {'peak_freq': best_freq, 'peak_amp': best_amp, 'file': best_file}

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(best_freq_axis, best_fft, linewidth=0.8)
    ax.axvline(best_freq, color='red', linestyle='--', linewidth=1.2,
               label=f'Peak: {best_freq:.2f} Hz')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'{sensor_name} — Highest Frequency Activity\n(from {best_file})')
    ax.legend()
    ax.set_xlim(0, min(max(best_freq_axis), 50))
    plt.tight_layout()
    fname = sensor_name.replace(' ', '_').replace('/', '_') + '.png'
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=150)
    plt.close()
    print(f"  {sensor_name:30s}  peak = {best_freq:.3f} Hz  ({best_file})")

print("\n" + "="*65)
print(f"{'Sensor':<30} {'Highest Freq (Hz)':>18} {'File'}")
print("="*65)
for name, r in sorted(best_results.items(), key=lambda x: -x[1]['peak_freq']):
    print(f"{name:<30} {r['peak_freq']:>18.3f}  {r['file']}")
print(f"\nGraphs saved to: {OUT_DIR}")
