import csv
from pathlib import Path

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "throttle", "brake", "steer"])
        w.writerows(rows)

if __name__ == "__main__":
    # Scenario A: straight accel then coast then brake
    rows_a = [
        (0.0, 0.0, 0.0, 0.0),
        (2.0, 0.75, 0.0, 0.0),
        (10.0, 0.2, 0.0, 0.0),
        (14.0, 0.0, 0.4, 0.0),
        (18.0, 0.0, 0.0, 0.0),
    ]
    write_csv(Path("integration/synth_straight.csv"), rows_a)

    # Scenario B: your example (steer after 2.3s)
    rows_b = [
        (0.0, 0.0, 0.0, 0.0),
        (2.3, 0.75, 0.0, 0.2),
        (8.0, 0.5, 0.0, 0.2),
        (12.0, 0.3, 0.0, 0.0),
    ]
    write_csv(Path("integration/synth_steer.csv"), rows_b)

    # Scenario C: rapid steering changes (tests timeSinceLastSteer logic)
    rows_c = [
        (0.0, 0.3, 0.0, 0.0),
        (2.0, 0.6, 0.0, 0.1),
        (2.5, 0.6, 0.0, -0.1),
        (3.0, 0.6, 0.0, 0.1),
        (3.5, 0.6, 0.0, 0.0),
    ]
    write_csv(Path("integration/synth_steer_changes.csv"), rows_c)

    print("Wrote synthetic CSV files in integration/")
