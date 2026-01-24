import csv

class CSVControlsSource:
    """
    Loads a CSV with columns: time, throttle, brake, steer
    and returns the latest command with time <= t (hold-last-value).
    """
    def __init__(self, csv_path: str):
        self.rows = []
        with open(csv_path, "r") as f:
            r = csv.DictReader(f)
            for row in r:
                self.rows.append({
                    "time": float(row["time"]),
                    "throttle": float(row["throttle"]),
                    "brake": float(row["brake"]),
                    "steer": float(row["steer"]),
                })
        self.rows.sort(key=lambda x: x["time"])
        if not self.rows:
            raise ValueError("CSV has no rows")

        self.idx = 0

    def get_controls(self, t: float):
        # Move forward while next row time <= t
        while self.idx + 1 < len(self.rows) and self.rows[self.idx + 1]["time"] <= t:
            self.idx += 1

        row = self.rows[self.idx]
        return [row["throttle"], row["brake"], row["steer"]]
