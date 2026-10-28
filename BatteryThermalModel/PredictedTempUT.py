import os
import sys
import runpy
import unittest
import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")  # the simulation calls plt.show()

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PARQUET = os.environ.get(
    "BATTERY_PARQUET",
    os.path.expanduser("~/Desktop/fs-data/FS-3/08102025/08102025Endurance1_FirstHalf.parquet"),
)


@unittest.skipUnless(os.path.exists(PARQUET), f"parquet not found: {PARQUET}")
class PredictedTemperatureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.chdir(ROOT)
        sys.path.insert(0, ROOT)
        argv = sys.argv
        sys.argv = ["sim", PARQUET]
        try:
            cls.g = runpy.run_path(os.path.join(HERE, "Simulation_FanBatteryThermalModel.py"))
        finally:
            sys.argv = argv
        cls.f = staticmethod(cls.g["predicted_temperature"])
        cls.t = cls.g["simTime"]
        cls.mean = cls.g["logCells"].mean(axis=(1, 2))

    def test_returns_float(self):
        self.assertIsInstance(self.f(10.0), float)

    def test_start_is_ambient(self):
        self.assertAlmostEqual(self.f(0.0), self.g["ambientTemp"], places=6)

    def test_matches_simulation_at_grid_points(self):
        for i in (0, 1, 100, len(self.t) // 2, len(self.t) - 1):
            self.assertAlmostEqual(self.f(self.t[i]), self.mean[i], places=6)

    def test_between_steps_close_to_neighbours(self):
        for i in (10, 500, len(self.t) // 2):
            mid = (self.t[i] + self.t[i + 1]) / 2
            lo, hi = sorted((self.mean[i], self.mean[i + 1]))
            self.assertGreaterEqual(self.f(mid), lo - 1e-3)
            self.assertLessEqual(self.f(mid), hi + 1e-3)

    def test_random_timestamps_finite_and_plausible(self):
        rng = np.random.default_rng(0)
        for ts in rng.uniform(0, self.t[-1], 50):
            v = self.f(ts)
            self.assertTrue(np.isfinite(v))
            self.assertGreaterEqual(v, self.mean.min() - 0.1)
            self.assertLessEqual(v, self.mean.max() + 0.1)

    def test_before_start_clamps_to_first_value(self):
        self.assertAlmostEqual(self.f(-100.0), self.mean[0], places=6)

    def test_after_end_clamps_to_last_value(self):
        self.assertAlmostEqual(self.f(self.t[-1] + 100.0), self.mean[-1], places=6)


if __name__ == "__main__":
    unittest.main()
