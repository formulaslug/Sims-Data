import unittest
import pathlib
import numpy as np
import polars as pl
from numpy.testing import assert_array_almost_equal
from numpy.typing import NDArray

from fs4BatteryThermalModelingEx import (
    THERMAL_COEFFICIENT,
    thermal_ode_solve_ivp,
)

HERE = pathlib.Path(__file__).parent
PARQUET_PATH = HERE / "testdata" / "parquet" / "08102025Endurance1_FirstHalf.parquet"
COLUMN_NAME = "SME_TEMP_BusCurrent"


class TestThermalIntegrationParquet(unittest.TestCase):
    def setUp(self) -> None:
        # read only the needed column
        df = pl.read_parquet(PARQUET_PATH, columns=["SME_TEMP_BusCurrent"])
        self.current_draw: NDArray[np.float64] = df[
            "SME_TEMP_BusCurrent"
        ].to_numpy()

        # basic time grid: assume data covers 0–60s
        self.t_span = (0.0, 60.0)
        n_points = len(self.current_draw)
        self.t_eval = np.linspace(
            self.t_span[0], self.t_span[1], n_points, dtype=np.float64
        )
        self.initial_temp = 25.0

    def test_integration_runs_and_temperature_increases_on_average(self) -> None:
        sol = thermal_ode_solve_ivp(
            current_draw=self.current_draw,
            t_span=self.t_span,
            initial_temp=self.initial_temp,
            t_eval=self.t_eval,
        )

        # 1) solver succeeded
        self.assertTrue(sol.success, msg=sol.message)

        temps = sol.y[0]

        # 2) length matches t_eval / current array
        self.assertEqual(temps.shape, self.t_eval.shape)

        # 3) on average, temperature should not drop far below initial
        self.assertGreaterEqual(temps.mean(), self.initial_temp - 1.0)

        # 4) first value equals initial temperature
        self.assertAlmostEqual(temps[0], self.initial_temp, places=6)


if __name__ == "__main__":
    unittest.main()
