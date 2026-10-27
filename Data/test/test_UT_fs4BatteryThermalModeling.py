import unittest
import numpy as np
from numpy.typing import NDArray
from numpy.testing import assert_array_almost_equal

from fs4BatteryThermalModelingEx import (
    THERMAL_COEFFICIENT,
    thermal_ode_solve_ivp,
)

class TestThermal_ode_solve_ivp(unittest.TestCase):
    def setUp(self) -> None:
        self.t_span = (0.0, 10.0)
        self.n_points = 1000
        self.t_eval: NDArray[np.float64] = np.linspace(
            self.t_span[0], self.t_span[1], self.n_points, dtype=np.float64
        )
        self.initial_temp: float = 25.0

    def test_constant_current_matches_analytic(self) -> None:
        """
        For constant I, dT/dt = k * I^2 -> T(t) = T0 + k * I^2 * t
        """
        I_const = 5.0
        current_draw = np.full(self.n_points, I_const, dtype=np.float64)

        sol = thermal_ode_solve_ivp(
            current_draw=current_draw,
            t_span=self.t_span,
            initial_temp=self.initial_temp,
            t_eval=self.t_eval,
        )

        self.assertTrue(sol.success, msg=sol.message)
        temps = sol.y[0]

        expected = self.initial_temp + THERMAL_COEFFICIENT * (I_const ** 2) * self.t_eval
        assert_array_almost_equal(temps, expected, decimal=4)

    def test_zero_current_results_in_constant_temperature(self) -> None:
        zero_current = np.zeros(self.n_points, dtype=np.float64)

        sol = thermal_ode_solve_ivp(
            current_draw=zero_current,
            t_span=self.t_span,
            initial_temp=self.initial_temp,
            t_eval=self.t_eval,
        )

        self.assertTrue(sol.success, msg=sol.message)
        temps = sol.y[0]
        expected = np.full_like(temps, self.initial_temp)
        assert_array_almost_equal(temps, expected, decimal=8)

    def test_extreme_currents_no_nan_or_inf(self) -> None:
        extreme_currents = np.linspace(0.0, 1e3, self.n_points, dtype=np.float64)

        sol = thermal_ode_solve_ivp(
            current_draw=extreme_currents,
            t_span=self.t_span,
            initial_temp=self.initial_temp,
            t_eval=self.t_eval,
        )

        self.assertTrue(sol.success, msg=sol.message)
        temps = sol.y[0]
        self.assertFalse(np.isnan(temps).any())
        self.assertFalse(np.isinf(temps).any())


if __name__ == "__main__":
    unittest.main()
