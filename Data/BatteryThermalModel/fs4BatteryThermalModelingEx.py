import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from typing import Tuple


THERMAL_COEFFICIENT: float = 12.6316 / (49.9 * 1000.0)
ambient_temp: float = 32.0
"""
Effective thermal coefficient relating I² to temperature rate of change.

Units
-----
(°C / s) / A²
"""


def thermal_ode(
    t: float,
    T: float,
    current_draw: NDArray[np.float64],
    total_time: float,
) -> float:
    """
    Compute dT/dt for a single cell using a sampled current profile.

    The model assumes Joule heating only:

        dT/dt = THERMAL_COEFFICIENT * I(t)²

    where I(t) is obtained by indexing into ``current_draw``, which is
    assumed to be uniformly sampled over the time interval
    ``[0, total_time]``.

    Parameters
    ----------
    t : float
        Current time (seconds) at which the derivative is evaluated.
    T : float
        Current cell temperature (degrees Celsius). Currently unused,
        but included for compatibility with ``scipy.integrate.solve_ivp``.
    current_draw : NDArray[np.float64]
        1D array of current samples (amperes), uniformly spaced in time.
    total_time : float
        Total duration (seconds) covered by ``current_draw``. Used to map
        the continuous time ``t`` to an index in ``current_draw``.

    Returns
    -------
    float
        Instantaneous temperature time derivative ``dT/dt`` in
        degrees Celsius per second.
    """
    mc = 967.6 # J/K
    Area = 0.71 # m^2
    idx = int(t * len(current_draw) / total_time)
    idx = min(max(idx, 0), len(current_draw) - 1)
    I_t = current_draw[idx]
    heatingCoeff = 5.0
    coolingCoeff = 20.0
    return float(THERMAL_COEFFICIENT * (I_t ** 2) * heatingCoeff + (T-ambient_temp) * -184.90636861 * Area/mc * coolingCoeff)


def thermal_ode_solve_ivp(
    current_draw: NDArray[np.float64],
    t_span: Tuple[float, float],
    initial_temp: float,
    t_eval: NDArray[np.float64] | None = None,
):
    """
    Integrate the thermal ODE over a time interval using ``solve_ivp``.

    This solves

        dT/dt = THERMAL_COEFFICIENT * I(t)²

    where ``I(t)`` is obtained from the discrete array ``current_draw``,
    assumed to be uniformly sampled over the interval ``t_span``.

    Parameters
    ----------
    current_draw : NDArray[np.float64]
        1D array of current samples (amperes) over the driving cycle.
    t_span : tuple of float
        Integration interval ``(t0, tf)`` in seconds.
    initial_temp : float
        Initial temperature at ``t0`` in degrees Celsius.
    t_eval : NDArray[np.float64], optional
        1D array of times at which to store the computed solution.
        If ``None``, the solver chooses its own time steps.

    Returns
    -------
    OdeResult
        The object returned by ``scipy.integrate.solve_ivp``, containing
        at least ``t`` (times) and ``y`` (temperatures).
    """
    total_time = t_span[1] - t_span[0]

    sol = solve_ivp(
        fun=lambda t, T: thermal_ode(t, T, current_draw, total_time),
        t_span=t_span,
        y0=[initial_temp],
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
    )
    return sol
