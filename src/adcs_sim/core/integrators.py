"""Generic fixed-step RK4 integrator reusable for anything"""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

StateDerivative = Callable[[float, NDArray[np.float64]], NDArray[np.float64]]
PostStep = Callable[[float, NDArray[np.float64]], NDArray[np.float64]]


def rk4_step(f: StateDerivative, t: float, x: ArrayLike, dt: float) -> NDArray[np.float64]:
    """Perform a single RK4 step

    Args:
        f: function that computes the state derivative
        t: current time
        x: current state
        dt: time step

    Returns:
        The new state after the RK4 step
    """
    x = np.asarray(x, dtype=np.float64)

    k1 = f(t, x)
    k2 = f(t + dt / 2, x + dt / 2 * k1)
    k3 = f(t + dt / 2, x + dt / 2 * k2)
    k4 = f(t + dt, x + dt * k3)

    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(
    f: StateDerivative,
    x0: ArrayLike,
    t0: float,
    t_end: float,
    dt: float,
    post_step: PostStep | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Integrate a system of ODEs using RK4"""

    x0 = np.asarray(x0, dtype=np.float64)

    n = round((t_end - t0) / dt)
    if n < 1 or not np.isclose(n * dt, t_end - t0, rtol=0, atol=1e-9 * max(1.0, abs(t_end))):
        raise ValueError("(t_end - t0) must be a positive integer multiple of dt")
    t_values = t0 + dt * np.arange(n + 1, dtype=np.float64)

    x_values = np.empty((len(t_values), len(x0)), dtype=np.float64)
    x_values[0] = x0

    for i in range(1, len(t_values)):
        x_values[i] = rk4_step(f, t_values[i - 1], x_values[i - 1], dt)
        if post_step is not None:
            x_values[i] = post_step(t_values[i], x_values[i])

    return t_values, x_values
