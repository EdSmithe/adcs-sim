"""Tests for integrators.py script. Tests:

-   Exactness for-low order polynomials,ẋ = 4t³, x(0) = 0 -> x = t^4.
RK4 reduces to Simpson's rule, exact to degree 3 in the integrand.
-   Order of convergence: Harmonic oscillator ẍ = -x as a 2-state system,
T=10 s, dt = {0.1, 0.05, 0.025}. Observed order p=log₂(e(dt)/e(dt/2)).
-   Time grid and bookkeeping: integrate returns n+1 samples, t[-1] == t_end,
post_step applied every step, raises when span isn't a multiple of dt.

"""

import itertools

import numpy as np
import pytest

from adcs_sim.core.integrators import integrate, rk4_step


def test_exactness_low_order_poly():
    def f(t, x):
        return np.array([4.0 * t**3])

    t, X = integrate(f, [0.0], t0=0.0, t_end=2.0, dt=0.1)

    assert abs(X[-1, 0] - 16.0) < 1e-12
    np.testing.assert_allclose(X[:, 0], t**4, rtol=0, atol=1e-12)


def test_order_of_convergence_harmonic_oscillator():
    # State [x, v], ẋ = v, v̇ = -x; x(0) = 1, v(0) = 0 -> x = cos t, v = -sin t
    def f(t, s):
        return np.array([s[1], -s[0]])

    T = 10.0
    exact = np.array([np.cos(T), -np.sin(T)])

    errors = []
    for dt in (0.1, 0.05, 0.025):
        _, X = integrate(f, [1.0, 0.0], t0=0.0, t_end=T, dt=dt)
        errors.append(np.linalg.norm(X[-1] - exact))

    for e_coarse, e_fine in itertools.pairwise(errors):
        p = np.log2(e_coarse / e_fine)
        assert 3.8 < p < 4.2


def test_time_grid_and_sample_count():
    def f(t, x):
        return np.zeros_like(x)

    t, X = integrate(f, [1.0, 2.0, 3.0], t0=0.0, t_end=1.0, dt=0.1)

    assert t.shape == (11,)
    assert X.shape == (11, 3)
    assert t[0] == 0.0
    assert t[-1] == 1.0
    np.testing.assert_array_equal(X[0], [1.0, 2.0, 3.0])


def test_post_step_applied_every_step():
    calls = []

    def f(t, x):
        return np.ones_like(x)

    def post_step(t, x):
        calls.append(t)
        return np.zeros_like(x)

    t, X = integrate(f, [5.0], t0=0.0, t_end=1.0, dt=0.25, post_step=post_step)

    # Called once per step (not on the initial sample), with the new sample's time
    np.testing.assert_array_equal(calls, t[1:])
    # Initial state untouched, every later sample is post_step's output
    assert X[0, 0] == 5.0
    np.testing.assert_array_equal(X[1:, 0], 0.0)


def test_post_step_output_feeds_next_step():
    # ẋ = 1, post_step adds 10 -> each step advances x by exactly dt + 10
    def f(t, x):
        return np.ones_like(x)

    def post_step(t, x):
        return x + 10.0

    _, X = integrate(f, [0.0], t0=0.0, t_end=1.0, dt=0.5, post_step=post_step)

    np.testing.assert_array_equal(X[:, 0], [0.0, 10.5, 21.0])


def test_rk4_step_matches_integrate_single_step():
    def f(t, x):
        return np.array([x[1], -x[0] + t])

    x0 = np.array([0.3, -0.7])
    x1 = rk4_step(f, 1.0, x0, 0.2)
    _, X = integrate(f, x0, t0=1.0, t_end=1.2, dt=0.2)

    np.testing.assert_array_equal(X[-1], x1)


@pytest.mark.parametrize(
    "t0, t_end, dt",
    [
        (0.0, 1.0, 0.3),  # not an integer multiple
        (0.0, 1.0, 0.7),  # rounds to n = 1 but doesn't match the span
        (0.0, 0.0, 0.1),  # zero span
        (1.0, 0.0, 0.1),  # negative span
    ],
)
def test_raises_when_span_not_multiple_of_dt(t0, t_end, dt):
    def f(t, x):
        return np.zeros_like(x)

    with pytest.raises(ValueError):
        integrate(f, [0.0], t0=t0, t_end=t_end, dt=dt)
