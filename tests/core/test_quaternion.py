"""Tests for the quaternion module.

Starts by generating random unit quaternions, then uses a SciPy cross-check.
Sign ambiguity: q and -q represent the same rotation, so we need to account for that in the tests.
"""

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from adcs_sim.core import quaternion as qt

Arr = NDArray[np.float64]


@st.composite
def unit_quaternion(draw: st.DrawFn) -> Arr:
    """Generate a random unit quaternion."""
    q = np.array(draw(st.lists(st.floats(-1, 1), min_size=4, max_size=4)))
    n = np.linalg.norm(q)
    assume(n > 1e-3)  # Avoid normalising a near-zero vector
    return np.asarray(q / n, dtype=np.float64)


vectors = st.lists(st.floats(-1, 1), min_size=3, max_size=3).map(np.array)


def assert_same_rotation(q1: Arr, q2: Arr, atol: float = 1e-12) -> None:
    """Assert q1 and q2 represent the same rotation, i.e. q1 = ±q2."""
    assert np.allclose(q1, q2, atol=atol) or np.allclose(q1, -q2, atol=atol), f"{q1} != ±{q2}"


@given(unit_quaternion(), unit_quaternion())
def test_composition_matches_scipy(q1: Arr, q2: Arr) -> None:
    """Test that quaternion multiplication matches scipy's Rotation multiplication."""
    r1 = Rotation.from_quat(q1, scalar_first=True)
    r2 = Rotation.from_quat(q2, scalar_first=True)
    expected = (r1 * r2).as_matrix()
    np.testing.assert_allclose(qt.to_dcm(qt.multiply(q1, q2)), expected, atol=1e-12)


@given(unit_quaternion())
def test_q_times_conjugate_is_identity(q: Arr) -> None:
    """Test that q * q_conjugate is the identity quaternion."""
    np.testing.assert_allclose(qt.multiply(q, qt.conjugate(q)), qt.identity(), atol=1e-12)


@given(unit_quaternion())
def test_dcm_is_orthonormal(q: Arr) -> None:
    """Test that the DCM from a quaternion is orthonormal and also has unit determinant."""
    dcm = qt.to_dcm(q)
    np.testing.assert_allclose(dcm @ dcm.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(dcm), 1.0, atol=1e-12)


@given(unit_quaternion(), unit_quaternion())
def test_dcm_of_product_is_product_of_dcms(q1: Arr, q2: Arr) -> None:
    """Test that DCM(q1 ⊗ q2) = DCM(q1) DCM(q2)."""
    dcm1 = qt.to_dcm(q1)
    dcm2 = qt.to_dcm(q2)
    dcm_product = qt.to_dcm(qt.multiply(q1, q2))
    np.testing.assert_allclose(dcm_product, dcm1 @ dcm2, atol=1e-12)


@given(unit_quaternion())
def test_from_dcm_to_dcm_recovers_quaternion(q: Arr) -> None:
    """Test that converting a quaternion to DCM and back recovers the original (up to sign)."""
    assert_same_rotation(qt.from_dcm(qt.to_dcm(q)), q)


@given(unit_quaternion())
def test_exp_log_q_equals_q(q: Arr) -> None:
    """Test that exp(log(q)) = q (up to sign)."""
    assert_same_rotation(qt.exp(qt.log(q)), q)


@given(unit_quaternion())
def test_log_angle_is_at_most_pi(q: Arr) -> None:
    """Test that log returns the shortest rotation, θ ∈ [0, π]."""
    assert np.linalg.norm(qt.log(q)) <= np.pi + 1e-12


@given(unit_quaternion(), vectors)
def test_rotate_is_to_dcm(q: Arr, v: Arr) -> None:
    """Test that the rotate function is equivalent to the DCM multiplication."""
    np.testing.assert_allclose(qt.rotate(q, v), qt.to_dcm(q) @ v, atol=1e-12)


@given(unit_quaternion(), vectors)
def test_rotate_preserves_norm(q: Arr, v: Arr) -> None:
    """Test that rotating a vector preserves its norm."""
    assert np.isclose(np.linalg.norm(qt.rotate(q, v)), np.linalg.norm(v), atol=1e-12)


@given(unit_quaternion(), unit_quaternion())
def test_error_composes_back(q_ref: Arr, q: Arr) -> None:
    """Test that q_ref ⊗ δq = q, so δq really is the rotation from q_ref to q."""
    assert_same_rotation(qt.multiply(q_ref, qt.error(q_ref, q)), q)


@given(unit_quaternion())
def test_scipy_round_trip(q: Arr) -> None:
    """Test that to_scipy and from_scipy agrees with the DCM."""
    r = qt.to_scipy(q)
    np.testing.assert_allclose(r.as_matrix(), qt.to_dcm(q), atol=1e-12)
    assert_same_rotation(qt.from_scipy(r), q)
