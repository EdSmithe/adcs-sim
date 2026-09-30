"""Tests for the rigid body dynamics module.

Test inertia validation in RigidBody.__post_init__:
 - A valid inertia (diagonal or rotated) gives J_inv with J @ J_inv = I
 - Non-3x3 inertia raises ValueError
 - Non-symmetric inertia raises ValueError
 - Inertia with a zero or negative principal moment raises ValueError
 - Principal moments violating the triangle inequality raise ValueError
 - Principal moments on the triangle inequality boundary (flat plate) are accepted
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from adcs_sim.core import quaternion as qt
from adcs_sim.dynamics.rigid_body import RigidBody


def rotated(moments: list[float], q: np.ndarray) -> np.ndarray:
    """Inertia with the given principal moments expressed in axes rotated by q."""
    R = qt.to_dcm(q)
    return R @ np.diag(moments) @ R.T


def test_j_inv_diagonal() -> None:
    J = np.diag([10.0, 12.0, 15.0])
    body = RigidBody(J)
    np.testing.assert_allclose(body.J_inv, np.diag([1 / 10.0, 1 / 12.0, 1 / 15.0]))


@given(
    st.lists(st.floats(-1.0, 1.0), min_size=4, max_size=4).filter(
        lambda v: np.linalg.norm(v) > 1e-3
    )
)
def test_j_inv_rotated(q: list[float]) -> None:
    J = rotated([10.0, 12.0, 15.0], qt.normalise(np.asarray(q)))
    body = RigidBody(J)
    np.testing.assert_allclose(J @ body.J_inv, np.eye(3), atol=1e-12)


def test_flat_plate_on_triangle_boundary_accepted() -> None:
    # Thin flat plate: J3 = J1 + J2
    RigidBody(np.diag([1.0, 2.0, 3.0]))


@pytest.mark.parametrize("J", [np.eye(2), np.ones(3), np.eye(4)])
def test_wrong_shape_raises(J: np.ndarray) -> None:
    with pytest.raises(ValueError, match="3x3"):
        RigidBody(J)


def test_non_symmetric_raises() -> None:
    J = np.diag([10.0, 12.0, 15.0])
    J[0, 1] = 1.0
    with pytest.raises(ValueError, match="symmetric"):
        RigidBody(J)


@pytest.mark.parametrize("moments", [[0.0, 1.0, 1.0], [-1.0, 5.0, 5.0], [-1.0, -1.0, -1.0]])
def test_not_positive_definite_raises(moments: list[float]) -> None:
    with pytest.raises(ValueError, match="positive definite"):
        RigidBody(np.diag(moments))


def test_not_positive_definite_rotated_raises() -> None:
    # Positive diagonal entries but a negative principal moment
    J = rotated([-1.0, 5.0, 5.0], qt.normalise(np.array([0.9, 0.1, 0.3, -0.2])))
    assert np.all(np.diag(J) > 0)
    with pytest.raises(ValueError, match="positive definite"):
        RigidBody(J)


@pytest.mark.parametrize("moments", [[1.0, 1.0, 3.0], [5.0, 1.0, 1.0], [1.0, 10.0, 2.0]])
def test_triangle_inequality_violation_raises(moments: list[float]) -> None:
    with pytest.raises(ValueError, match="triangle inequality"):
        RigidBody(np.diag(moments))


def test_triangle_inequality_violation_rotated_raises() -> None:
    J = rotated([1.0, 1.0, 3.0], qt.normalise(np.array([0.5, 0.5, -0.5, 0.5])))
    with pytest.raises(ValueError, match="triangle inequality"):
        RigidBody(J)
