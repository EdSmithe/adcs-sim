"""Quaternion maths functions.

Conventions (see docs/conventions.md):
    Hamilton product, scalar-first q = [w, x, y, z].
    q rotates body -> inertial: v_I = q ⊗ [0, v_B] ⊗ q*.
    q and -q represent the same rotation.

Quaternions are float64 arrays of shape (4,). Free functions rather than a class,
so the integrator can treat the state as one flat array.

Source: Solà (2017), "Quaternion kinematics for the error-state Kalman filter".
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

Quat = NDArray[np.float64]
Vec3 = NDArray[np.float64]
Mat3 = NDArray[np.float64]

_NORM_TOL = 1e-12  # below this a quaternion is treated as zero
_SMALL_ANGLE = 1e-8  # below this (rad) exp/log use Taylor expansions


def multiply(p: ArrayLike, q: ArrayLike) -> Quat:
    """Multiply two quaternions p and q, using the Hamilton product p ⊗ q.

    Composition: if q_ab rotates b -> a and q_bc rotates c -> b,
    then q_ac = q_ab ⊗ q_bc.
    """
    p0, p1, p2, p3 = np.asarray(p, dtype=np.float64)
    q0, q1, q2, q3 = np.asarray(q, dtype=np.float64)
    return np.array(
        [
            p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3,
            p0 * q1 + p1 * q0 + p2 * q3 - p3 * q2,
            p0 * q2 - p1 * q3 + p2 * q0 + p3 * q1,
            p0 * q3 + p1 * q2 - p2 * q1 + p3 * q0,
        ]
    )


def identity() -> Quat:
    """Return the identity quaternion."""
    return np.array([1.0, 0.0, 0.0, 0.0])


def conjugate(q: ArrayLike) -> Quat:
    """Return the conjugate of a quaternion. For a unit quaternion this is the inverse."""
    q0, q1, q2, q3 = np.asarray(q, dtype=np.float64)
    return np.array([q0, -q1, -q2, -q3])


def normalise(q: ArrayLike) -> Quat:
    """Return the normalised quaternion. Raises ValueError if ‖q‖ ≈ 0."""
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < _NORM_TOL:
        raise ValueError("Cannot normalise a (near-)zero quaternion.")
    return np.asarray(q / norm, dtype=np.float64)


def rotate(q: ArrayLike, v: ArrayLike) -> Vec3:
    """Rotate a vector v by a quaternion q: q ⊗ [0, v] ⊗ q*. Maps body -> inertial."""
    q_conj = conjugate(q)
    v_quat = np.array([0.0, *np.asarray(v, dtype=np.float64)])
    rotated_v_quat = multiply(multiply(q, v_quat), q_conj)
    return rotated_v_quat[1:]


def to_dcm(q: ArrayLike) -> Mat3:
    """Convert a quaternion to a direction cosine matrix (DCM), v_I = R(q) v_B.

    R = (w^2 - v.v) I + 2 v v^T + 2 w [v]_x, with [v]_x the skew-symmetric cross-product matrix.
    Source: Solà (2017), §2 (check eq. number).
    """
    q0, q1, q2, q3 = np.asarray(q, dtype=np.float64)
    return np.array(
        [
            [q0**2 + q1**2 - q2**2 - q3**2, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
            [2 * (q1 * q2 + q0 * q3), q0**2 - q1**2 + q2**2 - q3**2, 2 * (q2 * q3 - q0 * q1)],
            [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), q0**2 - q1**2 - q2**2 + q3**2],
        ]
    )


def from_dcm(dcm: ArrayLike) -> Quat:
    """Convert a direction cosine matrix (DCM) to a quaternion using Shepperd's method.

    Pick the largest of w², x², y², z² to divide by, avoiding numerical instabilities
    (poor conditioning) near 180° rotations.
    """
    m = np.asarray(dcm, dtype=np.float64)
    trace = np.trace(m)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def exp(phi: ArrayLike) -> Quat:
    """Return the rotation quaternion for a rotation vector phi = θu: [cos(θ/2), sin(θ/2) u]."""
    phi = np.asarray(phi, dtype=np.float64)
    norm = np.linalg.norm(phi)
    if norm < _SMALL_ANGLE:
        # Taylor expansion for small angles to avoid 0/0
        w = 1 - norm**2 / 8
        x, y, z = phi / 2
        return np.array([w, x, y, z])
    axis = phi / norm
    w = np.cos(norm / 2)
    x, y, z = np.sin(norm / 2) * axis
    return np.array([w, x, y, z])


def log(q: ArrayLike) -> Vec3:
    """Return the rotation vector phi = θu for a rotation quaternion q, with θ ∈ [0, π]."""
    q = normalise(q)
    if q[0] < 0:
        q = -q  # q and -q are the same rotation; pick w >= 0 so θ <= π
    w = q[0]
    v = q[1:]
    norm_v = np.linalg.norm(v)
    theta = 2 * np.arctan2(norm_v, w)  # well conditioned for all angles, unlike arccos(w)
    if theta < _SMALL_ANGLE:
        # Taylor expansion for small angles: θ/sin(θ/2) ≈ 2
        return 2 * v
    return np.asarray(theta * v / norm_v, dtype=np.float64)


def error(q_ref: ArrayLike, q: ArrayLike) -> Quat:
    """Return the error quaternion δq = q_ref* ⊗ q.

    The rotation taking the reference attitude to the actual one, expressed in the body
    frame. Use log(error(q_ref, q)) for the rotation-vector form.
    """
    return multiply(conjugate(q_ref), q)


def to_scipy(q: ArrayLike) -> Rotation:
    """Convert a quaternion to a scipy Rotation object (the only scalar-first -> last swap)."""
    return Rotation.from_quat(np.asarray(q, dtype=np.float64), scalar_first=True)


def from_scipy(r: Rotation) -> Quat:
    """Convert a scipy Rotation object to a scalar-first quaternion."""
    return np.asarray(r.as_quat(scalar_first=True), dtype=np.float64)
