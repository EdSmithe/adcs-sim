"""Attitude dynamics of the rigid body satellite.

State layout: x = [q_eci_body (4), w_body (3)] shape = (7,)
q_eci_body is the quaternion representing the rotation from body frame to ECI frame
w_body is the angular velocity of the body frame in the body frame
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from adcs_sim.core import quaternion as qt
from adcs_sim.core.integrators import integrate
from adcs_sim.core.quaternion import Mat3, Vec3

TorqueFn = Callable[[float, NDArray[np.float64]], Vec3]


def pack_state(q_eci_body: ArrayLike, w_body: ArrayLike) -> NDArray[np.float64]:
    """Pack the state vector from quaternion and angular velocity."""
    q_eci_body = np.asarray(q_eci_body, dtype=np.float64)
    w_body = np.asarray(w_body, dtype=np.float64)
    return np.concatenate((q_eci_body, w_body))


def unpack_state(x: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Unpack the state vector into quaternion and angular velocity."""
    x = np.asarray(x, dtype=np.float64)
    q_eci_body = x[:4]
    w_body = x[4:]
    return q_eci_body, w_body


def zero_torque(t: float, x: NDArray[np.float64]) -> Vec3:
    """A torque function that always returns zero torque."""
    return np.zeros(3, dtype=np.float64)


@dataclass(frozen=True)
class RigidBody:
    """A rigid body satellite with attitude dynamics.

    Attributes:
        inertia: The inertia matrix of the rigid body (3x3).
        torque_fn: A function that computes the external torque on the body.
    """

    inertia: Mat3  # kg.m^2, body axes; J_inv computed in __post_init__
    J_inv: Mat3 = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the inertia matrix and compute its inverse.

        Raises:
            ValueError: If the inertia matrix is not 3x3, not symmetric, not positive
                definite, or its principal moments violate the triangle inequality.
        """
        J = np.asarray(self.inertia, dtype=np.float64)
        if J.shape != (3, 3):
            raise ValueError(f"Inertia matrix must be 3x3, got shape {J.shape}")
        if not np.allclose(J, J.T):
            raise ValueError("Inertia matrix must be symmetric")

        # Principal moments; strictly positive so that J is invertible
        moments = np.linalg.eigvalsh(J)
        tol = 1e-12 * max(np.max(np.abs(moments)), 1.0)
        if np.any(moments <= tol):
            raise ValueError(f"Inertia must be positive definite, principal moments {moments}")

        # Triangle inequality: each principal moment <= sum of the other two
        J1, J2, J3 = moments  # ascending order
        if J3 > J1 + J2 + tol:
            raise ValueError(f"Principal moments {moments} violate the triangle inequality")

        # Frozen dataclass, so bypass __setattr__
        object.__setattr__(self, "inertia", J)
        object.__setattr__(self, "J_inv", np.linalg.inv(J))

    def derivative(
        self, t: float, x: ArrayLike, torque: TorqueFn = zero_torque
    ) -> NDArray[np.float64]:
        """Compute the time derivative of the state vector.

        Args:
            t: Current time.
            x: Current state vector (quaternion and angular velocity).
            torque: Function to compute external torque.

        Returns:
            The time derivative of the state vector.
        """
        x = np.asarray(x, dtype=np.float64)

        q_eci_body, w_body = unpack_state(x)

        q_dot = 0.5 * qt.multiply(q_eci_body, np.concatenate(([0], w_body)))
        w_dot = np.linalg.solve(
            self.inertia, torque(t, x) - np.cross(w_body, self.inertia @ w_body)
        )

        return pack_state(q_dot, w_dot)

    def propagate(
        self,
        x0: ArrayLike,
        t0: float,
        t_end: float,
        dt: float,
        torque: TorqueFn = zero_torque,
        renormalise: bool = True,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Propagate the state vector over time using RK4 integration.

        Args:
            x0: Initial state vector (quaternion and angular velocity).
            t0: Initial time.
            t_end: Final time.
            dt: Time step for integration.
            torque: Function to compute external torque.
            renormalise: Whether to renormalise the quaternion after each step.

        Returns:
            A tuple containing the time values and the corresponding state vectors.
        """

        def derivative(t: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
            return self.derivative(t, x, torque)

        def post_step(t: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
            if renormalise:
                q_eci_body, w_body = unpack_state(x)
                q_eci_body = qt.normalise(q_eci_body)
                return pack_state(q_eci_body, w_body)
            return x

        return integrate(derivative, x0, t0, t_end, dt, post_step)

    @staticmethod
    def kinetic_energy(inertia: ArrayLike, w_body: ArrayLike) -> float:
        """Compute the kinetic energy of the rigid body.

        Args:
            inertia: The inertia matrix of the rigid body (3x3).
            w_body: The angular velocity of the body frame in the body frame.

        Returns:
            The kinetic energy of the rigid body.
        """
        w_body = np.asarray(w_body, dtype=np.float64)
        inertia = np.asarray(inertia, dtype=np.float64)
        return float(0.5 * w_body @ inertia @ w_body)

    @staticmethod
    def angular_momentum_eci(inertia: ArrayLike, q_eci_body: ArrayLike, w_body: ArrayLike) -> Vec3:
        """Compute the anfular momentum of the rigit body.

        Args:
            inertia: The inertia matrix of the rigid body (3x3).
            w_body: The angular velocity of the body frame in the body frame.
            q_eci_body: The rotation quaternion from the body frame to the eci frame

        Returns:
            The angular momentum vector of the rigid body.
        """
        Rq = qt.to_dcm(q_eci_body)  # Rotation matrix
        w_body = np.asarray(w_body, dtype=np.float64)
        inertia = np.asarray(inertia, dtype=np.float64)

        H_I = Rq @ inertia @ w_body
        return np.asarray(H_I, dtype=np.float64)
