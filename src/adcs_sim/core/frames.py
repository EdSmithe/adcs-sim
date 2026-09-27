"""Module uses quaternion.py to implement rotations between ECI, ECEF and LVLH frames.

Naming follows docs/conventions.md: q_a_b rotates frame b -> frame a, so v_a = rotate(q_a_b, v_b).
GMST uses the linear (degree-form) approximation of the IAU-82 model.
GMST ignores precession, nutation and polar motion. The difference is roughly 0.35 degrees in 2026.
"""

import datetime

import numpy as np
from numpy.typing import ArrayLike, NDArray

from adcs_sim.core import quaternion as qt

Quat = NDArray[np.float64]
Vec3 = NDArray[np.float64]
Mat3 = NDArray[np.float64]

OMEGA_EARTH = 7.2921150e-5  # rad/s, Earth's rotation rate, from WGS-84 model


def julian_date(dt: datetime.datetime) -> float:
    """Convert a timezone-aware datetime to Julian date.

    Raises ValueError if dt is naive (has no timezone).
    """
    jd_epoch = 2440587.5  # Julian date of the Unix epoch (1970-01-01 00:00:00 UTC)
    if dt.tzinfo is None:
        raise ValueError("Datetime object must be timezone-aware (UTC).")

    unix_time = dt.timestamp()  # Seconds since Unix epoch
    jd = jd_epoch + unix_time / 86400.0  # Convert seconds to days and add to Julian date of epoch
    return jd


def gmst(jd: float) -> float:
    """Calculate Greenwich Mean Sidereal Time (GMST) in radians in [0, 2π) from Julian date.

    Uses the linear approximation of the IAU-82 model.
    """
    d = jd - 2451545.0  # Days since J2000.0
    gmst_deg = 280.46061837 + 360.98564736629 * d  # in degrees
    return float(np.radians(gmst_deg % 360))  # Convert to radians and wrap to [0, 2π)


# --------- ECI <-> ECEF rotation functions ---------
def eci_to_ecef_rotation(theta: float) -> Quat:
    """Return the quaternion rotating ECI -> ECEF (q_ecef_eci) for a GMST angle theta in radians.

    r_ecef = qt.rotate(eci_to_ecef_rotation(theta), r_eci).
    """
    # Rotation about the z-axis by -theta (ECI to ECEF)
    return qt.exp(np.array([0.0, 0.0, -theta]))  # Use the exponential map to get the quaternion


def ecef_to_eci_rotation(theta: float) -> Quat:
    """Return the quaternion rotating ECEF -> ECI (q_eci_ecef) for a GMST angle theta in radians.

    r_eci = qt.rotate(ecef_to_eci_rotation(theta), r_ecef).
    """
    # Rotation about the z-axis by theta (ECEF to ECI), the inverse (conjugate) of ECI to ECEF
    return qt.conjugate(eci_to_ecef_rotation(theta))


def velocity_eci_to_ecef(theta: float, r_eci: ArrayLike, v_eci: ArrayLike) -> Vec3:
    """Convert velocity from ECI to ECEF frame, given position and velocity in ECI.

    v_ecef = R v_eci - cross(ω, r_ecef), where R rotates ECI -> ECEF and ω = [0, 0, OMEGA_EARTH].
    """
    q_ecef_eci = eci_to_ecef_rotation(theta)
    # Convert position to ECEF
    r_ecef = qt.rotate(q_ecef_eci, r_eci)
    # Earth's rotation vector in ECEF frame
    omega_ecef = np.array([0.0, 0.0, OMEGA_EARTH])
    # Compute the velocity in ECEF frame
    v_ecef: Vec3 = qt.rotate(q_ecef_eci, v_eci) - np.cross(omega_ecef, r_ecef)
    return v_ecef


# LVLH from position and velocity
def dcm_eci_lvlh(r_eci: ArrayLike, v_eci: ArrayLike) -> Mat3:
    """Return the DCM taking LVLH vectors to ECI, v_eci = R v_lvlh.

    Columns are the LVLH x, y, z axes expressed in ECI.
    Raises ValueError if r or v is near zero, or if r and v are (near-)parallel.
    """
    r = np.asarray(r_eci, dtype=np.float64)
    v = np.asarray(v_eci, dtype=np.float64)
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)

    # Check for near-zero vectors to avoid division by zero
    if r_norm < 1e-6 or v_norm < 1e-6:
        raise ValueError("Position and velocity vectors must be non-zero.")

    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)
    if h_norm < 1e-12 * r_norm * v_norm:
        raise ValueError("Position and velocity are parallel; LVLH is undefined.")

    # Compute the LVLH axes
    z_lvlh = -r / r_norm  # Negative radial direction
    y_lvlh = -h / h_norm  # Negative orbit normal
    x_lvlh = np.cross(y_lvlh, z_lvlh)  # Along-track direction
    # Construct the rotation matrix from LVLH to ECI
    dcm: Mat3 = np.column_stack((x_lvlh, y_lvlh, z_lvlh))  # Columns are the LVLH axes in ECI frame
    return dcm


def q_eci_lvlh(r_eci: ArrayLike, v_eci: ArrayLike) -> Quat:
    """Return the quaternion rotating LVLH -> ECI (q_eci_lvlh), v_eci = qt.rotate(q, v_lvlh)."""
    dcm = dcm_eci_lvlh(r_eci, v_eci)
    return qt.from_dcm(dcm)
