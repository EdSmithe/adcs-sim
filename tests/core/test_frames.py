"""Tests for the frames module.

Test GMST using known values:
 - At J2000 GMST = 280.46061837 degrees.
 - At 1992-08-20 12:14 UT1 GMST = 152.578788 degrees (Vallado Example 3-5).
 - One sidereal day later GMST is back to the same angle.
Test ECI <-> ECEF rotation:
 - Basic check theta = 90 deg: ECEF x-axis [1, 0, 0] is ECI [0, 1, 0]
 - to_dcm(q_eci_ecef) equals R_z(theta)
 - For random theta and r, round-trip ecef_to_eci(eci_to_ecef(r)) = r (norm and z unchanged)
 - A point fixed on the Earth has zero ECEF velocity
Test LVLH:
 - Circular equatorial orbit, r = [R, 0, 0], v = [0, V, 0], gives LVLH axes
   x = [0, 1, 0], y = [0, 0, -1], z = [-1, 0, 0]
 - For random non-parallel r, v, the DCM is orthonormal with det +1, z dot r = -1, y || -(r x v)
 - If v perpendicular to r (circular), x || v
 - r cross v = 0 raises ValueError
"""

import datetime

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from adcs_sim.core import frames as fr
from adcs_sim.core.frames import Vec3

SIDEREAL_DAY = 86164.0905  # seconds

vectors = st.lists(st.floats(-1, 1), min_size=3, max_size=3).map(np.array)


@st.composite
def random_datetime(draw: st.DrawFn) -> datetime.datetime:
    """Generate a random datetime object."""
    year = draw(st.integers(min_value=2000, max_value=2100))
    month = draw(st.integers(min_value=1, max_value=12))
    day = draw(st.integers(min_value=1, max_value=28))  # Simplify to avoid month length issues
    hour = draw(st.integers(min_value=0, max_value=23))
    minute = draw(st.integers(min_value=0, max_value=59))
    second = draw(st.integers(min_value=0, max_value=59))
    return datetime.datetime(year, month, day, hour, minute, second, tzinfo=datetime.UTC)


def angle_diff(a: float, b: float) -> float:
    """Return a - b wrapped to [-π, π), so angles either side of 0/2π compare correctly."""
    return float((a - b + np.pi) % (2 * np.pi) - np.pi)


# ------ test GMST ------


def test_gmst_known_values() -> None:
    """Test GMST against known values."""
    # J2000.0 epoch
    jd_j2000 = 2451545.0
    gmst_j2000 = fr.gmst(jd_j2000)
    np.testing.assert_allclose(np.degrees(gmst_j2000), 280.46061837, rtol=0, atol=1e-6)

    # 1992-08-20 12:14 UT1 (the linear model omits the tiny T² term, hence atol 1e-5 deg)
    dt_1992 = datetime.datetime(1992, 8, 20, 12, 14, 0, tzinfo=datetime.UTC)
    jd_1992 = fr.julian_date(dt_1992)
    gmst_1992 = fr.gmst(jd_1992)
    np.testing.assert_allclose(np.degrees(gmst_1992), 152.578788, rtol=0, atol=1e-5)

    # One sidereal day later (approximately 23h 56m 4s)
    dt_next_sidereal_day = dt_1992 + datetime.timedelta(seconds=SIDEREAL_DAY)
    gmst_next_sidereal_day = fr.gmst(fr.julian_date(dt_next_sidereal_day))
    assert abs(angle_diff(gmst_next_sidereal_day, gmst_1992)) < 1e-8


def test_julian_date_rejects_naive_datetime() -> None:
    """A datetime without a timezone raises ValueError."""
    with pytest.raises(ValueError):
        fr.julian_date(datetime.datetime(2000, 1, 1, 12, 0, 0))


@given(random_datetime())
def test_time_chain_properties(dt: datetime.datetime) -> None:
    """One solar day adds 1 to the JD and ~0.9856 deg to GMST; one sidereal day adds 0 to GMST."""
    jd = fr.julian_date(dt)
    jd_solar = fr.julian_date(dt + datetime.timedelta(days=1))
    jd_sidereal = fr.julian_date(dt + datetime.timedelta(seconds=SIDEREAL_DAY))
    np.testing.assert_allclose(jd_solar - jd, 1.0, rtol=0, atol=1e-8)
    solar_advance = np.degrees(angle_diff(fr.gmst(jd_solar), fr.gmst(jd)))
    np.testing.assert_allclose(solar_advance, 0.98564736629, rtol=0, atol=1e-6)
    assert abs(angle_diff(fr.gmst(jd_sidereal), fr.gmst(jd))) < 1e-6


# ------- test ECI <-> ECEF rotation -------


def test_eci_ecef_basic() -> None:
    """At theta = 90 deg, Greenwich (ECEF x) lies along ECI +y, and ECI x lies along ECEF -y."""
    theta = np.pi / 2  # 90 degrees
    ecef_x_in_eci = fr.qt.rotate(fr.ecef_to_eci_rotation(theta), [1, 0, 0])
    eci_x_in_ecef = fr.qt.rotate(fr.eci_to_ecef_rotation(theta), [1, 0, 0])
    np.testing.assert_allclose(ecef_x_in_eci, [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(eci_x_in_ecef, [0, -1, 0], atol=1e-12)


@pytest.mark.parametrize("theta", [0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2])
def test_dcm_eci_ecef_equals_rz(theta: float) -> None:
    """Check that the DCM of q_eci_ecef (ECEF -> ECI) equals R_z(theta)."""
    dcm = fr.qt.to_dcm(fr.ecef_to_eci_rotation(theta))
    rz = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    np.testing.assert_allclose(dcm, rz, atol=1e-12)


@given(st.floats(-2 * np.pi, 2 * np.pi), vectors)
def test_eci_ecef_round_trip(theta: float, r_eci: Vec3) -> None:
    """For random theta and r, ecef_to_eci(eci_to_ecef(r)) = r, with norm and z unchanged."""
    r_ecef = fr.qt.rotate(fr.eci_to_ecef_rotation(theta), r_eci)
    r_recovered = fr.qt.rotate(fr.ecef_to_eci_rotation(theta), r_ecef)
    np.testing.assert_allclose(r_recovered, r_eci, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(r_ecef), np.linalg.norm(r_eci), atol=1e-12)
    np.testing.assert_allclose(r_ecef[2], r_eci[2], atol=1e-12)  # z component unchanged


@given(st.floats(0, 2 * np.pi), vectors)
def test_earth_fixed_point_has_zero_ecef_velocity(theta: float, r_eci: Vec3) -> None:
    """A point rotating with the Earth (v_eci = ω x r_eci) is stationary in ECEF."""
    v_eci = np.cross([0.0, 0.0, fr.OMEGA_EARTH], r_eci)
    v_ecef = fr.velocity_eci_to_ecef(theta, r_eci, v_eci)
    np.testing.assert_allclose(v_ecef, [0, 0, 0], atol=1e-15)


# ------- test LVLH -------


def test_lvlh_circular_equatorial() -> None:
    """Check LVLH axes for a circular equatorial orbit."""
    R = 7000e3  # 7000 km
    V = 7.5e3  # 7.5 km/s
    r_eci = np.array([R, 0, 0])
    v_eci = np.array([0, V, 0])
    q_lvlh = fr.q_eci_lvlh(r_eci, v_eci)
    dcm_lvlh = fr.qt.to_dcm(q_lvlh)
    np.testing.assert_allclose(dcm_lvlh, fr.dcm_eci_lvlh(r_eci, v_eci), atol=1e-12)
    x_lvlh = dcm_lvlh[:, 0]
    y_lvlh = dcm_lvlh[:, 1]
    z_lvlh = dcm_lvlh[:, 2]
    np.testing.assert_allclose(x_lvlh, [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(y_lvlh, [0, 0, -1], atol=1e-12)
    np.testing.assert_allclose(z_lvlh, [-1, 0, 0], atol=1e-12)


@given(vectors, vectors)
def test_lvlh_orthonormal(r_eci: Vec3, v_eci: Vec3) -> None:
    """For random non-parallel r, v the DCM is orthonormal, det +1, z·r̂ = -1 and y || -(r x v)."""
    assume(np.linalg.norm(r_eci) > 1e-3 and np.linalg.norm(v_eci) > 1e-3)
    assume(np.linalg.norm(np.cross(r_eci, v_eci)) > 1e-3)  # Not parallel
    dcm_lvlh = fr.qt.to_dcm(fr.q_eci_lvlh(r_eci, v_eci))
    # Check orthonormality
    np.testing.assert_allclose(dcm_lvlh @ dcm_lvlh.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(dcm_lvlh), 1.0, atol=1e-12)
    y_lvlh = dcm_lvlh[:, 1]
    z_lvlh = dcm_lvlh[:, 2]
    # Check z dot r = -1
    np.testing.assert_allclose(np.dot(z_lvlh, r_eci / np.linalg.norm(r_eci)), -1.0, atol=1e-12)
    # Check y || -(r x v)
    neg_h = -np.cross(r_eci, v_eci)
    np.testing.assert_allclose(y_lvlh, neg_h / np.linalg.norm(neg_h), atol=1e-12)


@given(vectors, vectors)
def test_lvlh_circular_x_along_v(r_eci: Vec3, v_raw: Vec3) -> None:
    """If v is perpendicular to r (circular orbit), x || v."""
    assume(np.linalg.norm(r_eci) > 1e-3)
    r_hat = r_eci / np.linalg.norm(r_eci)
    v_eci = v_raw - np.dot(v_raw, r_hat) * r_hat  # Remove the radial component
    assume(np.linalg.norm(v_eci) > 1e-3)
    x_lvlh = fr.dcm_eci_lvlh(r_eci, v_eci)[:, 0]
    np.testing.assert_allclose(x_lvlh, v_eci / np.linalg.norm(v_eci), atol=1e-12)


def test_lvlh_r_cross_v_zero_raises() -> None:
    """r cross v = 0 raises ValueError."""
    r_eci = np.array([7000e3, 0, 0])
    v_eci = np.array([7.5e3, 0, 0])  # Parallel to r
    with pytest.raises(ValueError):
        fr.q_eci_lvlh(r_eci, v_eci)
