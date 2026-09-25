# Conventions

Frame, quaternion and unit conventions for this project. Fixed once, here, and referenced from module docstrings.

## Quaternions

- **Hamilton** convention (`i² = j² = k² = ijk = −1`).
- **Scalar-first**: `q = [w, x, y, z]`, stored as a `float64` array of shape `(4,)`.
- A quaternion rotates **body → inertial**:

  ```
  v_I = q ⊗ [0, v_B] ⊗ q*
  ```

- Rotation order is described by the subscripts: if `q_ab` rotates b → a and `q_bc` rotates c → b, then `q_ac = q_ab ⊗ q_bc`. The product is not commutative.

## Direction cosine matrices

- `R(q)` maps body → inertial: `v_I = R(q) v_B`. Same direction as the quaternion above.

- Rotation follows the same rule as the quaternion rule: `R(q1 ⊗ q2) = R(q1) R(q2)`.
- `R` is orthonormal with determinant +1, so `R⁻¹ = Rᵀ` corresponds to `q*`.

## SciPy interop

`scipy.spatial.transform.Rotation` stores quaternions **scalar-last** (`[x, y, z, w]`) by default, but otherwise uses the same Hamilton composition and the same rotation convention described above.

Conversion happens only in `core.quaternion.to_scipy()` and `from_scipy()`, so the ordering swap is kepts clear and happens only in one place.

## Reference frames

| Frame | Origin | Axes | Used for |
|---|---|---|---|
| **ECI** (Earth Centered Intertial) | Earth centre | x towards the vernal equinox (this is where the elliptic plane and celestial equator intersect), z along the celestial north pole, y completing the right-hand rule | Inertial reference for attitude and orbital mechanics |
| **ECEF** (Earth Centered Earth Fixed) | Earth centre | x through the intersection of the Greenwich meridian and the equator, z along the north pole; rotates with the Earth | Fixed ground locations |
| **LVLH** (orbit frame) | Spacecraft centre of mass | z towards nadir (`−r̂` towards Earth), y along the negative orbit normal (`−(r × v)/‖r × v‖`, opposite of angular momentum vector), x completing the set (points along direction of travel for a circular orbit) | Nadir-pointing reference attitude |
| **Body** | Spacecraft centre of mass | Principal axes: z along the 3U long axis (the minimum-inertia axis), x and y normal to the long side faces | Sensors, actuators, rigid-body dynamics |

## Naming

- Every vector carries its frame as a suffix: `r_eci`, `b_body`, `s_eci`.
- A quaternion names both frames in the order "to, from", matching `R(q)`: `q_eci_body` rotates body → ECI.
- Angular rate is `w_body`: the rate of the body frame relative to inertial, resolved in body axes.

## Units

SI internally, with no exceptions inside the package:

| Quantity | Unit |
|---|---|
| Angle | rad |
| Angular rate | rad/s |
| Length | m |
| Mass | kg |
| Inertia | kg·m² |
| Magnetic flux density | T |
| Torque | N·m |
| Magnetic dipole | A·m² |


## Time

Time is a float in **seconds since the scenario epoch**..
