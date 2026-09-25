# adcs-sim

A quaternion-based attitude determination and control simulation for a 3U CubeSat in a 500 km sun-synchronous orbit: B-dot detumbling, a Multiplicative Extended Kalman Filter fusing gyro, magnetometer and sun sensor, and reaction-wheel pointing control.

> **Status: work in progress.** Currently in Phase 0 (foundations: repo setup, conventions, quaternion library). See [`docs/planning/PROJECT_PLAN.md`](docs/planning/PROJECT_PLAN.md) for the full plan.

## Requirements

The simulation is built to demonstrate the following. Results will be reported against each one as phases complete.

| ID | Requirement | Status |
|---|---|---|
| R-1 | Detumble from 10 °/s per axis to < 0.5 °/s magnitude within 3 orbits using magnetorquers only | Not started |
| R-2 | MEKF attitude estimate error < 2° (3σ) in sunlight, < 5° through eclipse, after convergence | Not started |
| R-3 | MEKF gyro bias estimate converges to within 0.01 °/s of truth within one orbit | Not started |
| R-4 | Nadir pointing held to < 1° (RMS) over one full orbit, wheels below 80% momentum capacity | Not started |
| R-5 | Slew of 90° completed in < 120 s with no overshoot beyond 5% | Not started |

## Quick start

```bash
git clone https://github.com/<your-username>/adcs-sim.git
cd adcs-sim
uv sync
uv run pytest
```

## Design principles

- **Truth model and flight software are separate code paths.** The truth simulation runs at 100 Hz; the flight-software loop (estimator + controller) runs at 10 Hz and only sees sensor outputs.

Conventions (quaternions, frames, units) are defined in [`docs/conventions.md`](docs/conventions.md).

## Documentation

- [`docs/conventions.md`](docs/conventions.md): quaternion convention, reference frames, units