# Balloon Wind-Band Simulator

6-DOF helium weather balloon flight simulator with COESA-76 atmospheric modeling, layered wind bands, and RK45 numerical integration.

---

## Usage

```bash
python main.py
```

Configure constants and wind bands at the top of `main.py`. Outputs `fig.png`.

---

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `helium_mass` | 1.0 kg | Helium fill mass |
| `skin_mass` | 2.5 kg | Skin + payload mass |
| `drag_coeff` | 0.5 | Cd (sphere) |
| `burst_radius` | 8 m | Burst condition |
| `time_step` | 1 s | Max RK45 step size |
| `t_end` | 50000 s | Simulation time limit |
| `launch_time` | `2025-01-01 09:00:00` | Wall-clock launch time |

Wind bands take a 3D velocity vector [km/hr] and an altitude range [m]:

```python
first_range,  first_vector  = [0, 5000],      [-5, 0,   7]
second_range, second_vector = [5000, 10000],  [ 0, 5, -20]
third_range,  third_vector  = [10000, 1e9],   [10, 0,   0]
```

The z-component of each wind vector drives a vertical downdraft or updraft within that band. The balloon has inertia in all three axes and responds to wind continuously via drag rather than snapping to the wind velocity.

---

## Physics Model

**Atmosphere**: COESA-76 standard atmosphere (ρ, T, P as functions of altitude). Valid to 86 km.

**Balloon volume**: Ideal gas law — `V = m_He · R_He · T / P`. Radius from sphere assumption: `r = (3V/4π)^(1/3)`.

**Drag** (`_drag`): Applied uniformly across all three axes against velocity relative to the local wind:
`F_drag = -0.5 · sign(v_rel) · Cd · ρ · πr² · v_rel²`

**Vertical** (`vertical_acceleration`):
`a_z = (ρ_air · V · g) / m_total + F_drag,z - g`

**Horizontal**: `a_x`, `a_y` are pure drag terms against wind-relative velocity. The balloon accelerates toward the wind vector over time rather than matching it instantly.

**Integration**: RK45 via `scipy.solve_ivp`, `max_step = 1s`. If no burst occurs within `t_end`, the full trajectory is plotted and a no-burst message is logged.

---

## Assumptions & Limitations

- Perfect sphere geometry; no skin elasticity or deformation
- No heat transfer — helium temperature equals ambient at all times
- Wind bands are step functions in altitude; no vertical shear or interpolation between bands
- Drag is applied independently per axis rather than against the full 3D relative velocity vector
- No turbulence or stochastic atmospheric variation
- Simulation terminates at burst; no descent phase