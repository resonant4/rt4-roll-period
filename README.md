# rt4-roll-period

[![Tests](https://github.com/resonant4/rt4-roll-period/actions/workflows/tests.yml/badge.svg)](https://github.com/resonant4/rt4-roll-period/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Amplitude-aware roll-period GM correction and reporting for ship stability workflows. The package corrects the systematic GM overestimation caused by the standard small-angle approximation and selects the richest available method from vessel GZ tables, validated wall-sided correction, or a linear-GZ fallback.

## Publication Status

This repository is being prepared as an open technical reference implementation. It is not class-approved software, a loading computer, a substitute for a statutory stability booklet, or a replacement for professional naval architecture review.

## The Problem

During a ship inclining experiment, the standard formula:

```
GM = (C * B / T_obs)^2
```

systematically **overestimates GM** because it assumes small roll angles. At 20-degree amplitude, this causes ~1.54% GM bias, meaning the vessel is certified as more stable than it actually is.

## The Solution

The exact formula (classical Bernoulli/Euler mechanics, 1749):

```
T = T0 * (2/pi) * K(sin^2(phi_max/2))
```

where **K(m)** is the complete elliptic integral of the first kind. This corrects the amplitude dependence exactly for the linear GZ model.

**Result:** RT4 mean GM error = **0.000 mm** vs small-angle's **20.72 mm** at 20-degree amplitude.

## Install

```bash
pip install rt4-roll-period
```

## Quick Start

```python
from rt4_roll_period import (
    roll_period_exact,
    recover_gm_rt4,
    recover_gm_wall_sided,
    gm_correction_factor,
    load_gz_table_csv,
    assess_gz_table_quality,
    build_roll_period_report,
    roll_period_gz_table_ratio,
)

# Exact roll period at 20-degree amplitude
T0 = 15.0  # small-angle period (seconds)
T_exact = roll_period_exact(phi_max_deg=20.0, T0=T0)
# T_exact = 15.117... (longer than T0 due to nonlinearity)

# Recover corrected GM from observed roll period
GM_corrected = recover_gm_rt4(
    T_obs=14.8,        # observed period (s)
    phi_max_deg=18.0,   # observed max roll amplitude (deg)
    C=0.797,            # vessel C-factor
    B=28.0,             # beam (m)
)

# How much does the small-angle formula overestimate GM?
factor = gm_correction_factor(phi_max_deg=20.0)
# factor = 0.9847... (small-angle GM is ~1.54% too high)

# Wall-sided correction when BM is known and the case is inside
# the validated envelope: phi <= 30 deg, BM/GM <= 4
GM_wall = recover_gm_wall_sided(
    T_obs=14.8,
    phi_max_deg=18.0,
    C=0.797,
    B=28.0,
    BM=3.0,
)

# Vessel-specific GZ curve workflow from a CSV table:
# angle_deg,GZ_m
angle_deg, gz_m = load_gz_table_csv("stability_curve.csv")
quality = assess_gz_table_quality(angle_deg, gz_m, phi_max_deg=20.0)
ratio = roll_period_gz_table_ratio(angle_deg, gz_m, phi_max_deg=20.0)
# ratio is T/T0 for the supplied righting-arm curve

# Practitioner workflow: selects arbitrary-GZ, wall-sided, or linear fallback
report = build_roll_period_report(
    T_obs=14.8,
    phi_max_deg=18.0,
    C=0.797,
    B=28.0,
    gz_csv_path="stability_curve.csv",
)
```

## API Reference

### Core Functions

| Function | Description |
|---|---|
| `roll_period_exact(phi_max_deg, T0)` | Exact period via K(m) elliptic integral |
| `roll_period_small_angle(T0)` | Small-angle period (returns T0, for comparison) |
| `gm_correction_factor(phi_max_deg)` | Multiplicative correction: `GM_true = factor * GM_small_angle` |
| `recover_gm_rt4(T_obs, phi_max_deg, C, B)` | Recover corrected GM from observed period and amplitude |
| `wall_sided_shape_factor(phi_max_deg, bm_gm)` | Interpolated `T_wall / T_linear_K` factor for validated wall-sided range |
| `wall_sided_period_ratio(phi_max_deg, bm_gm)` | Interpolated wall-sided `T/T0` ratio |
| `wall_sided_gm_correction_factor(phi_max_deg, bm_gm)` | Wall-sided `GM_true / GM_small_angle` correction factor |
| `recover_gm_wall_sided(T_obs, phi_max_deg, C, B, BM)` | Recover GM with validated wall-sided interpolation and root solving |
| `recover_gm_small_angle(T_obs, C, B)` | Recover GM without correction (baseline) |

### Vessel Helpers

| Function | Description |
|---|---|
| `T0_from_vessel(GM, B, k_factor)` | Compute small-angle period from vessel parameters |
| `C_from_k_factor(k_factor)` | Compute Schofield C-factor from gyration radius ratio |
| `C_LOOKUP` | Dict of empirical C-factors by vessel type |

### GZ Models

| Function | Description |
|---|---|
| `gz_linear(phi_deg, GM)` | Linear GZ = GM * sin(phi) |
| `gz_wall_sided(phi_deg, GM, BM)` | Wall-sided GZ = sin(phi) * (GM + BM/2 * tan^2(phi)) |
| `roll_period_gz_numerical(gz_func, phi_max_deg, T0_ref, GM)` | Numerical period for arbitrary GZ curve |
| `load_gz_table_csv(path)` | Load and validate `angle_deg,GZ_m` CSV data |
| `gz_table_interpolator(angle_deg, gz_m)` | PCHIP interpolation for a validated GZ table |
| `gz_table_gm(angle_deg, gz_m)` | Estimate initial GM from a low-angle GZ slope fit |
| `assess_gz_table_quality(angle_deg, gz_m, phi_max_deg=None)` | Return table-quality flags and report warnings |
| `build_roll_period_report(T_obs, phi_max_deg, C, B, ...)` | Select the best correction workflow and return report-ready results |
| `roll_period_gz_table_ratio(angle_deg, gz_m, phi_max_deg)` | Compute arbitrary-GZ `T/T0` from table data |
| `roll_period_gz_table(angle_deg, gz_m, phi_max_deg, T0_ref)` | Compute arbitrary-GZ absolute period from table data |

### Analysis Tables

| Function | Description |
|---|---|
| `period_vs_amplitude_table(T0, angles)` | Table of exact vs small-angle periods |
| `gm_overestimate_table(angles)` | Table of GM overestimation by amplitude |

## Limitations

The K(m) formula is **exact for the linear GZ model** (GZ = GM * sin(phi), pure pendulum). For the **wall-sided hull model** (GZ = sin(phi) * (GM + BM/2 * tan^2(phi))), the linear K(m) correction is not exact and can be materially wrong at high BM/GM ratios and large amplitudes.

The wall-sided interpolation functions are validated for the primary product envelope:

```
0 <= phi_max_deg <= 30
0 <= BM/GM <= 4
```

Inside that envelope, numerical validation found wall-sided GM recovery max error below 0.1% on midpoint test cases. Outside that envelope, use direct numerical integration or an actual vessel GZ curve workflow rather than extrapolating the interpolation table.

For full generality with arbitrary GZ curves, use `roll_period_gz_table_ratio()` for `angle_deg,GZ_m` tables or `roll_period_gz_numerical()` for custom callables. The table API estimates the initial GM from a low-angle slope fit, so sparse or digitized curves should carry uncertainty labels.

Sensitivity checks support 5-degree table spacing as a practical minimum and 2.5-degree spacing as preferred for highly curved GZ curves or near-limit amplitudes. Use `assess_gz_table_quality()` to surface warnings for coarse spacing, sparse low-angle data, near-vanishing-stability amplitudes, and digitized-curve uncertainty.

## Mathematical Background

The roll equation of motion for a ship with linear restoring moment:

```
I * phi'' + W * GM * sin(phi) = 0
```

has the exact period solution involving the complete elliptic integral of the first kind K(m):

```
T = 4 * sqrt(I / (W * GM)) * K(sin^2(phi_max / 2))
  = T0 * (2/pi) * K(sin^2(phi_max / 2))
```

where T0 = 2*pi*sqrt(I/(W*GM)) is the small-angle period. This result is due to Bernoulli (1749) and is classical mechanics, not novel research.

The key insight for inclining experiments: when T_obs is measured at amplitude phi_max, the small-angle formula T0 = T_obs underestimates T0, which overestimates GM. The correction factor is:

```
GM_true = GM_small_angle * [pi / (2 * K(sin^2(phi_max / 2)))]^2
```

## Citation

Paper forthcoming. For now:

```bibtex
@software{rt4_roll_period,
  title = {rt4-roll-period: Amplitude-Aware Roll-Period GM Correction and Reporting},
  author = {R4RPI},
  year = {2026},
  version = {1.1.0},
  url = {https://github.com/resonant4/rt4-roll-period},
  license = {MIT}
}
```

## License

MIT. See [LICENSE](LICENSE).
