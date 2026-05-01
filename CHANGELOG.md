# Changelog

## [1.1.0] - 2026-05-01

### Added
- Validated wall-sided correction helpers and inverse GM recovery.
- Wall-sided recovery validation with numerical ground truth.
- Documentation for the validated wall-sided envelope: `phi <= 30 deg`, `BM/GM <= 4`.
- Arbitrary GZ table workflow for `angle_deg,GZ_m` CSV data, including PCHIP interpolation, initial-GM estimation, and `T/T0` period ratio calculation.
- Public digitized GZ table fixture from TSB Canada Le Marsouin I Appendix D Graph 1.
- Table-spacing and digitization-noise sensitivity analysis.
- `assess_gz_table_quality()` for arbitrary-GZ reporting warnings and quality flags.
- Report-style arbitrary-GZ workflow example and outputs.
- `build_roll_period_report()` practitioner workflow selector for arbitrary-GZ, wall-sided, and linear correction paths.

### Changed
- Improved arbitrary-GZ initial GM estimation from a first-point estimate to a low-angle least-squares slope fit.

### Publishing
- Prepared the expanded package for first public technical release.
- Added citation metadata.

## [1.0.0] - 2026-03-15

### Added
- Core `roll_period_exact()` function using complete elliptic integral K(m)
- GM recovery functions: `recover_gm_rt4()`, `recover_gm_small_angle()`
- `gm_correction_factor()` for quick amplitude-dependent correction
- Wall-sided GZ model support via `gz_wall_sided()`
- Numerical period computation for arbitrary GZ curves via `roll_period_gz_numerical()`
- Analysis tables: `period_vs_amplitude_table()`, `gm_overestimate_table()`
- Vessel helpers: `T0_from_vessel()`, `C_from_k_factor()`, `C_LOOKUP`
- 69 unit tests matching S4 validation benchmarks
- Interactive web demo (examples/web_demo.html)
- Jupyter notebook demo (examples/demo.ipynb)
