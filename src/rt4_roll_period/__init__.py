"""
rt4_roll_period
===============
Exact nonlinear roll period calculator for ship stability assessment
and inclining experiment GM correction.

Quick start
-----------
>>> from rt4_roll_period import roll_period_exact, recover_gm_rt4
>>> T = roll_period_exact(phi_max_deg=20.0, T0=15.0)
>>> GM = recover_gm_rt4(T_obs=T, phi_max_deg=20.0, C=0.797, B=28.0)

Main API
--------
roll_period_exact(phi_max_deg, T0)
    Exact period via K(m) elliptic integral.

roll_period_small_angle(T0)
    Small-angle (uncorrected) period — returns T0.

gm_correction_factor(phi_max_deg)
    Multiplicative factor: GM_small_angle / GM_true.
    Recover linear-GZ truth with GM_true = GM_small_angle / factor.

recover_gm_rt4(T_obs, phi_max_deg, C, B)
    Recover corrected GM from observed period + amplitude.

recover_gm_small_angle(T_obs, C, B)
    Recover GM without amplitude correction (baseline).

T0_from_vessel(GM, B, k_factor)
    Compute small-angle period from vessel parameters.

C_from_k_factor(k_factor)
    Compute Schofield coefficient from gyration radius ratio.

gz_linear(phi_deg, GM)
    Linear GZ = GM * sin(phi).

gz_wall_sided(phi_deg, GM, BM)
    Wall-sided GZ = sin(phi) * (GM + BM/2 * tan^2(phi)).

gz_table_interpolator(angle_deg, gz_m)
    Shape-preserving interpolation for user-supplied GZ tables.

load_gz_table_csv(path)
    Load and validate a CSV table with angle_deg,GZ_m columns.

gz_table_gm(angle_deg, gz_m)
    Estimate initial GM from a GZ table.

assess_gz_table_quality(angle_deg, gz_m, phi_max_deg=None)
    Return reporting-quality flags and warnings for a GZ table.

build_roll_period_report(T_obs, phi_max_deg, C, B, ...)
    Select the best correction method and return a practitioner report.

roll_period_gz_table_ratio(angle_deg, gz_m, phi_max_deg)
    Compute arbitrary-GZ period ratio T/T0 from a table.

roll_period_gz_table(angle_deg, gz_m, phi_max_deg, T0_ref)
    Compute arbitrary-GZ absolute period from a table and known T0.

wall_sided_period_ratio(phi_max_deg, bm_gm)
    Interpolated wall-sided T/T0 correction in the validated envelope.

recover_gm_wall_sided(T_obs, phi_max_deg, C, B, BM)
    Recover GM using the validated wall-sided interpolation + root solve.

C_LOOKUP
    Dict of empirical C-factors by vessel type.
"""

from .core import (
    roll_period_exact,
    roll_period_small_angle,
    gm_correction_factor,
    recover_gm_rt4,
    recover_gm_small_angle,
    recover_gm_wall_sided,
    recover_gm_cfactor,
    wall_sided_shape_factor,
    wall_sided_period_ratio,
    wall_sided_gm_correction_factor,
    C_from_k_factor,
    T0_from_vessel,
    gz_linear,
    gz_wall_sided,
    roll_period_gz_numerical,
    gz_table_interpolator,
    load_gz_table_csv,
    gz_table_gm,
    assess_gz_table_quality,
    build_roll_period_report,
    roll_period_gz_table_ratio,
    roll_period_gz_table,
    period_vs_amplitude_table,
    gm_overestimate_table,
    C_LOOKUP,
)

__version__ = "1.1.2"
__all__ = [
    "roll_period_exact",
    "roll_period_small_angle",
    "gm_correction_factor",
    "recover_gm_rt4",
    "recover_gm_small_angle",
    "recover_gm_wall_sided",
    "recover_gm_cfactor",
    "wall_sided_shape_factor",
    "wall_sided_period_ratio",
    "wall_sided_gm_correction_factor",
    "C_from_k_factor",
    "T0_from_vessel",
    "gz_linear",
    "gz_wall_sided",
    "roll_period_gz_numerical",
    "gz_table_interpolator",
    "load_gz_table_csv",
    "gz_table_gm",
    "assess_gz_table_quality",
    "build_roll_period_report",
    "roll_period_gz_table_ratio",
    "roll_period_gz_table",
    "period_vs_amplitude_table",
    "gm_overestimate_table",
    "C_LOOKUP",
]
