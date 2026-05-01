"""
tests/test_core.py
================
Unit tests for rt4_roll_period.core matching S4 validation benchmarks.

Run with:
    python3 tests/test_core.py

S4 benchmarks (from validation run):
  - K(m) formula max error: 211.059 ppm vs numerical ODE (linear GZ)
  - GM recovery: RT4 mean error < 1e-6 m (vs SA mean ~20.7 mm)
  - GM overestimation at 20 deg: ~1.54% (small-angle)
  - RT4 improvement over small-angle: >20 billion times
  - Wall-sided max period error: 21.5% (formula is approximate)
"""

import math
import sys
import traceback
import numpy as np
from scipy import special

# Allow running from repo root: python3 tests/test_core.py


from rt4_roll_period.core import (
    roll_period_exact,
    roll_period_small_angle,
    gm_correction_factor,
    recover_gm_rt4,
    recover_gm_small_angle,
    recover_gm_wall_sided,
    wall_sided_shape_factor,
    wall_sided_period_ratio,
    wall_sided_gm_correction_factor,
    C_from_k_factor,
    T0_from_vessel,
    gz_linear,
    gz_wall_sided,
    roll_period_gz_numerical,
    gz_table_gm,
    gz_table_interpolator,
    assess_gz_table_quality,
    build_roll_period_report,
    roll_period_gz_table_ratio,
    roll_period_gz_table,
    C_LOOKUP,
)

_G = 9.81
_PUBLIC_GZ_ANGLE = np.array(
    [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    dtype=float,
)
_PUBLIC_GZ = np.array(
    [0.0, 0.025, 0.055, 0.085, 0.110, 0.125, 0.130, 0.115, 0.080, 0.030, 0.0],
    dtype=float,
)


def _check(name: str, condition: bool, msg: str = ""):
    assert condition, f"{name}: {msg}" 


# ---------------------------------------------------------------------------
# Test 1: K(m) formula accuracy vs numerical ODE (pure pendulum)
# ---------------------------------------------------------------------------

def test_period_accuracy_linear_gz():
    """
    For GZ = GM*sin(phi), the formula T = T0*(2/pi)*K(m) is exact.
    S4 benchmark: max error < 300 ppm across 5-80 deg.
    """
    print("\n[1] Period formula accuracy (linear GZ, pure pendulum)")

    def gz_sin(phi_deg):
        return 1.0 * math.sin(math.radians(phi_deg))

    T0 = 20.0
    GM_ref = 1.0
    max_ppm = 0.0

    for phi_deg in range(5, 85, 5):
        T_exact = roll_period_exact(phi_deg, T0)
        T_num = roll_period_gz_numerical(gz_sin, phi_deg, T0, GM_ref)
        ppm = abs(T_exact - T_num) / T_num * 1e6
        if ppm > max_ppm:
            max_ppm = ppm

    _check(
        "max_period_error_linear_gz_ppm < 300",
        max_ppm < 300.0,
        f"max_ppm={max_ppm:.4f}"
    )
    _check(
        "max_period_error_linear_gz_ppm matches S4 (~211 ppm)",
        max_ppm < 250.0,
        f"max_ppm={max_ppm:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 2: Small-angle period = T0
# ---------------------------------------------------------------------------

def test_small_angle_period():
    print("\n[2] Small-angle period = T0")
    T0 = 15.7
    T_sa = roll_period_small_angle(T0)
    _check("small_angle_period_equals_T0", abs(T_sa - T0) < 1e-12)


# ---------------------------------------------------------------------------
# Test 3: GM correction factor
# ---------------------------------------------------------------------------

def test_gm_correction_factor():
    print("\n[3] GM correction factor")
    # At phi=0, correction = 1.0 (K(0) = pi/2)
    cf0 = gm_correction_factor(0.001)
    _check("correction_factor_near_zero_is_one", abs(cf0 - 1.0) < 1e-4)

    # At 20 deg, small-angle over-estimates GM by ~1.54%
    cf20 = gm_correction_factor(20.0)
    overest_pct = (1.0 / cf20 - 1.0) * 100.0
    _check(
        "GM_overestimate_at_20deg_between_1_and_2_pct",
        1.0 < overest_pct < 2.0,
        f"overest={overest_pct:.4f}%"
    )
    _check(
        "GM_overestimate_at_20deg_matches_S4_1p54pct",
        abs(overest_pct - 1.54) < 0.1,
        f"overest={overest_pct:.4f}%"
    )

    # Correction factor is always <= 1
    for phi_deg in [5, 10, 15, 20, 25, 30, 40]:
        cf = gm_correction_factor(phi_deg)
        _check(f"correction_factor_leq_1_at_{phi_deg}deg", cf <= 1.0 + 1e-12)


# ---------------------------------------------------------------------------
# Test 4: GM recovery — round-trip accuracy
# ---------------------------------------------------------------------------

def test_gm_recovery_round_trip():
    """
    Generate 120 cases matching S4 test suite.
    RT4 mean GM error must be < 1e-6 m (sub-micrometre).
    SA mean GM error must be > 0.01 m (10 mm, as in S4).
    """
    print("\n[4] GM recovery round-trip (120 cases, S4 benchmark)")

    rng = np.random.default_rng(7)  # same seed as S4 validation

    vtypes = list(C_LOOKUP.keys())
    cases = []

    for GM in [0.20, 0.50, 1.00, 2.00]:
        for B in [20.0, 30.0, 42.0]:
            for phi_deg in [5, 10, 15, 20, 25, 30, 35]:
                k = float(rng.uniform(0.33, 0.42))
                vt = str(rng.choice(vtypes))
                BM = float(rng.uniform(1.5, 6.0))
                c3 = float(rng.uniform(0.0, GM * 0.3))
                cases.append((GM, B, phi_deg, k, vt, BM, c3))

    while len(cases) < 120:
        GM = float(rng.uniform(0.15, 3.0))
        B = float(rng.uniform(12.0, 55.0))
        phi_deg = float(rng.uniform(3.0, 35.0))
        k = float(rng.uniform(0.30, 0.45))
        vt = str(rng.choice(vtypes))
        BM = float(rng.uniform(1.0, 8.0))
        c3 = float(rng.uniform(0.0, GM * 0.5))
        cases.append((GM, B, phi_deg, k, vt, BM, c3))

    cases = cases[:120]
    rt4_errors = []
    sa_errors = []

    for GM, B, phi_deg, k_factor, vtype, BM, c3 in cases:
        C_phys = C_from_k_factor(k_factor)
        T0_true = C_phys * B / math.sqrt(GM)

        # True observed period (linear GZ, exact formula)
        T_obs = roll_period_exact(phi_deg, T0_true)

        # Recover GM
        GM_rt4 = recover_gm_rt4(T_obs, phi_deg, C_phys, B)
        GM_sa = recover_gm_small_angle(T_obs, C_phys, B)

        rt4_errors.append(abs(GM_rt4 - GM))
        sa_errors.append(abs(GM_sa - GM))

    mean_rt4_m = float(np.mean(rt4_errors))
    mean_sa_m = float(np.mean(sa_errors))
    improvement = mean_sa_m / (mean_rt4_m + 1e-30)

    _check(
        "RT4_mean_GM_error_sub_micrometre",
        mean_rt4_m < 1e-6,
        f"mean_rt4={mean_rt4_m:.2e} m"
    )
    _check(
        "SA_mean_GM_error_gt_10mm",
        mean_sa_m > 0.01,
        f"mean_sa={mean_sa_m*1000:.2f} mm"
    )
    _check(
        "RT4_improvement_over_SA_gt_1e9",
        improvement > 1e9,
        f"improvement={improvement:.2e}x"
    )
    _check(
        "RT4_improvement_matches_S4_~20billion",
        improvement > 1e10,
        f"improvement={improvement:.2e}x"
    )



# ---------------------------------------------------------------------------
# Test 5: Wall-sided GZ — formula is approximate
# ---------------------------------------------------------------------------

def test_wall_sided_gz():
    """
    For wall-sided hull, formula is approximate.
    S4: max wall-sided period error reaches 21.5% at large BM/GM, high amplitude.
    """
    print("\n[5] Wall-sided GZ — formula is approximate")

    max_err = 0.0
    for phi_deg in [10, 15, 20, 25, 30]:
        phi_rad = math.radians(phi_deg)
        for BM_GM in [0.5, 1.0, 2.0, 4.0]:
            GM_ws = 1.0
            BM_ws = BM_GM * GM_ws
            T0_ws = 20.0

            def gz_ws(p, _GM=GM_ws, _BM=BM_ws):
                return gz_wall_sided(p, _GM, _BM)

            T_num = roll_period_gz_numerical(gz_ws, phi_deg, T0_ws, GM_ws)
            T_K = roll_period_exact(phi_deg, T0_ws)
            err_pct = abs(T_K - T_num) / T_num * 100.0
            if err_pct > max_err:
                max_err = err_pct

    # Formula is approximate for wall-sided; S4 found max ~21.5%
    _check(
        "wall_sided_error_is_nonzero_confirming_approximation",
        max_err > 1.0,
        f"max_err={max_err:.3f}%"
    )
    _check(
        "wall_sided_max_error_lt_30pct",
        max_err < 30.0,
        f"max_err={max_err:.3f}%"
    )
    _check(
        "wall_sided_max_error_matches_S4_~21pct",
        15.0 < max_err < 25.0,
        f"max_err={max_err:.3f}%"
    )


# ---------------------------------------------------------------------------
# Test 6: GZ curve models
# ---------------------------------------------------------------------------

def test_gz_models():
    print("\n[6] GZ curve models")

    # Linear GZ at 0 deg = 0
    _check("gz_linear_zero_at_zero", abs(gz_linear(0.0, 1.0)) < 1e-12)

    # Linear GZ at 90 deg = GM
    _check(
        "gz_linear_at_90deg_equals_GM",
        abs(gz_linear(90.0, 1.5) - 1.5) < 1e-10
    )

    # Wall-sided GZ at 0 deg = 0
    _check("gz_wall_sided_zero_at_zero", abs(gz_wall_sided(0.0, 1.0, 3.0)) < 1e-12)

    # Wall-sided GZ >= linear GZ for BM > 0 (wall-sided adds positive BM term)
    for phi_deg in [5, 10, 20, 30]:
        gzl = gz_linear(phi_deg, 1.0)
        gzw = gz_wall_sided(phi_deg, 1.0, 2.0)
        _check(
            f"wall_sided_gz_ge_linear_at_{phi_deg}deg",
            gzw >= gzl - 1e-12
        )


# ---------------------------------------------------------------------------
# Test 7: Vessel helpers
# ---------------------------------------------------------------------------

def test_vessel_helpers():
    print("\n[7] Vessel helper functions")

    # C_from_k_factor: C = 2*pi*k/sqrt(g)
    k = 0.37
    C = C_from_k_factor(k)
    C_expected = 2.0 * math.pi * k / math.sqrt(9.81)
    _check("C_from_k_factor_correct", abs(C - C_expected) < 1e-12)

    # T0_from_vessel: T0 = C*B/sqrt(GM)
    T0 = T0_from_vessel(GM=0.5, B=28.0, k_factor=0.37)
    T0_expected = C * 28.0 / math.sqrt(0.5)
    _check("T0_from_vessel_correct", abs(T0 - T0_expected) < 1e-10)

    # C_LOOKUP has all expected vessel types
    for vtype in ["ropax", "container", "tanker", "bulk", "general", "yacht"]:
        _check(f"C_LOOKUP_has_{vtype}", vtype in C_LOOKUP)


# ---------------------------------------------------------------------------
# Test 8: Period increases monotonically with amplitude
# ---------------------------------------------------------------------------

def test_period_monotone():
    print("\n[8] Period increases monotonically with amplitude")
    T0 = 12.0
    prev_T = T0
    for phi_deg in range(1, 60, 2):
        T = roll_period_exact(phi_deg, T0)
        _check(f"T_monotone_at_{phi_deg}deg", T >= prev_T - 1e-12)
        prev_T = T


# ---------------------------------------------------------------------------
# Test 9: Edge cases
# ---------------------------------------------------------------------------

def test_edge_cases():
    print("\n[9] Edge cases")

    # Very small amplitude: period ~= T0
    T = roll_period_exact(0.01, 10.0)
    _check("period_near_T0_at_tiny_amplitude", abs(T / 10.0 - 1.0) < 1e-4)

    # Large amplitude (60 deg): period is noticeably larger than T0
    T60 = roll_period_exact(60.0, 10.0)
    _check("period_gt_T0_at_60deg", T60 > 10.0)
    _check("period_lt_2_T0_at_60deg", T60 < 20.0)

    # GM recovery at small angle: RT4 and SA should agree
    C, B = 0.80, 25.0
    T0 = 14.0
    T_obs_small = roll_period_exact(2.0, T0)
    GM_rt4 = recover_gm_rt4(T_obs_small, 2.0, C, B)
    GM_sa = recover_gm_small_angle(T_obs_small, C, B)
    _check("rt4_sa_agree_at_small_amplitude", abs(GM_rt4 - GM_sa) / GM_sa < 0.001)


# ---------------------------------------------------------------------------
# Test 10: Wall-sided interpolation and GM recovery
# ---------------------------------------------------------------------------

def test_wall_sided_interpolation_and_recovery():
    print("\n[10] Wall-sided interpolation and GM recovery")

    # Primary-range worst corner from documented validation grid.
    H_30_4 = wall_sided_shape_factor(30.0, 4.0)
    _check(
        "wall_sided_shape_factor_matches_validation_corner",
        abs(H_30_4 - 0.823050090397) < 1e-10,
        f"H_30_4={H_30_4:.12f}",
    )

    # At BM/GM=0, the wall-sided correction reduces to linear K(m).
    for phi_deg in [0.0, 10.0, 20.0, 30.0]:
        ratio_wall = wall_sided_period_ratio(phi_deg, 0.0)
        ratio_linear = roll_period_exact(phi_deg, 1.0)
        _check(
            f"wall_sided_ratio_matches_linear_at_rho0_{phi_deg}",
            abs(ratio_wall - ratio_linear) < 5e-4,
            f"wall={ratio_wall:.12f}, linear={ratio_linear:.12f}",
        )

    # Midpoint cases validate interpolation against fresh numerical integration.
    T0 = 20.0
    GM = 1.0
    max_gm_factor_err_pct = 0.0
    for phi_deg, bm_gm in [(3.5, 0.125), (16.5, 2.25), (29.0, 3.75)]:
        BM = bm_gm * GM

        def gz_ws(p, _GM=GM, _BM=BM):
            return gz_wall_sided(p, _GM, _BM)

        T_num = roll_period_gz_numerical(gz_ws, phi_deg, T0, GM)
        ratio_hat = wall_sided_period_ratio(phi_deg, bm_gm)
        period_err_pct = abs(ratio_hat / (T_num / T0) - 1.0) * 100.0
        gm_factor_err_pct = abs(((T_num / T0) / ratio_hat) ** 2 - 1.0) * 100.0
        max_gm_factor_err_pct = max(max_gm_factor_err_pct, gm_factor_err_pct)

        _check(
            f"wall_sided_midpoint_period_error_lt_0p1pct_{phi_deg}_{bm_gm}",
            period_err_pct < 0.1,
            f"period_err={period_err_pct:.4f}%",
        )

    _check(
        "wall_sided_midpoint_gm_factor_error_lt_0p25pct",
        max_gm_factor_err_pct < 0.25,
        f"max_gm_factor_err={max_gm_factor_err_pct:.4f}%",
    )

    # Inverse recovery should solve GM when the observation is generated by
    # the same validated wall-sided correction.
    C, B = 0.80, 30.0
    GM_true, BM = 1.0, 3.0
    phi_deg = 25.0
    T0_true = C * B / math.sqrt(GM_true)
    T_obs = T0_true * wall_sided_period_ratio(phi_deg, BM / GM_true)
    GM_rec = recover_gm_wall_sided(T_obs, phi_deg, C, B, BM)
    _check(
        "recover_gm_wall_sided_round_trip",
        abs(GM_rec - GM_true) < 1e-10,
        f"GM_rec={GM_rec:.12f}",
    )

    # The validated interpolation envelope is intentionally enforced.
    try:
        wall_sided_period_ratio(35.0, 2.0)
    except ValueError:
        pass
    else:
        raise AssertionError("wall_sided_period_ratio should reject phi > 30 deg")


# ---------------------------------------------------------------------------
# Test 11: Arbitrary GZ table workflow
# ---------------------------------------------------------------------------

def test_arbitrary_gz_table_workflow():
    print("\n[11] Arbitrary GZ table workflow")

    # Linear-GZ table should reproduce the exact K(m) period ratio.
    GM = 1.25
    angle = np.array([0, 5, 10, 15, 20, 25, 30], dtype=float)
    gz = np.array([gz_linear(a, GM) for a in angle], dtype=float)

    gm_est = gz_table_gm(angle, gz)
    _check(
        "gz_table_gm_matches_linear_input",
        abs(gm_est - GM) / GM < 0.002,
        f"gm_est={gm_est:.6f}",
    )

    ratio_table = roll_period_gz_table_ratio(angle, gz, 20.0)
    ratio_exact = roll_period_exact(20.0, 1.0)
    _check(
        "gz_table_ratio_matches_linear_K",
        abs(ratio_table - ratio_exact) / ratio_exact < 0.002,
        f"table={ratio_table:.8f}, exact={ratio_exact:.8f}",
    )

    T = roll_period_gz_table(angle, gz, 20.0, T0_ref=14.0)
    _check(
        "gz_table_absolute_period_scales_with_T0",
        abs(T / 14.0 - ratio_table) < 1e-12,
        f"T={T:.8f}, ratio={ratio_table:.8f}",
    )

    # Public-style digitized table fixture with coarse spacing and finite range.
    angle_real, gz_real = _PUBLIC_GZ_ANGLE, _PUBLIC_GZ
    gm_real = gz_table_gm(angle_real, gz_real)
    ratio_real_25 = roll_period_gz_table_ratio(angle_real, gz_real, 25.0)
    interp = gz_table_interpolator(angle_real, gz_real)

    _check(
        "realworld_fixture_initial_gm_reasonable",
        0.25 < gm_real < 0.35,
        f"gm_real={gm_real:.6f}",
    )
    _check(
        "realworld_fixture_period_ratio_reasonable",
        1.00 < ratio_real_25 < 1.15,
        f"ratio_real_25={ratio_real_25:.6f}",
    )
    _check(
        "realworld_fixture_interpolates_positive_midrange",
        float(interp(12.5)) > 0.0,
        f"gz_12p5={float(interp(12.5)):.6f}",
    )

    try:
        roll_period_gz_table_ratio(angle_real, gz_real, 50.0)
    except ValueError:
        pass
    else:
        raise AssertionError("GZ table should reject amplitudes at vanishing stability")

    quality = assess_gz_table_quality(angle_real, gz_real, phi_max_deg=25.0)
    _check(
        "realworld_fixture_quality_is_caution",
        quality["quality"] == "caution",
        f"quality={quality}",
    )
    _check(
        "realworld_fixture_spacing_caution_flagged",
        "moderate_spacing" in quality["flags"],
        f"flags={quality['flags']}",
    )

    near_limit = assess_gz_table_quality(angle_real, gz_real, phi_max_deg=45.0)
    _check(
        "near_vanishing_stability_flagged",
        "near_vanishing_stability" in near_limit["flags"],
        f"flags={near_limit['flags']}",
    )

    coarse = assess_gz_table_quality(angle_real[::2], gz_real[::2], phi_max_deg=25.0)
    _check(
        "coarse_spacing_flagged",
        "coarse_spacing" in coarse["flags"],
        f"flags={coarse['flags']}",
    )

    noisy_digitized = assess_gz_table_quality(
        angle_real,
        gz_real,
        phi_max_deg=25.0,
        digitization_uncertainty_m=0.005,
    )
    _check(
        "moderate_digitization_uncertainty_flagged",
        "moderate_digitization_uncertainty" in noisy_digitized["flags"],
        f"flags={noisy_digitized['flags']}",
    )


# ---------------------------------------------------------------------------
# Test 12: Practitioner report method selection
# ---------------------------------------------------------------------------

def test_practitioner_report_method_selection():
    print("\n[12] Practitioner report method selection")

    C, B = 0.80, 30.0

    # GZ table takes priority when available.
    table_report = build_roll_period_report(
        T_obs=12.0,
        phi_max_deg=25.0,
        C=C,
        B=B,
        angle_deg=_PUBLIC_GZ_ANGLE,
        gz_m=_PUBLIC_GZ,
        digitization_uncertainty_m=0.0025,
    )
    _check(
        "report_selects_arbitrary_gz_table",
        table_report["method"] == "arbitrary_gz_table",
        f"method={table_report['method']}",
    )
    _check(
        "table_report_has_caution_quality",
        table_report["quality"] == "caution",
        f"quality={table_report['quality']}",
    )
    _check(
        "table_report_corrects_gm",
        table_report["results"]["GM_corrected_m"] > 0.0,
        f"results={table_report['results']}",
    )

    # Wall-sided correction is selected when BM is available and valid.
    GM_true, BM = 1.0, 3.0
    phi_deg = 25.0
    T0_true = C * B / math.sqrt(GM_true)
    T_obs_wall = T0_true * wall_sided_period_ratio(phi_deg, BM / GM_true)
    wall_report = build_roll_period_report(
        T_obs=T_obs_wall,
        phi_max_deg=phi_deg,
        C=C,
        B=B,
        BM=BM,
    )
    _check(
        "report_selects_wall_sided",
        wall_report["method"] == "wall_sided",
        f"method={wall_report['method']}, warnings={wall_report['warnings']}",
    )
    _check(
        "wall_report_recovers_gm",
        abs(wall_report["results"]["GM_corrected_m"] - GM_true) < 1e-10,
        f"GM={wall_report['results']['GM_corrected_m']}",
    )

    # Linear fallback is selected when no richer inputs are available.
    linear_report = build_roll_period_report(
        T_obs=roll_period_exact(20.0, C * B),
        phi_max_deg=20.0,
        C=C,
        B=B,
    )
    _check(
        "report_selects_linear_fallback",
        linear_report["method"] == "linear_gz",
        f"method={linear_report['method']}",
    )
    _check(
        "linear_report_warns_about_assumption",
        len(linear_report["warnings"]) > 0,
        f"warnings={linear_report['warnings']}",
    )

    # Out-of-envelope wall-sided cases fall back to linear with a flag.
    fallback_report = build_roll_period_report(
        T_obs=12.0,
        phi_max_deg=35.0,
        C=C,
        B=B,
        BM=3.0,
    )
    _check(
        "out_of_envelope_wall_sided_falls_back_to_linear",
        fallback_report["method"] == "linear_gz",
        f"method={fallback_report['method']}",
    )
    _check(
        "out_of_envelope_flag_present",
        "wall_sided_outside_validated_envelope" in fallback_report["flags"],
        f"flags={fallback_report['flags']}",
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
