"""
rt4_roll_period.core
====================
Exact nonlinear roll period calculator for ship stability assessment
and inclining experiment GM correction.

Formula (Bernoulli/Euler, exact for linear GZ = GM*sin(phi)):
    T = T0 * (2/pi) * K(sin^2(phi_max/2))
where K(m) is the complete elliptic integral of the first kind.

This is exact when the restoring moment is proportional to sin(phi)
(pure pendulum / linear-GZ model). For wall-sided hulls the formula
is approximate but still far superior to the uncorrected small-angle
method.
"""

import math
import csv
import numpy as np
from pathlib import Path
from scipy import special, optimize, interpolate

# Gravitational acceleration (m/s^2)
_G = 9.81


def _trapz(y, x) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))

# Empirical C-factor lookup by vessel type
C_LOOKUP = {
    "ropax":     0.800,
    "container": 0.780,
    "tanker":    0.820,
    "bulk":      0.810,
    "general":   0.797,
    "yacht":     0.750,
}

# Validated wall-sided correction grid.
# H(phi, rho) = T_wall_numeric / T_linear_K, rho = BM / GM.
# Scope: phi_max_deg in [0, 30], BM/GM in [0, 4].
_WALL_SIDED_PHI_GRID = np.array(
    [0.0, 2.0, 5.0, 8.0, 10.0, 12.0, 15.0, 18.0,
     20.0, 22.0, 25.0, 28.0, 30.0],
    dtype=float,
)
_WALL_SIDED_RHO_GRID = np.array(
    [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
     2.0, 2.5, 3.0, 3.5, 4.0],
    dtype=float,
)
_WALL_SIDED_H_GRID = np.array(
    [
        [1.000000000000, 1.000000000000, 1.000000000000, 1.000000000000, 1.000000000000, 1.000000000000, 1.000000000000, 1.000000000000, 1.000000000000, 1.000000000000, 1.000000000000, 1.000000000000, 1.000000000000],
        [1.000000000000, 0.999731365186, 0.999674178078, 0.999617001333, 0.999559834949, 0.999502678923, 0.999445533250, 0.999388397928, 0.999331272954, 0.999217054035, 0.999102876466, 0.998988740221, 0.998874645275],
        [1.000000000000, 0.999430403791, 0.999072228840, 0.998714460349, 0.998357097516, 0.998000139536, 0.997643585610, 0.997287434939, 0.996931686729, 0.996221394518, 0.995512702657, 0.994805604863, 0.994100094889],
        [1.000000000000, 0.998866543586, 0.997946011076, 0.997028162232, 0.996112983421, 0.995200461107, 0.994290581855, 0.993383332328, 0.992478699288, 0.990677230192, 0.988886070570, 0.987105117927, 0.985334271242],
        [1.000000000000, 0.998340297928, 0.996896744132, 0.995459787397, 0.994029375237, 0.992605455772, 0.991187977718, 0.989776890375, 0.988372143624, 0.985581474250, 0.982815579862, 0.980074079431, 0.977356600378],
        [1.000000000000, 0.997689487636, 0.995601537324, 0.993527381971, 0.991466863123, 0.989419824955, 0.987386114218, 0.985365580182, 0.983358074580, 0.979381567625, 0.975455454532, 0.971578632661, 0.967750034046],
        [1.000000000000, 0.996469034794, 0.993179864585, 0.989924897958, 0.986703518087, 0.983515124191, 0.980359130985, 0.977234968155, 0.974142079855, 0.968047972940, 0.962072635037, 0.956212093599, 0.950462565925],
        [1.000000000000, 0.994936867315, 0.990152907858, 0.985441225256, 0.980799931651, 0.976227210061, 0.971721310904, 0.967280548724, 0.962903299119, 0.954333128111, 0.945998917973, 0.937889597163, 0.929994831367],
        [1.000000000000, 0.993729430261, 0.987777802365, 0.981937944405, 0.976206236273, 0.970579225951, 0.965053619328, 0.959626270780, 0.954294174441, 0.943904365703, 0.933862647668, 0.924149248729, 0.914745983142],
        [1.000000000000, 0.992362277445, 0.985099425380, 0.978002857731, 0.971066026268, 0.964282751343, 0.957647194914, 0.951153835986, 0.944797448222, 0.932476033178, 0.920646297361, 0.909275160735, 0.898332669725],
        [1.000000000000, 0.989987375308, 0.980474071114, 0.971245597097, 0.962287353423, 0.953585804588, 0.945128379032, 0.936903380315, 0.928899908283, 0.913517511745, 0.898907429465, 0.885004859121, 0.871752573854],
        [1.000000000000, 0.987182791681, 0.975056002042, 0.963391154535, 0.952158285476, 0.941330179621, 0.930882045857, 0.920791239376, 0.911037022514, 0.892463729386, 0.875027199545, 0.858612241055, 0.843119785836],
        [1.000000000000, 0.985047350676, 0.970962280988, 0.957499505602, 0.944612396381, 0.932259245005, 0.920402598020, 0.909008700560, 0.898047028088, 0.877312091692, 0.858004527259, 0.839962694397, 0.823050090397],
    ],
    dtype=float,
)


# ---------------------------------------------------------------------------
# Core period formula
# ---------------------------------------------------------------------------

def roll_period_exact(phi_max_deg: float, T0: float) -> float:
    """
    Exact roll period for a ship modelled as a nonlinear pendulum.

    Parameters
    ----------
    phi_max_deg : float
        Roll amplitude in degrees (half the peak-to-peak roll).
    T0 : float
        Small-angle (linear) natural roll period in seconds.
        T0 = C * B / sqrt(GM) where C is the Schofield coefficient,
        B is beam, GM is metacentric height.

    Returns
    -------
    float
        Exact roll period T in seconds.

    Notes
    -----
    Formula: T = T0 * (2/pi) * K(m),  m = sin^2(phi_max/2)
    Exact for GZ = GM*sin(phi).  Approximate for wall-sided hulls.
    """
    phi_rad = math.radians(phi_max_deg)
    m = math.sin(phi_rad / 2.0) ** 2
    m = min(m, 1.0 - 1e-12)  # guard against phi_max -> 180 deg
    Km = float(special.ellipk(m))
    return T0 * (2.0 / math.pi) * Km


def roll_period_small_angle(T0: float) -> float:
    """
    Small-angle (linear) roll period estimate.

    Parameters
    ----------
    T0 : float
        Small-angle natural period in seconds.

    Returns
    -------
    float
        Period T = T0 (amplitude-independent approximation).
    """
    return float(T0)


def gm_correction_factor(phi_max_deg: float) -> float:
    """
    Amplitude correction factor: GM_small_angle / GM_true.

    The small-angle formula GM = (C*B/T_obs)^2 plugs the observed
    finite-amplitude period T_obs in place of the small-amplitude
    natural period T0.  Because T_obs > T0 (finite-amplitude pendulum
    period grows with amplitude), the small-angle method *under*-estimates
    GM.  The correction factor is:

        GM_sa / GM_true = (pi / (2 * K(m)))^2,  m = sin^2(phi_max/2)

    Parameters
    ----------
    phi_max_deg : float
        Roll amplitude in degrees.

    Returns
    -------
    float
        Multiplicative factor (<= 1 for phi_max > 0).
        GM_true = GM_small_angle / factor (i.e. divide, not multiply,
        to recover the true GM from the small-angle estimate).
    """
    phi_rad = math.radians(phi_max_deg)
    m = math.sin(phi_rad / 2.0) ** 2
    m = min(m, 1.0 - 1e-12)
    Km = float(special.ellipk(m))
    return (math.pi / (2.0 * Km)) ** 2


def _linear_period_ratio(phi_max_deg: float) -> float:
    phi_rad = math.radians(phi_max_deg)
    m = math.sin(phi_rad / 2.0) ** 2
    m = min(m, 1.0 - 1e-12)
    return (2.0 / math.pi) * float(special.ellipk(m))


def _interp2_bilinear(
    x: float,
    y: float,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    values: np.ndarray,
) -> float:
    if x < x_grid[0] or x > x_grid[-1] or y < y_grid[0] or y > y_grid[-1]:
        raise ValueError(
            f"Interpolation point out of range: x={x}, y={y}. "
            f"Valid x=[{x_grid[0]}, {x_grid[-1]}], "
            f"valid y=[{y_grid[0]}, {y_grid[-1]}]."
        )

    i = int(np.searchsorted(x_grid, x, side="right") - 1)
    j = int(np.searchsorted(y_grid, y, side="right") - 1)
    i = min(max(i, 0), len(x_grid) - 2)
    j = min(max(j, 0), len(y_grid) - 2)

    x0, x1 = float(x_grid[i]), float(x_grid[i + 1])
    y0, y1 = float(y_grid[j]), float(y_grid[j + 1])
    q00 = float(values[i, j])
    q01 = float(values[i, j + 1])
    q10 = float(values[i + 1, j])
    q11 = float(values[i + 1, j + 1])

    tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
    return (
        q00 * (1.0 - tx) * (1.0 - ty)
        + q10 * tx * (1.0 - ty)
        + q01 * (1.0 - tx) * ty
        + q11 * tx * ty
    )


def wall_sided_shape_factor(phi_max_deg: float, bm_gm: float) -> float:
    """
    Interpolated wall-sided shape factor H(phi, BM/GM).

    H = T_wall_sided / T_linear_K.  This factor was fitted from numerical
    wall-sided integration and validated for the primary product envelope:

        0 <= phi_max_deg <= 30
        0 <= BM/GM <= 4

    Parameters
    ----------
    phi_max_deg : float
        Roll amplitude in degrees.
    bm_gm : float
        Ratio BM/GM.

    Returns
    -------
    float
        Multiplicative period factor relative to the linear K(m) period.
    """
    return _interp2_bilinear(
        float(phi_max_deg),
        float(bm_gm),
        _WALL_SIDED_PHI_GRID,
        _WALL_SIDED_RHO_GRID,
        _WALL_SIDED_H_GRID,
    )


def wall_sided_period_ratio(phi_max_deg: float, bm_gm: float) -> float:
    """
    Validated wall-sided period ratio T_wall/T0.

    This combines the exact linear-GZ K(m) period ratio with the
    interpolated wall-sided shape factor.
    """
    return _linear_period_ratio(phi_max_deg) * wall_sided_shape_factor(
        phi_max_deg, bm_gm
    )


def wall_sided_gm_correction_factor(phi_max_deg: float, bm_gm: float) -> float:
    """
    Wall-sided GM correction factor relative to small-angle GM.

    Returns GM_wall / GM_small_angle for the validated wall-sided envelope.
    """
    ratio = wall_sided_period_ratio(phi_max_deg, bm_gm)
    return 1.0 / (ratio * ratio)


# ---------------------------------------------------------------------------
# GM recovery from observed period
# ---------------------------------------------------------------------------

def recover_gm_rt4(T_obs: float, phi_max_deg: float,
                   C: float, B: float) -> float:
    """
    Recover GM from an observed roll period using the exact K(m) correction.

    Procedure:
      1. Back-calculate T0 = T_obs * pi / (2*K(m))
      2. Apply T0 = C*B/sqrt(GM)  =>  GM = (C*B/T0)^2

    Parameters
    ----------
    T_obs : float
        Observed roll period during inclining experiment (seconds).
    phi_max_deg : float
        Measured roll amplitude in degrees.
    C : float
        Schofield roll period coefficient for the vessel type.
        Typical range 0.73-0.82.  See C_from_k_factor() or C_LOOKUP.
    B : float
        Ship breadth (beam) in metres.

    Returns
    -------
    float
        Corrected GM estimate in metres.
    """
    phi_rad = math.radians(phi_max_deg)
    m = math.sin(phi_rad / 2.0) ** 2
    m = min(m, 1.0 - 1e-12)
    Km = float(special.ellipk(m))
    T0_recovered = T_obs * math.pi / (2.0 * Km)
    return (C * B / T0_recovered) ** 2


def recover_gm_small_angle(T_obs: float, C: float, B: float) -> float:
    """
    Recover GM from an observed roll period using the small-angle formula.

    GM = (C*B/T_obs)^2  (no amplitude correction)

    Parameters
    ----------
    T_obs : float
        Observed roll period in seconds.
    C : float
        Schofield coefficient.
    B : float
        Beam in metres.

    Returns
    -------
    float
        GM estimate in metres (biased low when roll amplitude is large;
        the true GM is larger because T_obs > T0 for finite amplitude).
    """
    return (C * B / T_obs) ** 2


def recover_gm_wall_sided(T_obs: float, phi_max_deg: float,
                          C: float, B: float, BM: float) -> float:
    """
    Recover GM from an observed period using the wall-sided correction.

    The correction depends on rho = BM/GM, so GM is solved iteratively:

        T_obs = (C*B/sqrt(GM)) * F_linear(phi) * H(phi, BM/GM)

    This implementation is validated only for:

        0 <= phi_max_deg <= 30
        0 <= BM/GM <= 4

    Parameters
    ----------
    T_obs : float
        Observed roll period in seconds.
    phi_max_deg : float
        Roll amplitude in degrees.
    C : float
        Schofield roll period coefficient.
    B : float
        Ship breadth (beam) in metres.
    BM : float
        Transverse metacentric radius in metres.

    Returns
    -------
    float
        Wall-sided corrected GM estimate in metres.
    """
    if T_obs <= 0.0:
        raise ValueError("T_obs must be positive.")
    if C <= 0.0:
        raise ValueError("C must be positive.")
    if B <= 0.0:
        raise ValueError("B must be positive.")
    if BM < 0.0:
        raise ValueError("BM must be non-negative.")
    if phi_max_deg < _WALL_SIDED_PHI_GRID[0] or phi_max_deg > _WALL_SIDED_PHI_GRID[-1]:
        raise ValueError(
            f"phi_max_deg={phi_max_deg} is outside the validated "
            f"wall-sided range [{_WALL_SIDED_PHI_GRID[0]}, "
            f"{_WALL_SIDED_PHI_GRID[-1]}] degrees."
        )

    if BM == 0.0:
        return recover_gm_rt4(T_obs, phi_max_deg, C, B)

    rho_max = float(_WALL_SIDED_RHO_GRID[-1])
    gm_min = BM / rho_max

    def predicted_period(GM: float) -> float:
        rho = BM / GM
        ratio = wall_sided_period_ratio(phi_max_deg, rho)
        return (C * B / math.sqrt(GM)) * ratio

    def residual(GM: float) -> float:
        return predicted_period(GM) - T_obs

    low = gm_min
    f_low = residual(low)
    if f_low < 0.0:
        raise ValueError(
            "Observed period implies BM/GM outside the validated wall-sided "
            f"range. Minimum valid GM is {gm_min:.6g} m for BM={BM:.6g} m."
        )

    high = max(recover_gm_rt4(T_obs, phi_max_deg, C, B), low * 2.0, 1e-9)
    f_high = residual(high)
    for _ in range(80):
        if f_high <= 0.0:
            break
        high *= 2.0
        f_high = residual(high)
    else:
        raise ValueError("Could not bracket wall-sided GM solution.")

    return float(optimize.brentq(residual, low, high, xtol=1e-12, rtol=1e-12))


def recover_gm_cfactor(T_obs: float, vtype: str, B: float) -> float:
    """
    Recover GM using the empirical C-factor table method (no amplitude correction).

    Parameters
    ----------
    T_obs : float
        Observed roll period in seconds.
    vtype : str
        Vessel type string — key in C_LOOKUP (e.g. 'tanker', 'ropax').
    B : float
        Beam in metres.

    Returns
    -------
    float
        GM estimate in metres.
    """
    C_table = C_LOOKUP.get(vtype, 0.797)
    return (C_table * B / T_obs) ** 2


# ---------------------------------------------------------------------------
# Vessel parameter helpers
# ---------------------------------------------------------------------------

def C_from_k_factor(k_factor: float) -> float:
    """
    Compute physical Schofield coefficient from the gyration radius ratio k.

    C = 2*pi*k / sqrt(g)

    Parameters
    ----------
    k_factor : float
        Ratio of gyration radius to beam (dimensionless, typically 0.33-0.42).

    Returns
    -------
    float
        Schofield coefficient C (s/m^0.5 per metre of beam).
    """
    return 2.0 * math.pi * k_factor / math.sqrt(_G)


def T0_from_vessel(GM: float, B: float, k_factor: float) -> float:
    """
    Compute the small-angle natural roll period T0.

    T0 = C * B / sqrt(GM),  C = 2*pi*k/sqrt(g)

    Parameters
    ----------
    GM : float
        Metacentric height in metres.
    B : float
        Beam in metres.
    k_factor : float
        Gyration radius ratio (dimensionless).

    Returns
    -------
    float
        Small-angle roll period in seconds.
    """
    C = C_from_k_factor(k_factor)
    return C * B / math.sqrt(GM)


# ---------------------------------------------------------------------------
# GZ curve models
# ---------------------------------------------------------------------------

def gz_linear(phi_deg: float, GM: float) -> float:
    """
    Linear GZ curve: GZ = GM * sin(phi).

    Valid for small angles; produces pure-pendulum dynamics.

    Parameters
    ----------
    phi_deg : float
        Heel angle in degrees.
    GM : float
        Metacentric height in metres.

    Returns
    -------
    float
        Righting lever GZ in metres.
    """
    return GM * math.sin(math.radians(phi_deg))


def gz_wall_sided(phi_deg: float, GM: float, BM: float) -> float:
    """
    Wall-sided hull GZ curve.

    GZ = sin(phi) * (GM + BM/2 * tan^2(phi))

    The wall-sided formula accounts for the change in the metacentric
    radius BM with heel angle for a wall-sided hull form.

    Parameters
    ----------
    phi_deg : float
        Heel angle in degrees.
    GM : float
        Metacentric height at small angles in metres.
    BM : float
        Transverse metacentric radius at small angles in metres.

    Returns
    -------
    float
        Righting lever GZ in metres.
    """
    phi_rad = math.radians(phi_deg)
    return math.sin(phi_rad) * (GM + BM / 2.0 * math.tan(phi_rad) ** 2)


# ---------------------------------------------------------------------------
# Numerical period (arbitrary GZ curve, ground truth)
# ---------------------------------------------------------------------------

def roll_period_gz_numerical(gz_func, phi_max_deg: float,
                              T0_ref: float, GM: float,
                              n_pts: int = 2000) -> float:
    """
    Exact roll period for an arbitrary GZ curve by energy-conservation
    quadrature.

    Equation of motion (undamped):
        I * phi_ddot + Delta * GZ(phi) = 0

    Quarter period from energy integral:
        T/4 = sqrt(kappa/g / 2) * integral_0^{phi_max} dphi / sqrt(U(phi_max)-U(phi))
    where U(phi) = integral_0^phi GZ(x) dx and kappa = T0^2*g*GM/(4*pi^2).

    Parameters
    ----------
    gz_func : callable
        gz_func(phi_deg) -> float, righting lever in metres.
    phi_max_deg : float
        Roll amplitude in degrees.
    T0_ref : float
        Small-angle period in seconds (provides the length scale).
    GM : float
        Small-angle GM in metres (needed for kappa normalisation).
    n_pts : int
        Number of quadrature points (default 2000).

    Returns
    -------
    float
        Roll period in seconds.
    """
    phi_max_rad = math.radians(phi_max_deg)
    phi_arr = np.linspace(0.0, phi_max_rad, n_pts + 1)
    gz_arr = np.array([gz_func(math.degrees(p)) for p in phi_arr])
    U_arr = np.zeros(n_pts + 1)
    U_arr[1:] = np.cumsum(
        0.5 * (gz_arr[:-1] + gz_arr[1:]) * np.diff(phi_arr)
    )
    U_max = U_arr[-1]

    split = int(n_pts * 0.98)
    phi_inner = phi_arr[:split+1]
    U_inner = U_arr[:split+1]
    integrand_inner = np.zeros(split + 1)
    for i in range(split + 1):
        dU = U_max - U_inner[i]
        integrand_inner[i] = 1.0 / math.sqrt(dU) if dU > 1e-15 else 0.0
    I_inner = _trapz(integrand_inner, phi_inner)

    gz_max = gz_func(phi_max_deg)
    if gz_max > 0:
        phi_tail_start = phi_arr[split]
        delta_phi = phi_max_rad - phi_tail_start
        I_tail = 2.0 * math.sqrt(delta_phi / gz_max)
    else:
        I_tail = 0.0

    kappa_over_g = (T0_ref ** 2 * GM) / (4.0 * math.pi ** 2)
    T_quarter = math.sqrt(kappa_over_g / 2.0) * (I_inner + I_tail)
    return 4.0 * T_quarter


def _validated_gz_table(angle_deg, gz_m) -> tuple[np.ndarray, np.ndarray]:
    angle_arr = np.asarray(angle_deg, dtype=float)
    gz_arr = np.asarray(gz_m, dtype=float)

    if angle_arr.ndim != 1 or gz_arr.ndim != 1:
        raise ValueError("angle_deg and gz_m must be one-dimensional arrays.")
    if len(angle_arr) != len(gz_arr):
        raise ValueError("angle_deg and gz_m must have the same length.")
    if len(angle_arr) < 3:
        raise ValueError("At least three GZ table points are required.")
    if not np.all(np.isfinite(angle_arr)) or not np.all(np.isfinite(gz_arr)):
        raise ValueError("angle_deg and gz_m must contain only finite values.")
    if np.any(np.diff(angle_arr) <= 0.0):
        raise ValueError("angle_deg values must be strictly increasing.")
    if angle_arr[0] > 0.0:
        raise ValueError("GZ table must include 0 degrees or a negative heel angle.")
    if angle_arr[-1] <= 0.0:
        raise ValueError("GZ table must extend to positive heel angles.")

    if angle_arr[0] < 0.0:
        zero_gz = float(np.interp(0.0, angle_arr, gz_arr))
        positive = angle_arr > 0.0
        angle_arr = np.concatenate(([0.0], angle_arr[positive]))
        gz_arr = np.concatenate(([zero_gz], gz_arr[positive]))

    if abs(gz_arr[0]) > 1e-8:
        raise ValueError("GZ at 0 degrees must be approximately zero.")

    return angle_arr, gz_arr


def _estimate_gm_from_gz_table(angle_arr: np.ndarray, gz_arr: np.ndarray) -> float:
    positive = (angle_arr > 0.0) & (gz_arr > 0.0)
    if not np.any(positive):
        raise ValueError("GZ table must contain positive righting levers.")

    low_angle = positive & (angle_arr <= 5.0)
    if np.count_nonzero(low_angle) >= 2:
        fit_mask = low_angle
    else:
        positive_idx = np.flatnonzero(positive)
        fit_mask = np.zeros_like(positive, dtype=bool)
        fit_mask[positive_idx[: min(2, len(positive_idx))]] = True

    sin_phi = np.sin(np.radians(angle_arr[fit_mask]))
    gz_fit = gz_arr[fit_mask]
    denom = float(np.dot(sin_phi, sin_phi))
    if denom <= 0.0:
        raise ValueError("Could not estimate GM from the GZ table.")

    gm = float(np.dot(sin_phi, gz_fit) / denom)
    if gm <= 0.0 or not math.isfinite(gm):
        raise ValueError("Estimated GM from the GZ table must be positive.")
    return gm


def gz_table_interpolator(angle_deg, gz_m):
    """
    Build a shape-preserving interpolator for a user-supplied GZ table.

    Parameters
    ----------
    angle_deg : array-like
        Heel angles in degrees. Values must be strictly increasing and include
        0 degrees or a negative-to-positive span containing 0 degrees.
    gz_m : array-like
        Righting lever values in metres. GZ at 0 degrees must be approximately 0.

    Returns
    -------
    callable
        Function `f(phi_deg) -> GZ_m` for scalar or array inputs.
    """
    angle_arr, gz_arr = _validated_gz_table(angle_deg, gz_m)
    return interpolate.PchipInterpolator(angle_arr, gz_arr, extrapolate=False)


def load_gz_table_csv(path, angle_col: str = "angle_deg",
                      gz_col: str = "GZ_m") -> tuple[np.ndarray, np.ndarray]:
    """
    Load a CSV righting-arm table with `angle_deg,GZ_m` columns.

    Parameters
    ----------
    path : str or pathlib.Path
        CSV file path.
    angle_col : str
        Name of the heel-angle column in degrees.
    gz_col : str
        Name of the righting-lever column in metres.

    Returns
    -------
    (angle_deg, gz_m)
        Validated NumPy arrays ready for `roll_period_gz_table_ratio()`.
    """
    angles = []
    gzs = []
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("GZ CSV must contain a header row.")
        if angle_col not in reader.fieldnames or gz_col not in reader.fieldnames:
            raise ValueError(
                f"GZ CSV must contain columns {angle_col!r} and {gz_col!r}."
            )
        for row in reader:
            angles.append(float(row[angle_col]))
            gzs.append(float(row[gz_col]))

    return _validated_gz_table(angles, gzs)


def gz_table_gm(angle_deg, gz_m) -> float:
    """
    Estimate initial GM from the low-angle slope of a GZ table.

    The estimate fits `GZ ~= GM * sin(phi)` through the origin using positive
    points up to 5 degrees when available. If the table is too sparse, it uses
    the first positive points. This keeps the estimate transparent while
    reducing sensitivity to one low-angle digitization point.
    """
    angle_arr, gz_arr = _validated_gz_table(angle_deg, gz_m)
    return _estimate_gm_from_gz_table(angle_arr, gz_arr)


def assess_gz_table_quality(angle_deg, gz_m, phi_max_deg=None,
                            digitization_uncertainty_m=None) -> dict:
    """
    Assess whether a GZ table is suitable for arbitrary-GZ period reports.

    This helper does not replace calculation-time validation. It returns
    reporting guidance from sensitivity checks: table spacing,
    low-angle GM data quality, vanishing-stability margin, and optional
    digitization uncertainty.

    Parameters
    ----------
    angle_deg : array-like
        Heel angle table in degrees.
    gz_m : array-like
        GZ table in metres.
    phi_max_deg : float, optional
        Intended roll amplitude. When provided, the report checks table range,
        local GZ positivity, and margin to the first non-positive GZ point.
    digitization_uncertainty_m : float, optional
        Approximate absolute GZ uncertainty in metres for graph-digitized data.

    Returns
    -------
    dict
        Summary with `ok_for_report`, `quality`, `warnings`, and numeric
        `metrics` fields.
    """
    angle_arr, gz_arr = _validated_gz_table(angle_deg, gz_m)
    warnings = []
    flags = []

    diffs = np.diff(angle_arr)
    positive = (angle_arr > 0.0) & (gz_arr > 0.0)
    low_angle_count = int(np.count_nonzero(positive & (angle_arr <= 5.0)))
    gm_est = _estimate_gm_from_gz_table(angle_arr, gz_arr)

    max_spacing = float(np.max(diffs))
    median_spacing = float(np.median(diffs))
    if max_spacing > 10.0:
        flags.append("very_coarse_spacing")
        warnings.append(
            "GZ table spacing exceeds 10 deg; arbitrary-GZ period estimates "
            "are screening-quality only."
        )
    elif max_spacing > 5.0:
        flags.append("coarse_spacing")
        warnings.append(
            "GZ table spacing exceeds 5 deg; sensitivity checks indicate "
            "material error risk for curved GZ curves."
        )
    elif max_spacing > 2.5:
        flags.append("moderate_spacing")
        warnings.append(
            "GZ table spacing is above the preferred 2.5 deg spacing for "
            "highly curved curves or near-limit amplitudes."
        )

    if low_angle_count < 1:
        flags.append("missing_low_angle_positive_gz")
        warnings.append("No positive GZ point at or below 5 deg is available.")
    elif low_angle_count < 2:
        flags.append("sparse_low_angle_gz")
        warnings.append(
            "Only one positive GZ point at or below 5 deg is available; "
            "initial GM estimate is sensitive to that point."
        )

    positive_angle = angle_arr[positive]
    first_non_positive_after_zero = None
    after_zero = np.flatnonzero(angle_arr > 0.0)
    if len(after_zero) > 0:
        bad_after_zero = after_zero[gz_arr[after_zero] <= 0.0]
        if len(bad_after_zero) > 0:
            first_non_positive_after_zero = float(angle_arr[int(bad_after_zero[0])])

    if phi_max_deg is not None:
        phi = float(phi_max_deg)
        if phi <= 0.0:
            flags.append("invalid_phi_max")
            warnings.append("phi_max_deg must be positive.")
        if phi > float(angle_arr[-1]):
            flags.append("phi_exceeds_table_range")
            warnings.append("Requested amplitude exceeds the GZ table range.")
        if first_non_positive_after_zero is not None:
            margin = first_non_positive_after_zero - phi
            if margin <= 0.0:
                flags.append("amplitude_at_or_beyond_vanishing_stability")
                warnings.append(
                    "Requested amplitude reaches or exceeds the first "
                    "non-positive GZ point."
                )
            elif margin < max(5.0, 0.15 * first_non_positive_after_zero):
                flags.append("near_vanishing_stability")
                warnings.append(
                    "Requested amplitude is close to vanishing stability; "
                    "period estimates are highly sensitive."
                )

        in_range = angle_arr <= min(phi, float(angle_arr[-1]))
        if np.any((angle_arr[in_range] > 0.0) & (gz_arr[in_range] <= 0.0)):
            flags.append("non_positive_gz_within_amplitude")
            warnings.append("GZ is non-positive inside the requested amplitude range.")

    if digitization_uncertainty_m is not None:
        uncertainty = float(digitization_uncertainty_m)
        if uncertainty < 0.0:
            flags.append("invalid_digitization_uncertainty")
            warnings.append("digitization_uncertainty_m must be non-negative.")
        elif uncertainty > 0.005:
            flags.append("high_digitization_uncertainty")
            warnings.append(
                "Digitization uncertainty exceeds 0.005 m; sensitivity checks "
                "shows large period-ratio error risk."
            )
        elif uncertainty > 0.0025:
            flags.append("moderate_digitization_uncertainty")
            warnings.append(
                "Digitization uncertainty exceeds 0.0025 m; treat results as "
                "screening-quality unless independently verified."
            )
        elif uncertainty > 0.001:
            flags.append("low_digitization_uncertainty")
            warnings.append(
                "Digitization uncertainty exceeds 0.001 m; include uncertainty "
                "label in reports."
            )

    severe_flags = {
        "very_coarse_spacing",
        "missing_low_angle_positive_gz",
        "invalid_phi_max",
        "phi_exceeds_table_range",
        "amplitude_at_or_beyond_vanishing_stability",
        "non_positive_gz_within_amplitude",
        "invalid_digitization_uncertainty",
        "high_digitization_uncertainty",
    }
    caution_flags = {
        "coarse_spacing",
        "moderate_spacing",
        "sparse_low_angle_gz",
        "near_vanishing_stability",
        "moderate_digitization_uncertainty",
        "low_digitization_uncertainty",
    }

    if any(flag in severe_flags for flag in flags):
        quality = "screening"
    elif any(flag in caution_flags for flag in flags):
        quality = "caution"
    else:
        quality = "report"

    return {
        "ok_for_report": quality == "report",
        "quality": quality,
        "flags": flags,
        "warnings": warnings,
        "metrics": {
            "point_count": int(len(angle_arr)),
            "max_spacing_deg": max_spacing,
            "median_spacing_deg": median_spacing,
            "low_angle_positive_point_count": low_angle_count,
            "estimated_initial_gm_m": gm_est,
            "max_angle_deg": float(angle_arr[-1]),
            "first_non_positive_gz_angle_deg": first_non_positive_after_zero,
            "max_gz_m": float(np.max(gz_arr)),
            "max_gz_angle_deg": float(angle_arr[int(np.argmax(gz_arr))]),
        },
    }


def roll_period_gz_table_ratio(angle_deg, gz_m, phi_max_deg: float,
                               n_pts: int = 2000) -> float:
    """
    Compute T/T0 for an arbitrary righting-arm table.

    The table is interpolated with PCHIP and integrated with the same
    energy-conservation quadrature used for callable GZ curves. `T0` is the
    small-angle period implied by the table's initial GM and the vessel's roll
    inertia; this function returns the nonlinear amplitude ratio only.

    Parameters
    ----------
    angle_deg : array-like
        Heel angle table in degrees.
    gz_m : array-like
        GZ table in metres.
    phi_max_deg : float
        Roll amplitude in degrees. Must be within the supplied positive-angle
        table and before the curve loses positive righting lever.
    n_pts : int
        Number of quadrature intervals.

    Returns
    -------
    float
        Period ratio `T / T0`.
    """
    angle_arr, gz_arr = _validated_gz_table(angle_deg, gz_m)
    phi_max_deg = float(phi_max_deg)
    if phi_max_deg <= 0.0:
        raise ValueError("phi_max_deg must be positive.")
    if phi_max_deg > float(angle_arr[-1]):
        raise ValueError("phi_max_deg must not exceed the GZ table range.")
    if n_pts < 100:
        raise ValueError("n_pts must be at least 100.")

    gz_interp = gz_table_interpolator(angle_arr, gz_arr)
    gm = _estimate_gm_from_gz_table(angle_arr, gz_arr)
    phi_grid_deg = np.linspace(0.0, phi_max_deg, n_pts + 1)
    gz_grid = np.asarray(gz_interp(phi_grid_deg), dtype=float)

    if np.any(~np.isfinite(gz_grid)):
        raise ValueError("Interpolation failed inside the requested amplitude range.")
    if np.any(gz_grid[1:-1] <= 0.0) or gz_grid[-1] <= 0.0:
        raise ValueError(
            "GZ must remain positive over the requested roll-amplitude range."
        )

    return roll_period_gz_numerical(
        lambda p: float(gz_interp(p)),
        phi_max_deg,
        T0_ref=1.0,
        GM=gm,
        n_pts=n_pts,
    )


def roll_period_gz_table(angle_deg, gz_m, phi_max_deg: float,
                         T0_ref: float, n_pts: int = 2000) -> float:
    """
    Compute the absolute roll period for an arbitrary GZ table.

    Parameters
    ----------
    angle_deg : array-like
        Heel angle table in degrees.
    gz_m : array-like
        GZ table in metres.
    phi_max_deg : float
        Roll amplitude in degrees.
    T0_ref : float
        Small-angle period in seconds for the same vessel/loading condition.
    n_pts : int
        Number of quadrature intervals.

    Returns
    -------
    float
        Roll period in seconds.
    """
    if T0_ref <= 0.0:
        raise ValueError("T0_ref must be positive.")
    return float(T0_ref) * roll_period_gz_table_ratio(
        angle_deg, gz_m, phi_max_deg, n_pts=n_pts
    )


def build_roll_period_report(T_obs: float, phi_max_deg: float,
                             C: float, B: float,
                             BM: float = None,
                             angle_deg=None, gz_m=None,
                             gz_csv_path=None,
                             digitization_uncertainty_m=None,
                             n_pts: int = 2000) -> dict:
    """
    Build a practitioner-facing roll-period correction report.

    Method priority:

    1. Vessel-specific GZ table, when `angle_deg/gz_m` or `gz_csv_path` is supplied.
    2. Validated wall-sided correction, when `BM` is supplied and the case is
       inside the validated envelope.
    3. Linear-GZ K(m) correction fallback.

    Returns a dictionary with the selected method, corrected GM, small-angle GM,
    period ratio, warnings, and any table-quality report.
    """
    if T_obs <= 0.0:
        raise ValueError("T_obs must be positive.")
    if C <= 0.0:
        raise ValueError("C must be positive.")
    if B <= 0.0:
        raise ValueError("B must be positive.")
    if phi_max_deg <= 0.0:
        raise ValueError("phi_max_deg must be positive.")

    warnings = []
    flags = []
    table_quality = None
    method = "linear_gz"
    method_label = "Linear GZ exact K(m) correction"

    gm_small = recover_gm_small_angle(T_obs, C, B)
    ratio = None
    gm_corrected = None
    t0_recovered = None

    if gz_csv_path is not None:
        if angle_deg is not None or gz_m is not None:
            raise ValueError("Provide either gz_csv_path or angle_deg/gz_m, not both.")
        angle_deg, gz_m = load_gz_table_csv(gz_csv_path)

    if angle_deg is not None or gz_m is not None:
        if angle_deg is None or gz_m is None:
            raise ValueError("Both angle_deg and gz_m must be provided.")
        angle_arr, gz_arr = _validated_gz_table(angle_deg, gz_m)
        table_quality = assess_gz_table_quality(
            angle_arr,
            gz_arr,
            phi_max_deg=phi_max_deg,
            digitization_uncertainty_m=digitization_uncertainty_m,
        )
        warnings.extend(table_quality["warnings"])
        flags.extend(table_quality["flags"])
        ratio = roll_period_gz_table_ratio(
            angle_arr, gz_arr, phi_max_deg, n_pts=n_pts
        )
        method = "arbitrary_gz_table"
        method_label = "Vessel-specific arbitrary GZ table"
        t0_recovered = T_obs / ratio
        gm_corrected = (C * B / t0_recovered) ** 2
    elif BM is not None:
        try:
            gm_corrected = recover_gm_wall_sided(T_obs, phi_max_deg, C, B, BM)
            rho = BM / gm_corrected if gm_corrected > 0.0 else math.inf
            ratio = wall_sided_period_ratio(phi_max_deg, rho)
            method = "wall_sided"
            method_label = "Validated wall-sided correction"
            t0_recovered = T_obs / ratio
        except ValueError as exc:
            flags.append("wall_sided_outside_validated_envelope")
            warnings.append(
                "Wall-sided correction could not be applied inside the "
                f"validated envelope: {exc}. Falling back to linear GZ."
            )

    if gm_corrected is None:
        ratio = _linear_period_ratio(phi_max_deg)
        t0_recovered = T_obs / ratio
        gm_corrected = recover_gm_rt4(T_obs, phi_max_deg, C, B)
        method = "linear_gz"
        method_label = "Linear GZ exact K(m) correction"
        if BM is not None:
            flags.append("linear_fallback_with_bm_available")

    gm_factor = gm_corrected / gm_small
    if method == "linear_gz":
        warnings.append(
            "Linear-GZ correction assumes GZ = GM*sin(phi). Use a vessel GZ "
            "table or validated wall-sided correction when available."
        )

    return {
        "method": method,
        "method_label": method_label,
        "inputs": {
            "T_obs": float(T_obs),
            "phi_max_deg": float(phi_max_deg),
            "C": float(C),
            "B": float(B),
            "BM": None if BM is None else float(BM),
            "gz_csv_path": None if gz_csv_path is None else str(gz_csv_path),
            "digitization_uncertainty_m": digitization_uncertainty_m,
        },
        "results": {
            "GM_small_angle_m": float(gm_small),
            "GM_corrected_m": float(gm_corrected),
            "GM_correction_factor": float(gm_factor),
            "period_ratio_T_over_T0": float(ratio),
            "T0_recovered_s": float(t0_recovered),
            "GM_delta_pct_vs_small_angle": float((gm_factor - 1.0) * 100.0),
        },
        "quality": (
            "report" if table_quality is None and method != "linear_gz"
            else "caution" if method == "linear_gz"
            else table_quality["quality"]
        ),
        "flags": flags,
        "warnings": warnings,
        "table_quality": table_quality,
    }


# ---------------------------------------------------------------------------
# Vectorised helpers for plotting / tables
# ---------------------------------------------------------------------------

def period_vs_amplitude_table(T0: float,
                               phi_range_deg=None) -> tuple:
    """
    Compute T_exact and T_small_angle across a range of amplitudes.

    Parameters
    ----------
    T0 : float
        Small-angle period in seconds.
    phi_range_deg : array-like, optional
        Amplitude array in degrees (default 1-45 deg, 45 points).

    Returns
    -------
    (phi_arr, T_exact_arr, T_sa_arr, correction_pct_arr)
        Arrays suitable for plotting.
    """
    if phi_range_deg is None:
        phi_range_deg = np.linspace(1, 45, 45)
    phi_arr = np.asarray(phi_range_deg, dtype=float)
    T_exact_arr = np.array([roll_period_exact(p, T0) for p in phi_arr])
    T_sa_arr = np.full_like(phi_arr, T0)
    # period error relative to small-angle (positive = longer true period)
    correction_pct_arr = (T_exact_arr - T_sa_arr) / T_sa_arr * 100.0
    return phi_arr, T_exact_arr, T_sa_arr, correction_pct_arr


def gm_overestimate_table(phi_range_deg=None) -> tuple:
    """
    Compute the GM bias of the small-angle method, by amplitude.

    Despite the historical name, the small-angle method actually
    *under*-estimates GM (T_obs > T0 implies (C*B/T_obs)^2 < (C*B/T0)^2).
    The returned percentage is the magnitude of that bias:

        bias_pct[i] = (GM_true - GM_sa) / GM_sa * 100   (positive)

    i.e. how much larger the true GM is than the naive small-angle
    estimate.  At 20 deg amplitude this is ~1.54%.

    Note: the function name is preserved for backwards compatibility.
    A correctly-named alias (`gm_amplitude_bias_table`) is planned for
    v1.2.

    Returns
    -------
    (phi_arr, bias_pct_arr)
    """
    if phi_range_deg is None:
        phi_range_deg = np.linspace(1, 45, 45)
    phi_arr = np.asarray(phi_range_deg, dtype=float)
    overest = np.array(
        [(1.0 / gm_correction_factor(p) - 1.0) * 100.0 for p in phi_arr]
    )
    return phi_arr, overest
