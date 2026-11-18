"""Equilibrium deuteron absorption used by ssRF-beta training data.

This is the static lineshape only: C-D/O-D powder absorption as packet
capacity, Boltzmann level populations, then the I+ + I- readout. Parameters
are the d_butanol_matched material. RF, AFP, DNP, and spin diffusion are
not applied.
"""

from functools import lru_cache

import numpy as np

# materials/d_butanol_matched.json in ssRF-beta.
N_BINS = 500
R_MIN = -6.186203737076448
R_MAX = 6.186203737076448
LINE_HWHM = 0.023570570424828668
LINE_CD_SCALE = 1.0
LINE_OD_RATIO = 1.3547449396122688
LINE_ETA_CD = 0.06476982746346505
LINE_ETA_OD = 0.032668581840043336
LINE_OD_FRACTION = 0.02127355765149122
LINE_PHI_POINTS = 128
DISPLAY_SCALE = 0.10220936455661135

PLUS, ZERO, MINUS = 0, 1, 2


@lru_cache(maxsize=4)
def _azimuth_quadrature(points):
    if isinstance(points, bool) or not isinstance(points, (int, np.integer)) or not 8 <= points <= 512:
        raise ValueError("Multisite azimuth points must be an integer in [8, 512]")
    nodes, weights = np.polynomial.legendre.leggauss(int(points))
    return (nodes + 1) * np.pi / 4, weights / 2


def _powder_branch(x, epsilon=1, eta=0.03, gamma=0.05, points=64):
    """Complex branch with unit integrated absorption on the infinite x axis."""
    if epsilon not in (-1, 1) or not np.isfinite([eta, gamma]).all() or not 0 <= eta <= 1 or gamma <= 0:
        raise ValueError("Require epsilon=+/-1, eta in [0, 1], and positive linewidth")
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Frequency must be finite")
    phi, weight = _azimuth_quadrature(points)
    y = np.sqrt(3 - eta * np.cos(2 * phi))
    root = np.sqrt(1 - epsilon * x[..., None] - eta * np.cos(2 * phi) + 1j * gamma)
    kernel = np.arctanh(y / root) / root
    if epsilon == -1:
        kernel = -kernel.conjugate()
    return (kernel / (np.pi * y)) @ weight


def _absorption_density(R):
    if LINE_CD_SCALE <= 0 or LINE_OD_RATIO <= 1 or LINE_HWHM <= 0 or not 0 <= LINE_OD_FRACTION <= 1:
        raise ValueError("Require positive scale/width, O-D/C-D ratio > 1, and fraction in [0, 1]")
    R = np.asarray(R, dtype=float)
    od_scale = LINE_CD_SCALE * LINE_OD_RATIO
    cd = (1 - LINE_OD_FRACTION) / LINE_CD_SCALE * _powder_branch(
        R / LINE_CD_SCALE, 1, LINE_ETA_CD, LINE_HWHM / LINE_CD_SCALE, LINE_PHI_POINTS
    )
    od = LINE_OD_FRACTION / od_scale * _powder_branch(
        R / od_scale, 1, LINE_ETA_OD, LINE_HWHM / od_scale, LINE_PHI_POINTS
    )
    return -(cd + od).imag


def deuteron_R():
    """Physical-R bin centers of the matched butanol grid."""
    return np.linspace(R_MIN, R_MAX, N_BINS)


def _level_populations(P):
    """Normalized spin-1 Boltzmann fractions [p+, p0, p-]. Q = 2 - sqrt(4 - 3 P^2)."""
    P = float(np.clip(P, -0.999999, 0.999999))
    Q = 2.0 - np.sqrt(max(0.0, 4.0 - 3.0 * P * P))
    p_plus = 1.0 / 3.0 + 0.5 * P + Q / 6.0
    p_zero = (1.0 - Q) / 3.0
    p_minus = 1.0 / 3.0 - 0.5 * P + Q / 6.0
    p = np.array([p_plus, p_zero, p_minus], dtype=float)
    if np.any(p < -1e-12):
        raise ValueError(f"Invalid spin-1 populations from P={P}, Q={Q}: {p}")
    p = np.maximum(p, 0.0)
    return p / p.sum()


def _packet_weights(R):
    """Area weights mu_k from the absorption density, matching Spin1Model.reset."""
    R = np.asarray(R, dtype=float)
    dR = float(R[1] - R[0])
    density = _absorption_density(R)
    mu = density * dR
    total = float(mu.sum())
    if total <= 0 or not np.isfinite(total):
        raise ValueError("Lineshape area is not positive")
    return mu / total, dR


_WEIGHT_CACHE = {}


def equilibrium_spectrum(P, R=None):
    """Total measured signal I+ + I- for a Boltzmann state at polarization P.

    ``R`` defaults to the matched grid. A shifted grid rebuilds the packet
    weights; the default grid is cached.
    """
    if R is None:
        R = deuteron_R()
        key = None
    else:
        R = np.asarray(R, dtype=float)
        key = (int(R.size), float(R[0]), float(R[-1]))
    cached = _WEIGHT_CACHE.get(key)
    if cached is None:
        cached = _packet_weights(R)
        _WEIGHT_CACHE[key] = cached
    mu, dR = cached
    pref = _level_populations(P)
    n_plus = mu * pref[PLUS]
    n_zero = mu * pref[ZERO]
    n_minus = mu * pref[MINUS]
    i_plus = DISPLAY_SCALE * (n_plus - n_zero) / dR
    i_minus_packet = DISPLAY_SCALE * (n_zero - n_minus) / dR
    r_minus = -R
    order = np.argsort(r_minus)
    i_minus = np.interp(R, r_minus[order], i_minus_packet[order], left=0.0, right=0.0)
    return i_plus + i_minus
