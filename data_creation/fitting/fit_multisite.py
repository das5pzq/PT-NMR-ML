#!/usr/bin/env python3
"""
Multi-site deuteron NMR fitting (Hamada 1981 / Dulya 1997 line-shape theory).

Physical model: total absorption = (1 - K) * C-D + K * O-D, with shared P,
sigma, optional Q-meter false asymmetry (xi), and cubic background.

Frequency-like quantities share one unit system (typically MHz offset).
``split_cd`` / ``split_od`` are 3*w_q; peaks sit near ±split when eta = 0.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

N_BINS = 500


def _clip_small_positive(x, floor=1e-15):
    return np.clip(x, floor, None)


def p_to_r(P):
    """Deuteron polarization P -> Dulya asymmetry parameter r (weak-quadrupole)."""
    P = np.asarray(P, dtype=float)
    disc = np.clip(4.0 - 3.0 * P**2, 0.0, None)
    return (np.sqrt(disc) + P) / (2.0 * (1.0 - P))


def branch_kernel_fixed_phi(R, A, eta, phi, eps):
    """Dipolar-broadened branch kernel f_eps(R, A, eta, phi) from Dulya Eq. (13)/(14)."""
    R = np.asarray(R, dtype=float)
    A = max(float(A), 1e-15)
    phi = np.asarray(phi, dtype=float)

    c2 = np.cos(2.0 * phi)
    b = 1.0 - eps * R - eta * c2
    y_max = np.sqrt(_clip_small_positive(3.0 - eta * c2))

    z = b + 1j * A
    sqrt_z = np.sqrt(z)
    w = (1.0 / sqrt_z) * np.arctanh(y_max / sqrt_z)
    out = (-2.0 / np.pi) * np.imag(w)
    return np.clip(np.real(out), 0.0, None)


def powder_branch(R, A, eta, eps, nphi=64):
    """Powder-averaged branch F_eps(R, A, eta); phi-average skipped when eta = 0."""
    R = np.asarray(R, dtype=float).reshape(-1)
    A = max(float(A), 1e-15)
    eta = float(eta)

    if abs(eta) < 1e-14:
        return branch_kernel_fixed_phi(R, A, 0.0, 0.0, eps)

    phis = np.linspace(0.0, 0.5 * np.pi, int(nphi) + 1)
    c2 = np.cos(2.0 * phis)
    weight = np.sqrt(3.0 / _clip_small_positive(3.0 - eta * c2))
    rr = R[:, None]
    kernels = branch_kernel_fixed_phi(rr, A, eta, phis[None, :], eps)
    return np.mean(weight[None, :] * kernels, axis=1)


def transition_weights(R, P, split, wd, exact_intensity=False):
    """Multiplicative plus/minus branch weights (Dulya Eq. 24 or weak-quadrupole approx)."""
    R = np.asarray(R, dtype=float).reshape(-1)
    r = float(p_to_r(P))

    if not exact_intensity:
        return r * np.ones_like(R), np.ones_like(R)

    vartheta = abs(float(split)) / (3.0 * float(wd))
    plus = (r**2 - r**(1.0 - 3.0 * vartheta * R)) / (r**(1.0 - vartheta * R))
    minus = (r**(1.0 + 3.0 * vartheta * R) - 1.0) / (r**(1.0 + vartheta * R))
    return plus, minus


def site_transition_components(
    x_eff,
    P,
    split,
    sigma,
    eta,
    *,
    wd=16.35,
    exact_intensity=False,
    nphi=64,
):
    """Plus and minus transition contributions for one deuteron site."""
    x_eff = np.asarray(x_eff, dtype=float).reshape(-1)
    split = float(split)
    sigma = float(sigma)

    R = x_eff / split
    A = sigma / abs(split)

    F_plus = powder_branch(R, A, eta, eps=+1, nphi=nphi)
    F_minus = powder_branch(R, A, eta, eps=-1, nphi=nphi)
    w_plus, w_minus = transition_weights(R, P, split, wd, exact_intensity=exact_intensity)

    plus = w_plus * F_plus / abs(split)
    minus = w_minus * F_minus / abs(split)
    return plus, minus


def butanol_absorption_components(
    x_eff,
    P,
    split_cd,
    split_od,
    sigma,
    eta_od,
    K,
    *,
    wd=16.35,
    eta_cd=0.0,
    exact_intensity=False,
    nphi=64,
):
    """Two-site absorption: (1-K)*C-D + K*O-D, with transition-resolved components."""
    x_eff = np.asarray(x_eff, dtype=float).reshape(-1)
    K = float(K)

    cd_plus, cd_minus = site_transition_components(
        x_eff, P, split_cd, sigma, eta_cd,
        wd=wd, exact_intensity=exact_intensity, nphi=nphi,
    )
    od_plus, od_minus = site_transition_components(
        x_eff, P, split_od, sigma, eta_od,
        wd=wd, exact_intensity=exact_intensity, nphi=nphi,
    )

    cd_plus *= 1.0 - K
    cd_minus *= 1.0 - K
    od_plus *= K
    od_minus *= K

    plus_total = cd_plus + od_plus
    minus_total = cd_minus + od_minus

    return {
        "cd_plus": cd_plus,
        "cd_minus": cd_minus,
        "cd_total": cd_plus + cd_minus,
        "od_plus": od_plus,
        "od_minus": od_minus,
        "od_total": od_plus + od_minus,
        "plus_total": plus_total,
        "minus_total": minus_total,
        "absorption": plus_total + minus_total,
    }


def polynomial_background(x, b0, b1, b2, b3):
    x = np.asarray(x, dtype=float).reshape(-1)
    return b0 + b1 * x + b2 * x**2 + b3 * x**3


def qmeter_gain(x_eff, split_ref, xi):
    """Q-meter false-asymmetry factor D(omega) = 1 + 0.5 * xi * (1 + R)."""
    x_eff = np.asarray(x_eff, dtype=float).reshape(-1)
    Rq = x_eff / float(split_ref)
    return 1.0 + 0.5 * float(xi) * (1.0 + Rq)


def freq_bounds_to_window(f_lo_mhz: float, f_hi_mhz: float) -> tuple[float, float]:
    """Legacy Dulya convention: center = 0.5*(f_lo+f_hi), split = 0.5*(f_hi-f_lo)."""
    f_lo = min(float(f_lo_mhz), float(f_hi_mhz))
    f_hi = max(float(f_lo_mhz), float(f_hi_mhz))
    center = 0.5 * (f_lo + f_hi)
    half_width = max(0.5 * (f_hi - f_lo), 1e-6)
    return center, half_width


def signal_model(
    x,
    P,
    amp,
    center,
    cc,
    split_cd,
    split_od,
    sigma,
    eta_od,
    K,
    xi,
    b0,
    b1,
    b2,
    b3,
    *,
    wd=16.35,
    eta_cd=0.0,
    exact_intensity=False,
    nphi=64,
):
    """Full signal on a frequency-offset axis: x_eff = cc*(x - center)."""
    x = np.asarray(x, dtype=float).reshape(-1)
    x_eff = float(cc) * (x - float(center))

    comps = butanol_absorption_components(
        x_eff, P, split_cd, split_od, sigma, eta_od, K,
        wd=wd, eta_cd=eta_cd, exact_intensity=exact_intensity, nphi=nphi,
    )

    split_ref = split_od if abs(split_od) >= abs(split_cd) else split_cd
    gain = qmeter_gain(x_eff, split_ref, xi)
    background = polynomial_background(x, b0, b1, b2, b3)
    return float(amp) * comps["absorption"] * gain + background


def signal_model_mhz(
    freq_mhz,
    P,
    amp,
    f_lo_mhz,
    f_hi_mhz,
    cc,
    split_cd,
    split_od,
    sigma,
    eta_od,
    K,
    xi,
    b0,
    b1,
    b2,
    b3,
    *,
    wd=16.35,
    eta_cd=0.0,
    exact_intensity=False,
    nphi=64,
):
    """Multi-site model on an absolute MHz axis; center derived from f_lo/f_hi."""
    center, _ = freq_bounds_to_window(f_lo_mhz, f_hi_mhz)
    return signal_model(
        freq_mhz, P, amp, center, cc, split_cd, split_od, sigma, eta_od, K, xi,
        b0, b1, b2, b3,
        wd=wd, eta_cd=eta_cd, exact_intensity=exact_intensity, nphi=nphi,
    )


def component_curves(
    params,
    x,
    *,
    wd=16.35,
    eta_cd=0.0,
    exact_intensity=False,
    nphi=64,
):
    """Named model components on x, scaled by amp; baseline kept separate."""
    x = np.asarray(x, dtype=float).reshape(-1)
    p = dict(params)
    x_eff = float(p["cc"]) * (x - float(p["center"]))

    comps = butanol_absorption_components(
        x_eff, p["P"], p["split_cd"], p["split_od"], p["sigma"], p["eta_od"], p["K"],
        wd=wd, eta_cd=eta_cd, exact_intensity=exact_intensity, nphi=nphi,
    )

    amp = float(p["amp"])
    for key in list(comps.keys()):
        comps[key] = amp * comps[key]

    split_ref = p["split_od"] if abs(p["split_od"]) >= abs(p["split_cd"]) else p["split_cd"]
    gain = qmeter_gain(x_eff, split_ref, p["xi"])
    background = polynomial_background(x, p["b0"], p["b1"], p["b2"], p["b3"])
    absorption_measured = comps["absorption"] * gain

    comps.update(
        {
            "absorption_physical": comps["absorption"],
            "qmeter_gain": gain,
            "absorption_measured": absorption_measured,
            "background": background,
            "total": absorption_measured + background,
            "x_eff": x_eff,
        }
    )
    return comps


PAKE_PARAM_ORDER = (
    "P", "amp", "center", "cc", "split", "sigma", "eta", "b0", "b1", "b2", "b3",
)

PAKE_MHZ_PARAM_ORDER = (
    "P", "amp", "sigma", "eta", "f_lo_mhz", "f_hi_mhz", "b0", "b1", "b2", "b3",
)


def pake_doublet_absorption(
    x_eff, P, split, sigma, eta=0.0, *, wd=16.35, exact_intensity=False, nphi=64,
):
    plus, minus = site_transition_components(
        x_eff, P, split, sigma, eta,
        wd=wd, exact_intensity=exact_intensity, nphi=nphi,
    )
    return plus + minus


def pake_doublet_model(
    x, P, amp, center, cc, split, sigma, eta, b0, b1, b2, b3,
    *, wd=16.35, exact_intensity=False, nphi=64,
):
    x = np.asarray(x, dtype=float).reshape(-1)
    x_eff = float(cc) * (x - float(center))
    absorption = pake_doublet_absorption(
        x_eff, P, split, sigma, eta,
        wd=wd, exact_intensity=exact_intensity, nphi=nphi,
    )
    background = polynomial_background(x, b0, b1, b2, b3)
    return float(amp) * absorption + background


def pake_doublet_model_mhz(
    freq_mhz, P, amp, sigma, eta, f_lo_mhz, f_hi_mhz, b0, b1, b2, b3,
    *, wd=16.35, exact_intensity=False, nphi=64,
):
    freq_mhz = np.asarray(freq_mhz, dtype=float).reshape(-1)
    center, split = freq_bounds_to_window(f_lo_mhz, f_hi_mhz)
    x_eff = freq_mhz - center
    absorption = pake_doublet_absorption(
        x_eff, P, split, sigma, eta,
        wd=wd, exact_intensity=exact_intensity, nphi=nphi,
    )
    background = polynomial_background(freq_mhz, b0, b1, b2, b3)
    return float(amp) * absorption + background


def default_pake_p0(**kwargs) -> Dict[str, float]:
    defaults = dict(
        P=0.35, amp=1.0, center=0.0, cc=1.0, split=0.88, sigma=0.05, eta=0.0,
        b0=0.0, b1=0.0, b2=0.0, b3=0.0,
    )
    defaults.update(kwargs)
    return defaults


def default_pake_bounds(**kwargs) -> Dict[str, Tuple[float, float]]:
    defaults = dict(
        P=(-0.99, 0.99), amp=(-np.inf, np.inf), center=(-0.5, 0.5), cc=(0.5, 1.5),
        split=(1e-4, np.inf), sigma=(1e-5, np.inf), eta=(0.0, 0.95),
        b0=(-np.inf, np.inf), b1=(-np.inf, np.inf), b2=(-np.inf, np.inf), b3=(-np.inf, np.inf),
    )
    defaults.update(kwargs)
    return defaults


def default_pake_mhz_p0(**kwargs) -> Dict[str, float]:
    defaults = dict(
        P=0.35, amp=1.0, sigma=0.05, eta=0.0, f_lo_mhz=32.08, f_hi_mhz=33.28,
        b0=0.0, b1=0.0, b2=0.0, b3=0.0,
    )
    defaults.update(kwargs)
    return defaults


def default_pake_mhz_bounds(**kwargs) -> Dict[str, Tuple[float, float]]:
    defaults = dict(
        P=(-0.99, 0.99), amp=(-np.inf, np.inf), sigma=(1e-5, np.inf), eta=(0.0, 0.95),
        f_lo_mhz=(-np.inf, np.inf), f_hi_mhz=(-np.inf, np.inf),
        b0=(-np.inf, np.inf), b1=(-np.inf, np.inf), b2=(-np.inf, np.inf), b3=(-np.inf, np.inf),
    )
    defaults.update(kwargs)
    return defaults


def fit_chi_squared(y, y_model, yerr=None) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    y_model = np.asarray(y_model, dtype=float).reshape(-1)
    sigma = np.ones_like(y) if yerr is None else np.maximum(np.asarray(yerr, dtype=float).reshape(-1), 1e-12)
    residuals = (y - y_model) / sigma
    return float(np.sum(residuals**2))


PARAM_ORDER = (
    "P", "amp", "center", "cc", "split_cd", "split_od", "sigma", "eta_od", "K", "xi",
    "b0", "b1", "b2", "b3",
)

MULTISITE_MHZ_PARAM_ORDER = (
    "P", "amp", "f_lo_mhz", "f_hi_mhz", "cc", "split_cd", "split_od", "sigma",
    "eta_od", "K", "xi", "b0", "b1", "b2", "b3",
)


@dataclass
class FitSummary:
    params: Dict[str, float]
    free_names: Tuple[str, ...]
    pcov: Optional[np.ndarray]
    result: object


def _merge_params(p_free, param_order, p0, fixed):
    params = dict(p0)
    params.update(fixed)
    free_names = [name for name in param_order if name not in fixed]
    for name, value in zip(free_names, p_free):
        params[name] = float(value)
    return params


def _estimate_covariance(result):
    jac = getattr(result, "jac", None)
    if jac is None:
        return None

    jac = np.asarray(jac, dtype=float)
    if jac.ndim != 2:
        return None

    n_data, n_par = jac.shape
    if n_data <= n_par:
        return None

    try:
        _, svals, vt = np.linalg.svd(jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(jac.shape) * svals[0]
        keep = svals > threshold
        if not np.any(keep):
            return None

        svals = svals[keep]
        vt = vt[keep, :]
        jtj_inv = (vt.T / (svals**2)) @ vt

        rss = np.sum(np.asarray(result.fun, dtype=float) ** 2)
        dof = max(n_data - n_par, 1)
        return jtj_inv * (rss / dof)
    except np.linalg.LinAlgError:
        return None


def _run_fit(
    x,
    y,
    yerr,
    *,
    param_order,
    p0,
    bounds,
    fixed,
    model_fn,
    loss="soft_l1",
    f_scale=1.0,
    max_nfev=50000,
) -> FitSummary:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    sigma = np.ones_like(y) if yerr is None else np.maximum(np.asarray(yerr, dtype=float).reshape(-1), 1e-12)
    fixed = {} if fixed is None else dict(fixed)

    free_names = tuple(name for name in param_order if name not in fixed)
    if not free_names:
        raise ValueError("At least one parameter must be free.")

    p0_vec = np.array([p0[name] for name in free_names], dtype=float)
    lb = np.array([bounds[name][0] for name in free_names], dtype=float)
    ub = np.array([bounds[name][1] for name in free_names], dtype=float)

    def residuals(p_free):
        params = _merge_params(p_free, param_order, p0, fixed)
        return (model_fn(x, params) - y) / sigma

    result = least_squares(
        residuals, p0_vec, bounds=(lb, ub),
        loss=loss, f_scale=f_scale, max_nfev=max_nfev,
    )
    best = _merge_params(result.x, param_order, p0, fixed)
    return FitSummary(
        params=best,
        free_names=free_names,
        pcov=_estimate_covariance(result),
        result=result,
    )


def default_multisite_p0(**kwargs) -> Dict[str, float]:
    defaults = dict(
        P=0.35, amp=1.0, center=0.0, cc=1.0, split_cd=0.90, split_od=1.15,
        sigma=0.05, eta_od=0.15, K=0.12, xi=0.0, b0=0.0, b1=0.0, b2=0.0, b3=0.0,
    )
    defaults.update(kwargs)
    return defaults


def default_multisite_bounds(**kwargs) -> Dict[str, Tuple[float, float]]:
    defaults = dict(
        P=(-0.99, 0.99), amp=(-np.inf, np.inf), center=(-0.5, 0.5), cc=(0.5, 1.5),
        split_cd=(1e-4, np.inf), split_od=(1e-4, np.inf), sigma=(1e-5, np.inf),
        eta_od=(0.0, 0.95), K=(0.0, 1.0), xi=(-0.5, 0.5),
        b0=(-np.inf, np.inf), b1=(-np.inf, np.inf), b2=(-np.inf, np.inf), b3=(-np.inf, np.inf),
    )
    defaults.update(kwargs)
    return defaults


def default_multisite_mhz_p0(**kwargs) -> Dict[str, float]:
    defaults = dict(
        P=0.35, amp=1.0, f_lo_mhz=32.08, f_hi_mhz=33.28, cc=1.0,
        split_cd=0.90, split_od=1.15, sigma=0.05, eta_od=0.15, K=0.12, xi=0.0,
        b0=0.0, b1=0.0, b2=0.0, b3=0.0,
    )
    defaults.update(kwargs)
    return defaults


def default_multisite_mhz_bounds(**kwargs) -> Dict[str, Tuple[float, float]]:
    defaults = dict(
        P=(-0.99, 0.99), amp=(-np.inf, np.inf),
        f_lo_mhz=(-np.inf, np.inf), f_hi_mhz=(-np.inf, np.inf), cc=(0.5, 1.5),
        split_cd=(1e-4, np.inf), split_od=(1e-4, np.inf), sigma=(1e-5, np.inf),
        eta_od=(0.0, 0.95), K=(0.0, 1.0), xi=(-0.5, 0.5),
        b0=(-np.inf, np.inf), b1=(-np.inf, np.inf), b2=(-np.inf, np.inf), b3=(-np.inf, np.inf),
    )
    defaults.update(kwargs)
    return defaults


def fit_signal(
    x, y, yerr=None, p0=None, bounds=None, fixed=None,
    *, wd=16.35, eta_cd=0.0, exact_intensity=False, nphi=64,
    loss="soft_l1", f_scale=1.0, max_nfev=50000,
):
    if p0 is None:
        p0 = default_multisite_p0()
    if bounds is None:
        bounds = default_multisite_bounds()

    def model_fn(x_axis, params):
        return signal_model(
            x_axis, **params,
            wd=wd, eta_cd=eta_cd, exact_intensity=exact_intensity, nphi=nphi,
        )

    return _run_fit(
        x, y, yerr, param_order=PARAM_ORDER, p0=p0, bounds=bounds, fixed=fixed,
        model_fn=model_fn, loss=loss, f_scale=f_scale, max_nfev=max_nfev,
    )


def fit_signal_mhz(
    freq_mhz, y, yerr=None, p0=None, bounds=None, fixed=None,
    *, wd=16.35, eta_cd=0.0, exact_intensity=False, nphi=64,
    loss="soft_l1", f_scale=1.0, max_nfev=50000,
) -> FitSummary:
    if p0 is None:
        p0 = default_multisite_mhz_p0()
    if bounds is None:
        bounds = default_multisite_mhz_bounds()

    def model_fn(freq_axis, params):
        return signal_model_mhz(
            freq_axis, **params,
            wd=wd, eta_cd=eta_cd, exact_intensity=exact_intensity, nphi=nphi,
        )

    return _run_fit(
        freq_mhz, y, yerr, param_order=MULTISITE_MHZ_PARAM_ORDER,
        p0=p0, bounds=bounds, fixed=fixed, model_fn=model_fn,
        loss=loss, f_scale=f_scale, max_nfev=max_nfev,
    )


def fit_clean_then_full(
    x, y, yerr, p0_full, bounds_full, *,
    clean_window=None, fixed_main=None, fixed_full=None,
    wd=16.35, eta_cd=0.0, exact_intensity=False, nphi=64,
    loss="soft_l1", f_scale=1.0, max_nfev=50000,
):
    """Stage 1: C-D-only fit (K=0) on clean_window; stage 2: full two-site fit."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    yerr = None if yerr is None else np.asarray(yerr, dtype=float).reshape(-1)

    if clean_window is None:
        mask = np.ones_like(x, dtype=bool)
    else:
        lo, hi = clean_window
        mask = (x >= lo) & (x <= hi)

    stage1_fixed = {
        "K": 0.0,
        "eta_od": p0_full["eta_od"],
        "split_od": p0_full["split_od"],
        "xi": 0.0,
    }
    if fixed_main is not None:
        stage1_fixed.update(dict(fixed_main))

    main_fit = fit_signal(
        x[mask], y[mask], None if yerr is None else yerr[mask],
        p0=p0_full, bounds=bounds_full, fixed=stage1_fixed,
        wd=wd, eta_cd=eta_cd, exact_intensity=exact_intensity, nphi=nphi,
        loss=loss, f_scale=f_scale, max_nfev=max_nfev,
    )

    seeded = dict(p0_full)
    seeded.update(main_fit.params)

    full_fit = fit_signal(
        x, y, yerr, p0=seeded, bounds=bounds_full, fixed=fixed_full,
        wd=wd, eta_cd=eta_cd, exact_intensity=exact_intensity, nphi=nphi,
        loss=loss, f_scale=f_scale, max_nfev=max_nfev,
    )
    return main_fit, full_fit


def fit_pake_doublet(
    x, y, yerr=None, p0=None, bounds=None, fixed=None,
    *, wd=16.35, exact_intensity=False, nphi=64,
    loss="soft_l1", f_scale=1.0, max_nfev=50000,
) -> FitSummary:
    if p0 is None:
        p0 = default_pake_p0()
    if bounds is None:
        bounds = default_pake_bounds()

    def model_fn(x_axis, params):
        return pake_doublet_model(
            x_axis, **params,
            wd=wd, exact_intensity=exact_intensity, nphi=nphi,
        )

    return _run_fit(
        x, y, yerr, param_order=PAKE_PARAM_ORDER, p0=p0, bounds=bounds, fixed=fixed,
        model_fn=model_fn, loss=loss, f_scale=f_scale, max_nfev=max_nfev,
    )


def fit_pake_doublet_mhz(
    freq_mhz, y, yerr=None, p0=None, bounds=None, fixed=None,
    *, wd=16.35, exact_intensity=False, nphi=64,
    loss="soft_l1", f_scale=1.0, max_nfev=50000,
) -> FitSummary:
    if p0 is None:
        p0 = default_pake_mhz_p0()
    if bounds is None:
        bounds = default_pake_mhz_bounds()

    def model_fn(freq_axis, params):
        return pake_doublet_model_mhz(
            freq_axis, **params,
            wd=wd, exact_intensity=exact_intensity, nphi=nphi,
        )

    return _run_fit(
        freq_mhz, y, yerr, param_order=PAKE_MHZ_PARAM_ORDER,
        p0=p0, bounds=bounds, fixed=fixed, model_fn=model_fn,
        loss=loss, f_scale=f_scale, max_nfev=max_nfev,
    )


def compute_transition_areas(
    x, fit, *, wd=16.35, eta_cd=0.0, exact_intensity=False, nphi=64,
    n_mc=300, seed=123,
):
    """Integrate physical absorption components (before Q-meter gain and baseline)."""
    x = np.asarray(x, dtype=float).reshape(-1)
    curves = component_curves(
        fit.params, x, wd=wd, eta_cd=eta_cd,
        exact_intensity=exact_intensity, nphi=nphi,
    )

    out = {
        "area_cd_plus": np.trapz(curves["cd_plus"], x),
        "area_cd_minus": np.trapz(curves["cd_minus"], x),
        "area_od_plus": np.trapz(curves["od_plus"], x),
        "area_od_minus": np.trapz(curves["od_minus"], x),
    }
    out["area_plus_total"] = out["area_cd_plus"] + out["area_od_plus"]
    out["area_minus_total"] = out["area_cd_minus"] + out["area_od_minus"]
    out["area_cd_total"] = out["area_cd_plus"] + out["area_cd_minus"]
    out["area_od_total"] = out["area_od_plus"] + out["area_od_minus"]
    out["area_total_physical"] = out["area_plus_total"] + out["area_minus_total"]
    out["area_diff_total"] = out["area_plus_total"] - out["area_minus_total"]

    if fit.pcov is None:
        return out

    rng = np.random.default_rng(seed)
    free_names = list(fit.free_names)
    try:
        if not np.all(np.isfinite(fit.pcov)):
            return out
        if np.linalg.cond(fit.pcov) > 1e12:
            return out
        L = np.linalg.cholesky(fit.pcov)
    except np.linalg.LinAlgError:
        return out

    base = np.array([fit.params[name] for name in free_names], dtype=float)
    draws = base + rng.standard_normal((n_mc, len(free_names))) @ L.T

    plus_vals, minus_vals = [], []
    for draw in draws:
        trial = dict(fit.params)
        for name, value in zip(free_names, draw):
            trial[name] = float(value)

        trial["P"] = float(np.clip(trial["P"], -0.999, 0.999))
        trial["cc"] = float(np.clip(trial["cc"], 1e-9, np.inf))
        trial["split_cd"] = float(np.clip(trial["split_cd"], 1e-9, np.inf))
        trial["split_od"] = float(np.clip(trial["split_od"], 1e-9, np.inf))
        trial["sigma"] = float(np.clip(trial["sigma"], 1e-9, np.inf))
        trial["eta_od"] = float(np.clip(trial["eta_od"], 0.0, 0.999))
        trial["K"] = float(np.clip(trial["K"], 0.0, 1.0))

        curves_i = component_curves(
            trial, x, wd=wd, eta_cd=eta_cd,
            exact_intensity=exact_intensity, nphi=nphi,
        )
        plus_vals.append(np.trapz(curves_i["plus_total"], x))
        minus_vals.append(np.trapz(curves_i["minus_total"], x))

    plus_vals = np.asarray(plus_vals)
    minus_vals = np.asarray(minus_vals)
    out["area_plus_total_std"] = np.std(plus_vals, ddof=1)
    out["area_minus_total_std"] = np.std(minus_vals, ddof=1)
    out["area_diff_total_std"] = np.std(plus_vals - minus_vals, ddof=1)
    return out


def plot_fit(
    x, y, yerr, fit, *,
    title="Multi-site Dulya/Hamada fit",
    wd=16.35, eta_cd=0.0, exact_intensity=False, nphi=64,
    savepath="multisite_dulya_fit.png",
):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    yerr = None if yerr is None else np.asarray(yerr, dtype=float).reshape(-1)

    order = np.argsort(x)
    xs, ys = x[order], y[order]
    yerrs = None if yerr is None else yerr[order]

    dense = np.linspace(xs.min(), xs.max(), max(800, 3 * len(xs)))
    curves = component_curves(
        fit.params, dense, wd=wd, eta_cd=eta_cd,
        exact_intensity=exact_intensity, nphi=nphi,
    )
    y_model_data = signal_model(
        x, **fit.params, wd=wd, eta_cd=eta_cd,
        exact_intensity=exact_intensity, nphi=nphi,
    )
    residuals = y - y_model_data
    areas = compute_transition_areas(
        dense, fit, wd=wd, eta_cd=eta_cd,
        exact_intensity=exact_intensity, nphi=nphi,
    )

    fig = plt.figure(figsize=(9.0, 7.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.0], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)

    if yerrs is not None:
        ax.errorbar(xs, ys, yerr=yerrs, fmt="o", ms=3, alpha=0.85, label="data")
    else:
        ax.plot(xs, ys, "o", ms=3, alpha=0.85, label="data")

    ax.plot(dense, curves["total"], lw=2.2, label="total fit")
    ax.plot(dense, curves["cd_total"], lw=1.5, linestyle="--", label="C-D site")
    ax.plot(dense, curves["od_total"], lw=1.5, linestyle=":", label="O-D site")
    ax.plot(dense, curves["plus_total"], lw=1.2, linestyle="-.", label="plus branch")
    ax.plot(dense, curves["minus_total"], lw=1.2, linestyle=(0, (5, 2, 1, 2)), label="minus branch")
    ax.plot(dense, curves["background"], lw=1.0, alpha=0.8, label="background")

    ax.set_ylabel("Signal")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", ncols=2)

    txt = (
        f"P = {fit.params['P']:.5f}\n"
        f"r = {float(p_to_r(fit.params['P'])):.5f}\n"
        f"split_cd = {fit.params['split_cd']:.5g}\n"
        f"split_od = {fit.params['split_od']:.5g}\n"
        f"sigma = {fit.params['sigma']:.5g}\n"
        f"eta_od = {fit.params['eta_od']:.5g}\n"
        f"K = {fit.params['K']:.5g}\n"
        f"A_plus = {areas['area_plus_total']:.5g}\n"
        f"A_minus = {areas['area_minus_total']:.5g}\n"
        f"A_plus - A_minus = {areas['area_diff_total']:.5g}"
    )
    ax.text(
        0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.78, lw=0.5),
    )

    axr.axhline(0.0, lw=1.0, alpha=0.8)
    if yerrs is not None:
        axr.errorbar(x, residuals, yerr=yerr, fmt="o", ms=3, alpha=0.85)
    else:
        axr.plot(x, residuals, "o", ms=3, alpha=0.85)
    axr.set_xlabel("Frequency / offset")
    axr.set_ylabel("resid")
    axr.grid(alpha=0.25)

    for label in ax.get_xticklabels():
        label.set_visible(False)

    plt.tight_layout()
    plt.savefig(savepath, dpi=170)
    plt.show()
    return areas


if __name__ == "__main__":
    x_full = np.linspace(-6.0, 6.0, N_BINS)
    _event_id, y_full = load_event_csv(Path("event_1432659714.csv"))
    fit_window = (-6, 6)
    m = (x_full >= fit_window[0]) & (x_full <= fit_window[1])
    x = x_full[m]
    y = y_full[m]
    yerr = np.full_like(x, 1.816364e-05, dtype=np.float64)

    p0_full = dict(
        P=0.35, amp=0.9, center=0.0, cc=1.0, split_cd=0.88, split_od=1.12,
        sigma=0.05, eta_od=0.12, K=0.10, xi=0.0, b0=0.0, b1=0.0, b2=0.0, b3=0.0,
    )
    bounds_full = dict(
        P=(0.0, 0.95), amp=(-np.inf, np.inf), center=(-0.2, 0.2), cc=(0.0, 1.2),
        split_cd=(0.0, 1.5), split_od=(0.0, 1.8), sigma=(0.01, 0.20),
        eta_od=(0.0, 0.50), K=(0.0, 0.40), xi=(-0.15, 0.15),
        b0=(-1.0, 1.0), b1=(-1.0, 1.0), b2=(-1.0, 1.0), b3=(-1.0, 1.0),
    )
    fixed_full = {"cc": 1.0, "xi": 0.0, "b2": 0.0, "b3": 0.0}
    clean_window = (-1.05, 1.05)

    main_fit, full_fit = fit_clean_then_full(
        x, y, yerr, p0_full, bounds_full,
        clean_window=clean_window,
        fixed_main={"cc": 1.0, "b2": 0.0, "b3": 0.0},
        fixed_full=fixed_full,
        wd=16.35, eta_cd=0.0, exact_intensity=True, nphi=48,
        loss="soft_l1", f_scale=1.0, max_nfev=300,
    )

    print("Stage-1 C-D-only fit:")
    for name in PARAM_ORDER:
        if name in main_fit.params:
            print(f"  {name:>8s} = {main_fit.params[name]:.8g}")

    print("\nStage-2 full physical C-D + O-D fit:")
    for name in PARAM_ORDER:
        if name in full_fit.params:
            print(f"  {name:>8s} = {full_fit.params[name]:.8g}")

    print("\nOptimizer success:", full_fit.result.success)
    print("Optimizer message:", full_fit.result.message)

    areas = plot_fit(
        x, y, yerr, full_fit,
        title="Physical multi-site fit: C-D + O-D composition",
        wd=16.35, eta_cd=0.0, exact_intensity=True, nphi=48,
        savepath="./multisite_fit_demo.png",
    )

    print("\nTransition-resolved areas (physical absorption, no baseline):")
    for key in sorted(areas):
        print(f"  {key:>22s} : {areas[key]}")
