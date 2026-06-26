import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import json

FIT_PARAM_NAMES = (
    "P",
    "amp",
    "center",
    "cc",
    "split_cd",
    "split_od",
    "sigma",
    "eta_od",
    "eta_cd",
    "K",
    "xi",
    "b0",
    "b1",
    "b2",
    "b3",
)

class DulyaModel:

    def __init__(self, param_keys=FIT_PARAM_NAMES):
        self.param_keys = param_keys 
        self.param_names = {name: i for i, name in enumerate(param_keys)}

    def split_fit_params(self, params):
        fixed_params = {name: value for name, value in params.items() if name not in self.param_names}
        fit_params = {name: params[name] for name in self.param_names}
        return fit_params, fixed_params

    def params_to_vector(self, params):
        return np.array([params[name] for name in self.param_keys], dtype=float)

    def vector_to_params(self, vector):
        return {name: float(value) for name, value in zip(self.param_keys, vector)}


    ### Chi-Squared Function For this model ###

    def chi_squared(self,
        params,
        x,
        y,
        yerr,
        fixed_params=None,
    ):
        if isinstance(params, np.ndarray):
            fit_params = self.vector_to_params(params)
        else:
            fit_params = dict(params)
        if fixed_params is not None:
            fit_params.update(fixed_params)
        model = self.signal_model(x, **fit_params)
        return np.sum((y - model) ** 2 / yerr**2)


    def p_to_r(self, P):
        """Deuteron polarization P -> Dulya asymmetry parameter r (weak-quadrupole)."""
        P = np.asarray(P, dtype=float)
        disc = np.clip(4.0 - 3.0 * P**2, 0.0, None)
        return (np.sqrt(disc) + P) / (2.0 * (1.0 - P))



    ### Now actual physics here ###

    def branch_kernel_fixed_phi(self,R, A, eta, phi, eps):
        """
        Dipolar-broadened branch kernel f_eps(R, A, eta, phi).

        Parameters
        ----------
        R : array-like
            Dimensionless frequency variable of one site:
                R = (omega - omega_d) / (3*w_q)
        A : float
            Dimensionless Lorentzian width parameter for that site:
                A = sigma / (3*w_q)
        eta : float
            Quadrupole asymmetry parameter of that site.
        phi : float or array-like
            Azimuthal angle in radians.
        eps : {+1, -1}
            Branch index.

        Notes
        -----
        This is the closed-form evaluation of the convolution integral appearing in
        Eq. (13) / (14) of the Dulya paper.  The principal-value / branch-cut issues
        are handled automatically by NumPy's complex arithmetic.
        """
        R = np.asarray(R, dtype=float)
        A = max(float(A), 1e-15)
        phi = np.asarray(phi, dtype=float)

        c2 = np.cos(2.0 * phi)
        b = 1.0 - eps * R - eta * c2
        y_max = np.sqrt(3.0 - eta * c2)

        # Complex parameter z = b + i A.
        z = b + 1j * A
        sqrt_z = np.sqrt(z)

        # Closed form of the integral 2A/pi * ∫ dy / ((y^2 - b)^2 + A^2).
        w = (1.0 / sqrt_z) * np.arctanh(y_max / sqrt_z)
        out = (-2.0 / np.pi) * np.imag(w)

        return np.real(out)

    def powder_branch(self,
        R, 
        A, 
        eta, 
        eps, 
        nphi=64
    ):
        """
        Powder-averaged branch function F_eps(R, A, eta).

        This performs the phi-average from Dulya Eq. (15).
        For eta = 0 the branch is phi-independent, so we skip the average.
        For more direction ask your mother figure.
        """
        R = np.asarray(R, dtype=float).reshape(-1)
        A = max(float(A), 1e-15)
        eta = float(eta)

        if abs(eta) < 1e-14:
            return self.branch_kernel_fixed_phi(R, A, 0.0, 0.0, eps)

        phis = np.linspace(0.0, 0.5 * np.pi, int(nphi) + 1)
        c2 = np.cos(2.0 * phis)
        weight = np.sqrt(3.0 / (3.0 - eta * c2))

        rr = R[:, None]
        kernels = self.branch_kernel_fixed_phi(rr, A, eta, phis[None, :], eps)
        return np.mean(weight[None, :] * kernels, axis=1)

    def transition_weights(self,
        R, 
        P, 
        split, 
        wd, 
        exact_intensity=False
    ):
        """
        Return the multiplicative weights for the plus and minus branches.

        Parameters
        ----------
        R : array-like
            Dimensionless frequency variable of one site.
        P : float
            Deuteron vector polarization.
        split : float
            The site frequency scale 3*w_q in the same units as x.
        wd : float
            Larmor frequency in the same units as x.
        exact_intensity : bool
            If True, use the frequency-dependent Dulya factors (Eq. 24).
            If False, use the butanol weak-quadrupole approximation (Eq. 25).

        Notes
        -----
        Any overall R-independent scale factor is left out on purpose because the
        fit already has a free amplitude parameter.  What matters here is the shape
        asymmetry across the line.
        """
        R = np.asarray(R, dtype=float).reshape(-1)
        r = float(self.p_to_r(P))

        if not exact_intensity:
            return r * np.ones_like(R), np.ones_like(R)

        # vartheta = w_q / w_d = split / (3 * w_d)
        vartheta = abs(float(split)) / (3.0 * float(wd))

        # Dulya Eq. (24), ignoring the common 1/w_q prefactor that is absorbed into
        # the overall fit amplitude and the site-mixing normalization.
        plus = (r**2 - r**(1.0 - 3.0 * vartheta * R)) / (r**(1.0 - vartheta * R))
        minus = (r**(1.0 + 3.0 * vartheta * R) - 1.0) / (r**(1.0 + vartheta * R))
        return plus, minus

    def site_transition_components(self,
        x_eff,
        P,
        split,
        sigma,
        eta,
        *,
        wd=32.68,
        exact_intensity=False,
        nphi=64,
    ):
        """
        Transition-resolved contributions of a single deuteron site.

        Parameters
        ----------
        x_eff : array-like
            Effective frequency offset after any center/axis calibration correction.
        P : float
            Common deuteron vector polarization.
        split : float
            Site frequency scale 3*w_q (same units as x).
        sigma : float
            Common physical dipolar width (same units as x).
        eta : float
            Site asymmetry parameter.
        wd : float
            Larmor frequency.
        exact_intensity : bool
            Use frequency-dependent intensity factors if True.
        nphi : int
            Number of phi steps for the powder average.

        Returns
        -------
        plus, minus : ndarray, ndarray
            The two transition-family contributions for this site.
        """
        x_eff = np.asarray(x_eff, dtype=float).reshape(-1)
        split = float(split)
        sigma = float(sigma)

        R = x_eff / split
        A = sigma / abs(split)

        F_plus = self.powder_branch(R, A, eta, eps=+1, nphi=nphi)
        F_minus = self.powder_branch(R, A, eta, eps=-1, nphi=nphi)
        w_plus, w_minus = self.transition_weights(R, P, split, wd, exact_intensity=exact_intensity)

        # The 1/w_q prefactor from Dulya's absorption function translates to 1/split
        # up to a common constant factor of 3.  That constant can be absorbed into
        # the overall amplitude, but the *relative* site scaling with split matters.
        plus = w_plus * F_plus / abs(split)
        minus = w_minus * F_minus / abs(split)
        return plus, minus

    def branch_kernel_fixed_phi(self,
        R, 
        A, 
        eta, 
        phi, 
        eps
    ):
        """
        Dipolar-broadened branch kernel f_eps(R, A, eta, phi).

        Parameters
        ----------
        R : array-like
            Dimensionless frequency variable of one site:
                R = (omega - omega_d) / (3*w_q)
        A : float
            Dimensionless Lorentzian width parameter for that site:
                A = sigma / (3*w_q)
        eta : float
            Quadrupole asymmetry parameter of that site.
        phi : float or array-like
            Azimuthal angle in radians.
        eps : {+1, -1}
            Branch index.

        Notes
        -----
        This is the closed-form evaluation of the convolution integral appearing in
        Eq. (13) / (14) of the Dulya paper.  The principal-value / branch-cut issues
        are handled automatically by NumPy's complex arithmetic.
        """
        R = np.asarray(R, dtype=float)
        A = max(float(A), 1e-15)
        phi = np.asarray(phi, dtype=float)

        c2 = np.cos(2.0 * phi)
        b = 1.0 - eps * R - eta * c2
        y_max = np.sqrt(3.0 - eta * c2)

        # Complex parameter z = b + i A.
        z = b + 1j * A
        sqrt_z = np.sqrt(z)

        # Closed form of the integral 2A/pi * ∫ dy / ((y^2 - b)^2 + A^2).
        w = (1.0 / sqrt_z) * np.arctanh(y_max / sqrt_z)
        out = (-2.0 / np.pi) * np.imag(w)

        return np.real(out)

    def butanol_absorption_components(self,
        x_eff,
        P,
        split_cd,
        split_od,
        sigma,
        eta_od,
        K,
        *,
        wd=32.68,
        eta_cd=0.0,
        exact_intensity=False,
        nphi=64,
    ):
        """
        Physical two-site absorption model for deuterated-butanol-like spectra.
        Most of the time you can tell from the lineshape if its a contaminent or artifact,
        But you need to pay attention on which transition each area belongs to...

        Parameters
        ----------
        x_eff : array-like
            Effective frequency axis after center/cc correction.
        P : float
            Common deuteron vector polarization.
        split_cd, split_od : float
            Site frequency scales 3*w_q for the C-D and O-D sites.
        sigma : float
            Common physical dipolar width.
        eta_od : float
            O-D quadrupole asymmetry parameter.
        K : float
            Relative O-D-like contribution.  Total model is:
                (1-K) * C-D + K * O-D
            so K must lie between 0 and 1.
        wd : float
            Larmor frequency.
        eta_cd : float
            C-D asymmetry parameter.  Default is 0.0, which is standard for butanol.
        exact_intensity : bool
            Use Dulya Eq. (24) if True, Eq. (25)-style approximation if False.
        nphi : int
            Number of phi points for the powder average.

        Returns
        -------
        dict
            Transition-resolved and site-resolved physical absorption components.
        """
        x_eff = np.asarray(x_eff, dtype=float).reshape(-1)
        K = float(K)

        cd_plus, cd_minus = self.site_transition_components(
            x_eff,
            P,
            split_cd,
            sigma,
            eta_cd,
            wd=wd,
            exact_intensity=exact_intensity,
            nphi=nphi,
        )
        od_plus, od_minus = self.site_transition_components(
            x_eff,
            P,
            split_od,
            sigma,
            eta_od,
            wd=wd,
            exact_intensity=exact_intensity,
            nphi=nphi,
        )

        cd_plus *= (1.0 - K)
        cd_minus *= (1.0 - K)
        od_plus *= K
        od_minus *= K

        plus_total = cd_plus + od_plus
        minus_total = cd_minus + od_minus
        absorption = plus_total + minus_total

        return {
            "cd_plus": cd_plus,
            "cd_minus": cd_minus,
            "cd_total": cd_plus + cd_minus,
            "od_plus": od_plus,
            "od_minus": od_minus,
            "od_total": od_plus + od_minus,
            "plus_total": plus_total,
            "minus_total": minus_total,
            "absorption": absorption,
        }

    def polynomial_background(self, 
        x, 
        b0, 
        b1, 
        b2, 
        b3
    ):
        """Residual background polynomial, following Dulya's Eq. (27)/(29)."""
        x = np.asarray(x, dtype=float).reshape(-1)
        return b0 + b1 * x + b2 * x**2 + b3 * x**3

    def qmeter_gain(self,
        x_eff, 
        split_ref, 
        xi
    ):
        """
        Simple false-asymmetry correction factor.

        Dulya parameterizes the Q-meter distortion as:
            D(omega) = 1 + 0.5 * xi * (1 + R)

        Here R is taken with respect to the larger of the two site splittings (the
        same practical choice Hamada recommends for normalization when one bond has
        the larger quadrupole coupling).
        """
        x_eff = np.asarray(x_eff, dtype=float).reshape(-1)
        split_ref = float(split_ref)
        xi = float(xi)
        Rq = x_eff / split_ref
        return 1.0 + 0.5 * xi * (1.0 + Rq)

    def component_curves(self,
        params,
        x,
        *,
        wd=32.68,
        exact_intensity=True,
        nphi=64,
    ):
        x = np.asarray(x, dtype=float).reshape(-1)
        p = dict(params)
        x_eff = float(p["cc"]) * (x - float(p["center"]))

        comps = self.butanol_absorption_components(
            x_eff,
            p["P"],
            p["split_cd"],
            p["split_od"],
            p["sigma"],
            p["eta_od"],
            p["K"],
            wd=wd,
            eta_cd=p["eta_cd"],
            exact_intensity=exact_intensity,
            nphi=nphi,
        )

        amp = float(p["amp"])
        for key in list(comps.keys()):
            comps[key] = amp * comps[key]

        split_ref = p["split_od"] if abs(p["split_od"]) >= abs(p["split_cd"]) else p["split_cd"]
        gain = self.qmeter_gain(x_eff, split_ref, p["xi"])
        background = self.polynomial_background(x, p["b0"], p["b1"], p["b2"], p["b3"])
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

    def signal_model(self,
        x,
        P,
        amp,
        center,
        cc,
        split_cd,
        split_od,
        sigma,
        eta_od,
        eta_cd,
        K,
        xi,
        b0,
        b1,
        b2,
        b3,
        *,
        wd=32.68,
        exact_intensity=False,
        nphi=64,
    ):
        """
        Full measurable signal model.

        Model
        -----
        1. Apply a global center shift and optional x-axis calibration coefficient:
            x_eff = cc * (x - center)

        2. Build the physical absorption function:
            chi''_but = (1-K) * chi''_CD + K * chi''_OD

        3. Optionally apply the false-asymmetry Q-meter gain correction:
            D = 1 + 0.5 * xi * (1 + R)

        4. Add a cubic residual background.

        ``amp`` is allowed to be either positive or negative, so the script can fit
        spectra that have or have not been sign-flipped.
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        x_eff = float(cc) * (x - float(center))

        comps = self.butanol_absorption_components(
            x_eff,
            P,
            split_cd,
            split_od,
            sigma,
            eta_od,
            K,
            wd=wd,
            eta_cd=eta_cd,
            exact_intensity=exact_intensity,
            nphi=nphi,
        )

        split_ref = split_od if abs(split_od) >= abs(split_cd) else split_cd
        gain = self.qmeter_gain(x_eff, split_ref, xi)
        background = self.polynomial_background(x, b0, b1, b2, b3)
        return float(amp) * comps["absorption"] * gain + background


    def fit_dulya(self,
        x, 
        y, 
        yerr, 
        params, 
        bounds=None,
        method="Powell",
        max_nfev=50000,
    ):
        """Fit the Dulya model to the data."""
        _, fixed_params = self.split_fit_params(params)
        x0 = self.params_to_vector(params)
        scipy_bounds = None
        if bounds is not None:
            scipy_bounds = [bounds[name] for name in self.param_keys]

        result = minimize(
            self.chi_squared,
            x0,
            args=(x, y, yerr, fixed_params),
            method=method,
            bounds=scipy_bounds,
        )
        fitted_params = self.vector_to_params(result.x)
        fitted_params.update(fixed_params)
        return fitted_params
