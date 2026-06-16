import numpy as np

DEFAULT_CIRC_CONSTS = (
    3e-8,
    0.35,
    619,
    50,
    10,
    0.0343,
    4.752e-9,
    50,
    1.027e-10,
    2.542e-7,
    0,
    0,
    0,
    0,
)


def Baseline(
    f,
    U,
    Cknob,
    eta,
    trim,
    Cstray,
    phi_const,
    DC_offset,
    species: str,
    L0=DEFAULT_CIRC_CONSTS[0],
    Rcoil=DEFAULT_CIRC_CONSTS[1],
    R=DEFAULT_CIRC_CONSTS[2],
    R1=DEFAULT_CIRC_CONSTS[3],
    r=DEFAULT_CIRC_CONSTS[4],
    alpha=DEFAULT_CIRC_CONSTS[5],
    beta1=DEFAULT_CIRC_CONSTS[6],
    Z_cable=DEFAULT_CIRC_CONSTS[7],
    D=DEFAULT_CIRC_CONSTS[8],
    M=DEFAULT_CIRC_CONSTS[9],
    delta_C=DEFAULT_CIRC_CONSTS[10],
    delta_phi=DEFAULT_CIRC_CONSTS[11],
    delta_phase=DEFAULT_CIRC_CONSTS[12],
    delta_l=DEFAULT_CIRC_CONSTS[13],
):
    pi = np.pi
    im_unit = 1j
    sign = 1
    span = 6  ## default span for now

    I = U * 1000 / R  # Ideal constant current, mA

    if species == "proton":
        w_res = 2 * pi * 213e6
        w_low = 2 * pi * (213 - span) * 1e6
        w_high = 2 * pi * (213 + span) * 1e6
        delta_w = 2 * pi * 4e6 / 500
    elif species == "deuteron":
        w_res = 2 * pi * 32.68e6
        w_low = 2 * pi * (32.68 - span) * 1e6
        w_high = 2 * pi * (32.68 + span) * 1e6
        delta_w = 2 * pi * 4e6 / 500
    else:
        raise ValueError(f"Invalid species: {species}. Choose 'proton' or 'deuteron'.")

    # Convert frequency to angular frequency (rad/s)
    w = 2 * pi * f * 1e6

    # Functions
    def slope():
        return delta_C / (0.25 * 2 * pi * 1e6)

    def slope_phi():
        return delta_phi / (0.25 * 2 * pi * 1e6)

    def Ctrim(w):
        return slope() * (w - w_res)

    def Cmain():
        return 20 * 1e-12 * Cknob

    def C(w):
        return Cmain() + Ctrim(w) * 1e-12

    def Z0(w):
        S = 2 * Z_cable * alpha
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.sqrt((S + w * M * im_unit) / (w * D * im_unit))
        return np.where(w == 0, 0, result)  # Avoid invalid values for w=0

    def beta(w):
        return beta1 * w

    def gamma(w):
        return alpha + beta(w) * 1j  # Create a complex number using numpy

    def ZC(w):
        Cw = C(w)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(Cw != 0, 1 / (im_unit * w * Cw), 0)
        return np.where(w == 0, 0, result)  # Avoid invalid values for w=0

    def vel(w):
        return 1 / beta(w)

    def l(w):
        return trim * vel(w_res) + delta_l

    def ic(w):
        return 0.11133

    def chi(w):
        return np.zeros_like(w)  # Placeholder for x1(w) and x2(w)

    def pt(w):
        return ic(w)

    def L(w):
        return L0 * (1 + sign * 4 * pi * eta * pt(w) * chi(w))

    def ZLpure(w):
        return im_unit * w * L(w) + Rcoil

    def Zstray(w):
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(Cstray != 0, 1 / (im_unit * w * Cstray), 0)
        return np.where(w == 0, 0, result)  # Avoid invalid values for w=0

    def ZL(w):
        return ZLpure(w) * Zstray(w) / (ZLpure(w) + Zstray(w))

    def ZT(w):
        epsilon = 1e-10  # Small constant to avoid division by zero
        return Z0(w) * (ZL(w) + Z0(w) * np.tanh(gamma(w) * l(w))) / (
            Z0(w) + ZL(w) * np.tanh(gamma(w) * l(w)) + epsilon
        )

    def Zleg1(w):
        return r + ZC(w) + ZT(w)

    def Ztotal(w):
        return R1 / (1 + (R1 / Zleg1(w)))

    def parfaze(w):
        yp1 = 0
        yp2 = delta_phase
        yp3 = 0

        a = (
            (yp1 - yp2) * (w_low - w_high) - (yp1 - yp3) * (w_low - w_res)
        ) / (
            ((w_low**2) - (w_res**2)) * (w_low - w_high)
            - ((w_low**2) - (w_high**2)) * (w_low - w_res)
        )
        bb = (yp1 - yp3 - a * ((w_low**2) - (w_high**2))) / (w_low - w_high)
        c = yp1 - a * (w_low**2) - bb * w_low
        return a * w**2 + bb * w + c

    def phi_trim(w):
        return slope_phi() * (w - w_res) + parfaze(w)

    def phi(w):
        return phi_trim(w) + phi_const

    def V_out(w):
        return -1 * (I * Ztotal(w) * np.exp(im_unit * phi(w) * pi / 180))

    out_y = V_out(w)
    offset = np.array([x - min(out_y.real) for x in out_y.real])

    return offset.real + DC_offset
