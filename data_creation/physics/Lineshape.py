import numpy as np
from scipy.integrate import trapezoid
from scipy.special import wofz
from scipy.signal import hilbert

def Baseline(f, U, Cknob, eta, trim, Cstray, phi_const, DC_offset, species: str):
    # Preamble
    circ_consts = (3*10**(-8), 0.35, 619, 50, 10, 0.0343, 4.752*10**(-9), 50, 1.027*10**(-10), 2.542*10**(-7), 0, 0, 0, 0)
    pi = np.pi
    im_unit = 1j  
    sign = 1
    span = 6 ## default span for now

    # Main constants
    L0, Rcoil, R, R1, r, alpha, beta1, Z_cable, D, M, delta_C, delta_phi, delta_phase, delta_l = circ_consts

    I = U*1000/R  # Ideal constant current, mA

    if species == 'proton':
        w_res = 2 * pi * 213e6
        w_low = 2 * pi * (213 - span) * 1e6
        w_high = 2 * pi * (213 + span) * 1e6
        delta_w = 2 * pi * 4e6 / 500
    elif species == 'deuteron':
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
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.sqrt((S + w * M * im_unit) / (w * D * im_unit))
        return np.where(w == 0, 0, result)  # Avoid invalid values for w=0

    def beta(w):
        return beta1 * w

    def gamma(w):
        return alpha + beta(w) * 1j  # Create a complex number using numpy

    def ZC(w):
        Cw = C(w)
        with np.errstate(divide='ignore', invalid='ignore'):
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
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(Cstray != 0, 1 / (im_unit * w * Cstray), 0)
        return np.where(w == 0, 0, result)  # Avoid invalid values for w=0

    def ZL(w):
        return ZLpure(w) * Zstray(w) / (ZLpure(w) + Zstray(w))

    def ZT(w):
        epsilon = 1e-10  # Small constant to avoid division by zero
        return Z0(w) * (ZL(w) + Z0(w) * np.tanh(gamma(w) * l(w))) / (Z0(w) + ZL(w) * np.tanh(gamma(w) * l(w)) + epsilon)

    def Zleg1(w):
        return r + ZC(w) + ZT(w)

    def Ztotal(w):
        return R1 / (1 + (R1 / Zleg1(w)))

    def parfaze(w):
        yp1 = 0
        yp2 = delta_phase
        yp3 = 0

        a = ((yp1 - yp2) * (w_low - w_high) - (yp1 - yp3) * (w_low - w_res)) / \
            (((w_low ** 2) - (w_res ** 2)) * (w_low - w_high) - ((w_low ** 2) - (w_high ** 2)) * (w_low - w_res))
        bb = (yp1 - yp3 - a * ((w_low ** 2) - (w_high ** 2))) / (w_low - w_high)
        c = yp1 - a * (w_low ** 2) - bb * w_low
        return a * w ** 2 + bb * w + c

    def phi_trim(w):
        return slope_phi() * (w - w_res) + parfaze(w)

    def phi(w):
        return phi_trim(w) + phi_const

    def V_out(w):
        return -1 * (I * Ztotal(w) * np.exp(im_unit * phi(w) * pi / 180))

    out_y = V_out(w)
    offset = np.array([x - min(out_y.real) for x in out_y.real])
    
    return offset.real + DC_offset

def Lineshape(x,eps, eta, phi, g):

    def bigy(eta,phi):
        return np.sqrt(3 - eta*np.cos(2*phi))
    
    def cosal(x, eps):
        return (1 - eps * x - eta*np.cos(2*phi)) / bigxsquare(x, eps)

    def c(x):
        return np.sqrt(np.sqrt(g**2 + (1 - x - eta*np.cos(2*phi))**2))

    def bigxsquare(x, eps):
        return np.sqrt(g**2 + (1 - eps * x - eta*np.cos(2*phi))**2)

    def mult_term(x, eps):
        return 1 / (2 * np.pi * np.sqrt(bigxsquare(x, eps)))

    def cosaltwo(x, eps):
        return np.sqrt((1 + cosal(x, eps)) / 2)

    def sinaltwo(x, eps):
        return np.sqrt((1 - cosal(x, eps)) / 2)

    def termone(x, eps):
        return np.pi / 2 + np.arctan((bigy(eta,phi)**2 - bigxsquare(x, eps)) / (2 * bigy(eta,phi) * np.sqrt(bigxsquare(x, eps)) * sinaltwo(x, eps)))

    def termtwo(x, eps):
        return np.log((bigy(eta,phi)**2 + bigxsquare(x, eps) + 2 * bigy(eta,phi) * np.sqrt(bigxsquare(x, eps)) * cosaltwo(x, eps)) /
                    (bigy(eta,phi)**2 + bigxsquare(x, eps) - 2 * bigy(eta,phi) * np.sqrt(bigxsquare(x, eps)) * cosaltwo(x, eps)))

    def icurve(x, eps):
        return mult_term(x, eps) * (2 * cosaltwo(x, eps) * termone(x, eps) + sinaltwo(x, eps) * termtwo(x, eps))
    
    return icurve(x,eps)/10



def GenerateVectorLineshape(P,x, CC, eta, phi, g):

    r = (np.sqrt(4-3*P**(2))+P)/(2-2*P)
    
    if P > 0:
        Iplus = r*Lineshape(x,1, eta, phi, g)
        Iminus = Lineshape(x,-1, eta, phi, g)
        r = r
    else:
        r = 1/r
        Iplus = -Lineshape(x,1, eta, phi, g)
        Iminus = -r*Lineshape(x,-1, eta, phi, g)

    ### Scaling
    pSummed = np.sum(Iplus + Iminus)
    deltaP = P/pSummed*CC
    Iplus = Iplus*deltaP
    Iminus = Iminus*deltaP
    signal = Iplus + Iminus

    return signal,Iplus,Iminus

def SamplingVectorLineshape(P, x, bound, CC, eta, phi, g):
    """Sampling the lineshape with a stochastic shift to frequency bins.

    Args:
        P (float): Polarization
        x (list): Frequency range
        bound (float): Bound of the shift
        CC (float): Calibration coefficient
        eta (float): eta parameter
        phi (float): Phase angle in degrees
        g (float): g parameter

    Returns:
        signal (list): Generated lineshape with a stochastic shift
    """
    shift = np.full(len(x),np.random.uniform( -bound , bound))
    x += shift
    ### Generate the lineshape with the shifted 
    signal, _, _ = GenerateVectorLineshape(P,x, CC, eta, phi, g)
    return signal

def GenerateTensorLineshape(x, P, phi_deg, eta, phi, g):
    """
    Calculate the total signal for given x, polarization P, and phase angle phi.
    
    Parameters:
    -----------
    x : float or array-like
        The x-coordinate value(s)
    P : float
        Input polarization (between 0 and 1)
    phi_deg : float
        Phase angle in degrees
        
    Returns:
    --------
    float or array-like
        The total signal value(s)
    """
    # System parameters
    g = 0.05
    s = 0.04
    bigy = np.sqrt(3 - s)

    # x = (x - 32.68) / 0.6
    
    # Calculate r from P
    r = (np.sqrt(4 - 3 * P**2) + P) / (2 - 2 * P)
    
    # Convert phase to radians
    phi_rad = np.deg2rad(phi_deg)
    
    # Calculate absorptive signals
    yvals_absorp1 = Lineshape(x, 1, eta, phi, g)        # χ''₊
    yvals_absorp2 = Lineshape(-x, 1, eta, phi, g)       # χ''₋
    
    # Calculate dispersive signals using Hilbert transform
    yvals_disp1 = np.imag(hilbert(yvals_absorp1))  # χ'₊
    yvals_disp2 = np.imag(hilbert(yvals_absorp2))  # χ'₋
    
    # Calculate phase-sensitive linear combination
    Iplus = r * (yvals_absorp1 * np.sin(phi_rad) + yvals_disp1 * np.cos(phi_rad))
    Iminus = yvals_absorp2 * np.sin(phi_rad) + yvals_disp2 * np.cos(phi_rad)

    signal = Iplus + Iminus
    
    # Return total signal
    return signal, Iplus, Iminus

def SamplingTensorLineshape(P, x, bound, eta, phi, g):
    """Sampling the lineshape with a stochastic shift to frequency bins.

    Args:
        P (float): Polarization
        x (list): Frequency range
        bound (float): Bound of the shift
        phi (float): Phase angle in degrees

    Returns:
        signal (list): Generated lineshape with a stochastic shift
    """
    shift = np.full(len(x),np.random.uniform( -bound , bound))
    x += shift
    ### Generate the lineshape with the shifted 
    signal, _, _ = GenerateTensorLineshape(x, P, phi, eta, phi, g)
    return signal

def Voigt(x, amp, s, g, x0):
    """
    Voigt profile function with an adjustable center (x0).
    
    :param x: Array of x values
    :param amp: Amplitude of the Voigt profile
    :param s: Width of the Gaussian component (sigma)
    :param g: Width of the Lorentzian component (gamma)
    :param x0: Center of the Voigt profile
    :return: Voigt profile values
    """
    z = (x - x0 + 1j * g) / (s * np.sqrt(2.0))
    v = wofz(z)  # Faddeeva function for Voigt profile
    out = amp * (np.real(v) / (s * np.sqrt(2 * np.pi)))
    return out

def generate_proton_signal(x, target_area=None):
    """Generate a proton signal using Voigt profile. Returns (signal, area). Area is ∫ y dx over x (MHz).

    When ``target_area`` is set, the Voigt shape parameters are sampled randomly and the
    profile is scaled so its integrated area matches ``target_area`` (for MC polarization sampling).
    """
    sig = 0.1 + np.random.uniform(-0.009, 0.001)
    gam = 0.1 + np.random.uniform(-0.009, 0.001)
    amp = 0.005 + np.random.uniform(-0.0061, 0.003)
    center = 213 + np.random.uniform(-0.1, 0.1)

    y = Voigt(x, amp, sig, gam, center)
    area = float(trapezoid(y, x))
    if target_area is not None:
        if abs(area) < 1e-15:
            raise ValueError("Degenerate proton Voigt area; cannot scale to target_area")
        y = y * (float(target_area) / area)
        area = float(target_area)
    return y, area