import numpy as np

def subtract_polynomial_wings(
    x: np.ndarray,
    signal: np.ndarray,
    edge_fraction: float,
    degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:

    """Fit a polynomial to the outer wings and return the detrended signal."""

    n_bins = len(x)
    n_edge = max(degree + 1, int(n_bins * edge_fraction))
    wing_mask = np.zeros(n_bins, dtype=bool)
    wing_mask[:n_edge] = True
    wing_mask[-n_edge:] = True

    coeffs = np.polyfit(x[wing_mask], signal[wing_mask], deg=degree)
    polynomial = np.polyval(coeffs, x)
    detrended = signal - polynomial
    chi_squared = np.sum((signal[wing_mask] - polynomial[wing_mask]) ** 2 / np.std(signal[wing_mask])**2)
    return detrended, polynomial, wing_mask, coeffs, chi_squared