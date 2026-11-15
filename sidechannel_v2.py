"""Side-channel masking arithmetic: thesis Eqs. (47)-(49).

Synthetic-trace tooling only. The 98 % / 60 % SNR reductions in Sec 4.14 come
from ChipWhisperer captures; this module defines the quantities so a capture
set can be scored the same way, and runs a synthetic sanity check.
"""
import numpy as np


def snr_ratio(s, n):
    """||s||_2 / ||n||_2 as defined for Eq. (47)."""
    return float(np.linalg.norm(s) / np.linalg.norm(n))


def snr_reduction_pct(ratio_baseline, ratio_injected):
    """Relative reduction of the Eq. (47) ratio, in percent."""
    return 100.0 * (1.0 - ratio_injected / ratio_baseline)


def meets_objective(s, n, delta):
    return snr_ratio(s, n) <= delta


def adaptive_threshold(v_th0, beta, T_t):
    """Eq. (48): V_th(t) = V_th0 + beta*T(t), T in [0,1], beta >= 0."""
    T_t = np.asarray(T_t, float)
    if np.any((T_t < 0) | (T_t > 1)):
        raise ValueError("T(t) must lie in [0,1]")
    if beta < 0:
        raise ValueError("beta must be >= 0")
    return v_th0 + beta * T_t


def stdp_error_trace(delta_w, sigma_k, rng):
    """Eq. (49): E(t) = sum_{t'<=t} Theta(t') xi(t'),  Theta(t)=Theta(t-1)+dw(t),
    xi ~ N(0, sigma_k^2)."""
    theta = np.cumsum(np.asarray(delta_w, float))
    xi = rng.normal(0.0, sigma_k, size=theta.shape)
    return np.cumsum(theta * xi), theta


def noise_to_hit_target(s, noise_unit, delta):
    """Scale a fixed noise shape so that ||s||/||n|| == delta (diagnostic)."""
    return noise_unit * (np.linalg.norm(s) / (delta * np.linalg.norm(noise_unit)))
