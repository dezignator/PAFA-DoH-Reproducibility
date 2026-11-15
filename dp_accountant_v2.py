"""Moments/RDP accounting, thesis Eq. (60).

    eps(delta) = min_{alpha>1}  T*alpha/(2*sigma^2) + log(1/delta)/(alpha-1)

This is the un-subsampled Gaussian mechanism composed T times (every client
participates every round). It is NOT the Poisson-subsampled bound of Abadi et
al.; if the run used client sampling, a subsampled accountant is required.
"""
import math
import numpy as np

DEFAULT_ALPHAS = np.concatenate([np.linspace(1.01, 10, 200), np.arange(11, 257)])


def eps_from_rdp(T, sigma, delta, alphas=DEFAULT_ALPHAS):
    if sigma <= 0 or not (0 < delta < 1) or T < 1:
        raise ValueError("need sigma>0, 0<delta<1, T>=1")
    a = np.asarray(alphas, dtype=float)
    eps = T * a / (2.0 * sigma ** 2) + math.log(1.0 / delta) / (a - 1.0)
    i = int(np.argmin(eps))
    return float(eps[i]), float(a[i])


def sigma_for_target(T, eps_target, delta, lo=0.05, hi=1e4, iters=200):
    """Smallest noise multiplier with eps(delta) <= eps_target (bisection)."""
    if eps_from_rdp(T, hi, delta)[0] > eps_target:
        raise ValueError("eps_target unreachable inside search bracket")
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        if eps_from_rdp(T, mid, delta)[0] > eps_target:
            lo = mid
        else:
            hi = mid
    return hi


def eps_curve(T_values, sigma, delta):
    return [(int(T), eps_from_rdp(T, sigma, delta)[0]) for T in T_values]
