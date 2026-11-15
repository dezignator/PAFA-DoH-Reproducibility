"""Clipped, noised client-update aggregation, thesis Eq. (36).

Eq. (36) as typeset:   W <- W - eta * ( (1/K) sum clip(g_k) + N(0, s^2 C^2 I) )
puts the noise outside the 1/K average. With client-level sensitivity C/K the
effective noise multiplier is then s*K, not s, and Eq. (60) no longer matches.
The accountant-consistent form (McMahan et al. 2018) is
                       W <- W - eta/K * ( sum clip(g_k) + N(0, s^2 C^2 I) )
`noise_inside_average=True` selects it; the flag exists so a run can state
which of the two it used.
"""
import numpy as np


def clip_l2(g, C):
    n = np.linalg.norm(g)
    return g if n <= C else g * (C / n)


def dp_aggregate(updates, C, sigma, rng, noise_inside_average=True):
    """Return the aggregated (noised) mean update, shape = updates[0].shape."""
    U = np.stack([clip_l2(u, C) for u in updates])
    K = U.shape[0]
    noise = rng.normal(0.0, sigma * C, size=U.shape[1:])
    if noise_inside_average:
        return (U.sum(axis=0) + noise) / K
    return U.mean(axis=0) + noise


def effective_noise_multiplier(sigma, K, noise_inside_average=True):
    """Noise std relative to the client-level sensitivity of the average."""
    return sigma if noise_inside_average else sigma * K


def apply_step(W, agg, eta):
    return W - eta * agg
