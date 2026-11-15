"""Hypothesis-test helpers for Sec 4.8.1 (Eq. 42) and Table 16 validation tests."""
import numpy as np
from scipy import stats


def permutation_pvalue(a, b, stat=lambda x, y: abs(np.mean(x) - np.mean(y)),
                       n_resamples=10000, seed=0):
    """p = Pr(T >= t_obs | H0), Eq. (42), by label permutation.
    Add-one correction so p is never exactly 0. n_resamples and seed must be
    recorded in the run manifest (Sec 4.8.1 refers to them)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    t_obs = stat(a, b)
    pool = np.concatenate([a, b])
    na = len(a)
    hits = 0
    for _ in range(n_resamples):
        rng.shuffle(pool)
        if stat(pool[:na], pool[na:]) >= t_obs:
            hits += 1
    return (1 + hits) / (1 + n_resamples), t_obs


def bonferroni(pvals, alpha=0.05):
    """Sec 4.8.1: corrected level alpha/m across the m features."""
    m = len(pvals)
    thr = alpha / m
    return thr, [p <= thr for p in pvals]


def welch(a, b):
    return stats.ttest_ind(a, b, equal_var=False)


def mannwhitney(a, b):
    return stats.mannwhitneyu(a, b, alternative="two-sided")


def bootstrap_ci(x, stat=np.mean, level=0.95, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    s = np.array([stat(rng.choice(x, len(x))) for _ in range(n_boot)])
    lo, hi = np.percentile(s, [(1 - level) * 50, 100 - (1 - level) * 50])
    return float(lo), float(hi)


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp)
