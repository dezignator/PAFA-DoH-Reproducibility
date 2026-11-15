"""Detection metrics: Eqs. (63)-(65), threshold selection of Sec 4.26.1, and
sample-size arithmetic for the rare-event claims (FPR < 0.01 %)."""
import math
import numpy as np


def confusion(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true).astype(bool), np.asarray(y_pred).astype(bool)
    return (int((y_true & y_pred).sum()), int((~y_true & ~y_pred).sum()),
            int((~y_true & y_pred).sum()), int((y_true & ~y_pred).sum()))  # TP TN FP FN


def accuracy(tp, tn, fp, fn):            # Eq. (64)
    return (tp + tn) / (tp + tn + fp + fn)


def fpr(tp, tn, fp, fn):                 # Eq. (65)
    return fp / (fp + tn) if (fp + tn) else float("nan")


def tpr(tp, tn, fp, fn):
    return tp / (tp + fn) if (tp + fn) else float("nan")


def roc(scores, y_true):
    """Eq. (63): TPR/FPR over all thresholds gamma on S_anom (flag if S > gamma)."""
    s = np.asarray(scores, float)
    y = np.asarray(y_true).astype(bool)
    g = np.concatenate([[np.inf], np.sort(np.unique(s))[::-1], [-np.inf]])
    P, N = y.sum(), (~y).sum()
    T = np.array([((s > t) & y).sum() / P for t in g])
    F = np.array([((s > t) & ~y).sum() / N for t in g])
    return g, T, F


def best_threshold(scores, y_true, lam=1.0):
    """gamma* = argmax TPR - lambda*FPR  (Sec 4.26.1)."""
    g, T, F = roc(scores, y_true)
    j = int(np.argmax(T - lam * F))
    return float(g[j]), float(T[j]), float(F[j])


def auc(scores, y_true):
    g, T, F = roc(scores, y_true)
    return float(np.trapezoid(T, F)) if hasattr(np, "trapezoid") else float(np.trapz(T, F))


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def clopper_pearson_upper(k, n, conf=0.95):
    from scipy.stats import beta
    return 1.0 if k >= n else float(beta.ppf(conf, k + 1, n - k))


def min_negatives_for_fpr_bound(bound, conf=0.95, fp=0):
    """Smallest benign-sample count N for which an observed FP count `fp`
    certifies FPR <= bound at confidence `conf` (Clopper-Pearson).
    fp=0 gives the 'rule of three' N ~= 3/bound."""
    lo, hi = max(1, fp + 1), 10
    while clopper_pearson_upper(fp, hi, conf) > bound:
        hi *= 2
        if hi > 10**12:
            raise ValueError("bound too tight")
    lo = max(lo, hi // 2)
    while lo < hi:
        mid = (lo + hi) // 2
        if clopper_pearson_upper(fp, mid, conf) <= bound:
            hi = mid
        else:
            lo = mid + 1
    return lo
