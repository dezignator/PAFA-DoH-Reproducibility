# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# src/utils/metrics.py — PAFA-DoH Reproducibility Artifact
# ═══════════════════════════════════════════════════════════════════════════
#
# Thesis chapter: Chapter 4 (PAFA-DoH)
# Citation DOI:   10.1016/j.neunet.2025.108343
# License:        MIT
# Artifact version: 1.0.0
#
# ───────────────────────────────────────────────────────────────────────────
# Purpose
# ───────────────────────────────────────────────────────────────────────────
# The evaluation metrics reported in Chapter 4 (Table 16 and Table 17).
# Every function is a small, dependency-light implementation of a
# standard metric. There is deliberately NO sklearn import at module
# level; the metric definitions here are what the artifact uses, so a
# reviewer can read them without needing scikit-learn installed.
#
# The eight metrics:
#
#   accuracy   — Eq. 64
#   fpr        — Eq. 65
#   precision  — standard
#   recall     — standard
#   f1         — standard
#   auc_roc    — via a trapezoidal PR/ROC approximation; see docstring
#   cohens_d   — pooled effect size (used by the toy generator self-check)
#   bootstrap_ci — 95% CI on the mean via bootstrap resampling
#
# Table 16 uses `accuracy` and `fpr` at the reported 99.4% / 0.01%
# operating point. Table 17 uses `cohens_d` as a sanity metric for the
# synthetic toy data (`make_toy_dataset.py`, File 10f).
#
# `bootstrap_ci` implements the ±0.12% CI on Table 16's accuracy. The
# committed expected value was produced with n_bootstrap=1000 and
# bootstrap_seed=42, matching `configs/default_pafa_config.yaml::
# evaluation.confidence_intervals`.
#
# ───────────────────────────────────────────────────────────────────────────
# On the choice NOT to use scikit-learn
# ───────────────────────────────────────────────────────────────────────────
# scikit-learn is pinned in requirements.txt (File 4) and is used by
# the baselines (`src/baselines/centralized_ml.py`, File 13b), which
# need estimators, cross-validation, and its `roc_auc_score`. But the
# metrics in this file are so small — a dozen lines each — that
# duplicating them here keeps the artifact's evaluation layer
# self-contained and free of version drift. In particular,
# `sklearn.metrics.roc_auc_score` changed its tie-breaking between
# releases; the artifact's expected Table 16 values were produced with
# a specific tie-breaking convention, and reimplementing the metric
# here pins that convention.
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import numpy as np


# ─── Confusion-matrix primitives ─────────────────────────────────────────

def _as_int_array(y: np.ndarray | list[int]) -> np.ndarray:
    arr = np.asarray(y).astype(np.int64).ravel()
    unique = np.unique(arr)
    if not np.isin(unique, [0, 1]).all():
        raise ValueError(
            f"Expected binary labels (0 or 1); got unique values {unique.tolist()}."
        )
    return arr


def _as_score_array(s: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(s).astype(np.float64).ravel()
    if arr.size == 0:
        raise ValueError("Score array is empty.")
    if not np.isfinite(arr).all():
        raise ValueError("Score array contains non-finite values.")
    return arr


def confusion_counts(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
) -> tuple[int, int, int, int]:
    """Return (TP, TN, FP, FN)."""
    y_true = _as_int_array(y_true)
    y_pred = _as_int_array(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}."
        )
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


# ─── The eight metrics ──────────────────────────────────────────────────

def accuracy(y_true, y_pred) -> float:
    """Classification accuracy (Eq. 64 of the thesis).

    ``(TP + TN) / (TP + TN + FP + FN)``.
    """
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    total = tp + tn + fp + fn
    if total == 0:
        raise ValueError("Empty label array.")
    return (tp + tn) / total


def fpr(y_true, y_pred) -> float:
    """False-positive rate (Eq. 65).

    ``FP / (FP + TN)``. Returns 0.0 when there are no negatives (a
    convention shared with scikit-learn; the alternative — raising —
    would make fpr() unusable in a mixed loop).
    """
    _, tn, fp, _ = confusion_counts(y_true, y_pred)
    denom = fp + tn
    return float(fp / denom) if denom else 0.0


def precision(y_true, y_pred) -> float:
    """``TP / (TP + FP)``. Returns 0.0 when there are no positive predictions."""
    tp, _, fp, _ = confusion_counts(y_true, y_pred)
    denom = tp + fp
    return float(tp / denom) if denom else 0.0


def recall(y_true, y_pred) -> float:
    """``TP / (TP + FN)``. Returns 0.0 when there are no positives."""
    tp, _, _, fn = confusion_counts(y_true, y_pred)
    denom = tp + fn
    return float(tp / denom) if denom else 0.0


def f1(y_true, y_pred) -> float:
    """Harmonic mean of precision and recall. Returns 0.0 if either is 0."""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p + r == 0:
        return 0.0
    return 2.0 * p * r / (p + r)


def auc_roc(y_true, y_score) -> float:
    """Area under the ROC curve.

    Computed by sorting scores descending, then integrating TPR over
    FPR using the trapezoidal rule. Ties in `y_score` are broken by
    the order in which they appear in the input array — a stable
    convention, so two runs with the same input give the same AUC.

    Returns:
        The AUC in [0, 1]. Returns 0.5 (the chance level) when the
        label array has only one class.

    Raises:
        ValueError: if the label array has more than two classes.
    """
    y_true = _as_int_array(y_true)
    y_score = _as_score_array(y_score)
    if y_true.shape != y_score.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_score {y_score.shape}."
        )
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Stable sort preserves input order among equal scores.
    order = np.argsort(-y_score, kind="stable")
    y_true_sorted = y_true[order]
    # Cumulative TP and FP counts at each threshold (inclusive).
    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)
    tpr = tps / n_pos
    fpr_arr = fps / n_neg
    # Prepend (0, 0) so the trapezoid is closed at the origin.
    tpr = np.concatenate([[0.0], tpr])
    fpr_arr = np.concatenate([[0.0], fpr_arr])
    return float(np.trapz(tpr, fpr_arr))


def cohens_d(x, y) -> float:
    """Pooled Cohen's d effect size between two samples.

    ``(mean(x) - mean(y)) / pooled_std``, where ``pooled_std`` uses
    the sample variance with ddof=1. Returns 0.0 if the pooled std is
    0 (identical-constant samples). The sign is (x - y), so a positive
    value means `x` is larger than `y`.

    Used by `data/make_toy_dataset.py` (File 10f) to assert that the
    synthetic toy data is neither class-blind nor trivially separable.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    sx2 = x.var(ddof=1)
    sy2 = y.var(ddof=1)
    pooled = np.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2))
    if pooled == 0:
        return 0.0
    return float((x.mean() - y.mean()) / pooled)


def bootstrap_ci(
    values,
    *,
    n: int = 1000,
    level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval on the mean.

    Draws `n` resamples with replacement, computes the mean of each,
    and returns the `(1 - level) / 2` and `(1 + level) / 2` quantiles
    as the interval. Also returns the point estimate (the input mean).

    Args:
        values: 1-D array of observations.
        n: Number of bootstrap resamples (default 1000, matching
            `configs/default_pafa_config.yaml`).
        level: Confidence level (default 0.95).
        seed: Bootstrap RNG seed (default 42). Must match the config
            seed for the committed expected outputs to reproduce.

    Returns:
        (low, high, point_estimate).

    Notes:
        The bootstrap is the percentile method; it does not correct
        for bias. For the ±0.12% CI on 99.4% accuracy in Table 16,
        the percentile method matches the committed expected value
        within the tolerance in `REPRODUCIBILITY_CHECKLIST.md` §8.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty array.")
    if not (0 < level < 1):
        raise ValueError(f"level must be in (0, 1); got {level}.")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n, values.size))
    means = values[idx].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1.0 - alpha))
    return lo, hi, float(values.mean())
