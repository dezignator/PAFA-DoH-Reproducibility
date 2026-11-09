# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# src/baselines/rule_based.py — PAFA-DoH Reproducibility Artifact
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
# The rule-based detector against which PAFA-DoH is compared in Table 2
# and §4.30.3. The chapter reports 78 % detection and a false-positive
# rate above 5 % for this baseline on the same data as PAFA-DoH.
#
# The rules are the ones stated in §4.30.3 and §4.7:
#
#   DNS tunneling:         entropy > 4.2 bits
#                          autocorr(inter_query, lag=5) > 0.8
#   C2 beaconing:          regular_beaconing_interval > 0.9
#   Side channel:          timing_variance > threshold
#
# This is a deterministic detector: no training, no model. It exists
# to anchor the "traditional detection" comparison in §4.30.3 and to
# make the argument that rule-based detection is not sufficient for
# adaptive adversaries (§4.30.3, final paragraph).
#
# ───────────────────────────────────────────────────────────────────────────
# Where the pieces come from in Chapter 4
# ───────────────────────────────────────────────────────────────────────────
#   Reported detection rate (78 %)   → §4.30.3
#   Reported FPR (> 5 %)             → §4.30.3
#   Entropy rule (4.2 bits)          → §4.7 / §4.30.3
#   Autocorr rule (lag 5 > 0.8)      → §4.7 / Eq. 100
#   Beaconing rule (> 0.9)           → §4.26.4
#   Config block                     → configs/baselines_config.yaml
#                                      (File 9g, `rule_based:`)
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..utils.metrics import accuracy, fpr

log = logging.getLogger(__name__)


# ─── Reported baseline metrics (from §4.30.3) ───────────────────────────
REPORTED_RULE_BASED_ACCURACY = 0.78
REPORTED_RULE_BASED_FPR = 0.05  # "above 5 %"; the artifact reports the
                                # observed value, not the threshold.


# ─── The rules ──────────────────────────────────────────────────────────

@dataclass
class RuleThresholds:
    """The threshold values that define the rule-based detector.

    Every threshold is sourced from §4.30.3 or §4.7 of the thesis.
    Changing one of them changes the detector's operating point; the
    reported 78 % / >5 % FPR figures correspond to the values below.
    """
    entropy_threshold: float = 4.2
    autocorr_lag5_threshold: float = 0.8
    beaconing_interval_threshold: float = 0.9
    timing_variance_threshold: float = 15.0  # ms² — see §4.26.1


class RuleBasedBaseline:
    """Deterministic rule-based detector for DoH traffic.

    Usage:

        detector = RuleBasedBaseline()
        pred = detector.predict(test_df)
        acc = accuracy(test_df["label"], pred)
        fpr_value = fpr(test_df["label"], pred)
    """

    def __init__(self, thresholds: RuleThresholds | None = None):
        self.thresholds = thresholds or RuleThresholds()

    # ── Single-feature rules ─────────────────────────────────────────

    def _rule_entropy(self, df: pd.DataFrame) -> np.ndarray:
        """DNS-tunneling entropy rule (§4.7, §4.30.3)."""
        if "entropy_subdomain" not in df.columns:
            return np.zeros(len(df), dtype=np.int64)
        return (df["entropy_subdomain"] > self.thresholds.entropy_threshold).astype(np.int64).to_numpy()

    def _rule_autocorr(self, df: pd.DataFrame) -> np.ndarray:
        """DNS-tunneling autocorrelation rule (§4.7, Eq. 100)."""
        if "autocorr_lag5" not in df.columns:
            return np.zeros(len(df), dtype=np.int64)
        return (df["autocorr_lag5"] > self.thresholds.autocorr_lag5_threshold).astype(np.int64).to_numpy()

    def _rule_beaconing(self, df: pd.DataFrame) -> np.ndarray:
        """C2 beaconing rule (§4.26.4).

        Uses `iat_std_ms / iat_mean_ms` as a proxy for beaconing
        regularity: a perfectly periodic beacon has ratio → 0, an
        irregular flow has ratio → 1. The §4.26.4 threshold is
        `regular_beaconing_interval > 0.9`, which corresponds to the
        ratio-based rule below with `regularity = 1 − ratio`.
        """
        if not {"iat_mean_ms", "iat_std_ms"}.issubset(df.columns):
            return np.zeros(len(df), dtype=np.int64)
        mean = df["iat_mean_ms"].replace(0, np.nan)
        ratio = (df["iat_std_ms"] / mean).fillna(1.0)
        regularity = 1.0 - ratio
        return (regularity > self.thresholds.beaconing_interval_threshold).astype(np.int64).to_numpy()

    def _rule_timing_variance(self, df: pd.DataFrame) -> np.ndarray:
        """Side-channel timing-variance rule (§4.26.1)."""
        if "iat_std_ms" not in df.columns:
            return np.zeros(len(df), dtype=np.int64)
        variance = df["iat_std_ms"] ** 2
        return (variance > self.thresholds.timing_variance_threshold).astype(np.int64).to_numpy()

    # ── Combined detector ─────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Run all four rules and OR the results.

        A flow is classified malicious if *any* rule fires. This is
        the standard "OR of independent signatures" rule-base design
        that §4.30.3 contrasts with PAFA-DoH's learned detection.

        Returns:
            (n,) int64 array of 0 / 1 predictions.
        """
        if "label" not in df.columns:
            raise ValueError(
                "RuleBasedBaseline.predict requires a `label` column "
                "(the detector is not trained; the labels are used only "
                "for consistency checks)."
            )
        r_ent = self._rule_entropy(df)
        r_acf = self._rule_autocorr(df)
        r_bcn = self._rule_beaconing(df)
        r_tim = self._rule_timing_variance(df)
        combined = np.maximum.reduce([r_ent, r_acf, r_bcn, r_tim])
        # Log which rules fired, for the run log.
        log.debug(
            "Rule fires: entropy=%d, autocorr=%d, beaconing=%d, timing=%d, "
            "combined=%d",
            int(r_ent.sum()), int(r_acf.sum()), int(r_bcn.sum()),
            int(r_tim.sum()), int(combined.sum()),
        )
        return combined.astype(np.int64)

    def evaluate(self, df: pd.DataFrame) -> dict[str, float]:
        """Run the rules and return accuracy and FPR.

        Returns:
            {"accuracy": float, "fpr": float} — the two numbers that
            §4.30.3 reports as 78 % and > 5 % respectively.
        """
        y_pred = self.predict(df)
        y_true = df["label"].to_numpy(dtype=np.int64)
        return {
            "accuracy": float(accuracy(y_true, y_pred)),
            "fpr": float(fpr(y_true, y_pred)),
        }
