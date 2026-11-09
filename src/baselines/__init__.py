# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# src/baselines/__init__.py — PAFA-DoH Reproducibility Artifact
# ═══════════════════════════════════════════════════════════════════════════
#
# Thesis chapter: Chapter 4 (PAFA-DoH)
# Citation DOI:   10.1016/j.neunet.2025.108343
# License:        MIT
# Artifact version: 1.0.0
#
# ───────────────────────────────────────────────────────────────────────────
# Lazy exports for the two experimental baselines against which
# PAFA-DoH is compared in Table 2 and §4.30:
#
#   CentralizedMLBaseline  — §4.30.1 (CNN-LSTM, 92.1 % accuracy, 1.8 % FPR)
#   RuleBasedBaseline      — §4.30.3 (entropy + autocorr rules, 78 %
#                                       detection, > 5 % FPR)
#
# Both are retrained on the same data, with the same preprocessing and
# under the same tuning budget as PAFA-DoH. They are the "experimental
# baselines" of REPRODUCIBILITY_CHECKLIST.md §9, and their reported
# numbers appear in `configs/baselines_config.yaml` (File 9g).
#
# Importing this module does NOT pull in torch or scikit-learn; each
# class is loaded only when first accessed.
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

__all__ = [
    "CentralizedMLBaseline",
    "RuleBasedBaseline",
]


def __getattr__(name: str):
    if name == "CentralizedMLBaseline":
        from .centralized_ml import CentralizedMLBaseline
        return CentralizedMLBaseline
    if name == "RuleBasedBaseline":
        from .rule_based import RuleBasedBaseline
        return RuleBasedBaseline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
