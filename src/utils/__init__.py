# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# src/utils/__init__.py — PAFA-DoH Reproducibility Artifact
# ═══════════════════════════════════════════════════════════════════════════
#
# Thesis chapter: Chapter 4 (PAFA-DoH)
# Citation DOI:   10.1016/j.neunet.2025.108343
# License:        MIT
# Artifact version: 1.0.0
#
# ───────────────────────────────────────────────────────────────────────────
# Re-exports the four public utilities of the `utils` subpackage:
#
#   seed         — File 11c   — seed_everything(seed)
#   logger       — File 11d   — get_logger(name), ForbiddenObservablesGuard,
#                                resolve_artifact_root
#   metrics      — File 11e   — accuracy, fpr, precision, recall, f1,
#                                auc_roc, cohens_d, bootstrap_ci
#   math_utils   — File 11f   — every equation of Chapter 4 (Eqs. 32–67)
#
# `math_utils` is imported lazily (via __getattr__) because it has an
# optional `ripser` dependency for `persistence_diagram`. A reviewer
# who only wants `get_logger` should not need `ripser` installed. The
# other three modules have hard dependencies only on packages that are
# always available (numpy, scipy, pandas, PyYAML), so they are imported
# eagerly. `from pafa.utils import *` still works for all four.
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

# ─── Eager imports (soft dependencies only) ──────────────────────────────
from .logger import (
    ForbiddenObservablesGuard,
    get_logger,
    resolve_artifact_root,
)
from .metrics import (
    accuracy,
    auc_roc,
    bootstrap_ci,
    cohens_d,
    f1,
    fpr,
    precision,
    recall,
)
from .seed import seed_everything

# ─── math_utils is lazy — see docstring ──────────────────────────────────
__all__ = [
    "seed_everything",
    "get_logger",
    "ForbiddenObservablesGuard",
    "resolve_artifact_root",
    "accuracy",
    "fpr",
    "precision",
    "recall",
    "f1",
    "auc_roc",
    "cohens_d",
    "bootstrap_ci",
    "math_utils",
]


def __getattr__(name: str):
    """Lazy loader for `math_utils` only."""
    if name == "math_utils":
        import importlib
        module = importlib.import_module(f"{__name__}.math_utils")
        globals()["math_utils"] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
