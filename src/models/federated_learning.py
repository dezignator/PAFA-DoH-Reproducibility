# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# src/models/federated_learning.py — PAFA-DoH Reproducibility Artifact
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
# The federated learning coordinator. Implements the DP-SGD update
# (Eq. 36), the geometric-median Byzantine-robust aggregation (Eq. 38),
# the MAD-based outlier screen, and the convergence tracking used to
# produce Table 16.
#
# This is a reference implementation. It runs on NumPy arrays and does
# not depend on torch; the torch-based trainer used for the full 32 GB
# corpus lives in `scripts/train_pafa.py` (File 17a), which imports
# this class for the aggregation and DP logic.
#
# ───────────────────────────────────────────────────────────────────────────
# Where the pieces come from in Chapter 4
# ───────────────────────────────────────────────────────────────────────────
#   Eq. 36 — DP-SGD update          → §4.6.2
#   Eq. 38 — Geometric median       → §4.6.3
#   Eq. 39 — Convergence bound      → §4.6.4
#   Table 16 — Reported values      → §4.28.1
#   Config values (ε, δ, C, σ, K)   → configs/fl_dp_params.yaml (File 9b)
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..utils.math_utils import (
    dp_sgd_step,
    geometric_median,
)

log = logging.getLogger(__name__)


@dataclass
class ConvergenceTracker:
    """Tracks the convergence of the federated training loop.

    Used by PAFAFederatedLearning to determine when the global model has
    converged (gradient norm below the threshold for `patience`
    consecutive epochs) and to record the per-round metrics for Table 16.
    """
    target_accuracy: float = 0.992
    gradient_norm_threshold: float = 1e-3
    patience: int = 2
    
    history: list[dict[str, float]] = field(default_factory=list)
    _consecutive_below: int = 0

    def step(self, epoch: int, accuracy: float, gradient_norm: float) -> bool:
        self.history.append({
            "epoch": epoch,
            "accuracy": accuracy,
            "gradient_norm": gradient_norm,
        })
        if gradient_norm < self.gradient_norm_threshold:
            self._consecutive_below += 1
        else:
            self._consecutive_below = 0
            
        if self._consecutive_below >= self.patience and accuracy >= self.target_accuracy:
            log.info("Convergence reached at epoch %d (acc=%.4f, grad_norm=%.2e)", epoch, accuracy, gradient_norm)
            return True
        return False


class PAFAFederatedLearning:
    """Federated learning coordinator with DP and Byzantine robustness.

    Args:
        num_clients: Number of participating clients (Table 16: 100).
        epsilon: DP budget (Table 16: 1.2).
        delta: DP delta (default 1e-5).
        clip_norm: l2 clip for DP-SGD (Table 17a: 3.5).
        noise_multiplier: Calibrated sigma (default 1.1, yields ε=1.198).
        batch_size: Local batch size (default 64).
        mad_multiplier: MAD threshold for Byzantine rejection (default 2.5).
    """
    
    def __init__(
        self,
        num_clients: int = 100,
        epsilon: float = 1.2,
        delta: float = 1e-5,
        clip_norm: float = 3.5,
        noise_multiplier: float = 1.1,
        batch_size: int = 64,
        mad_multiplier: float = 2.5,
        seed: int = 42,
    ):
        self.num_clients = num_clients
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        self.batch_size = batch_size
        self.mad_multiplier = mad_multiplier
        self.rng = np.random.default_rng(seed)
        self.tracker = ConvergenceTracker()

    def aggregate(self, updates: Sequence[np.ndarray]) -> np.ndarray:
        """Aggregate client updates using geometric median (Eq. 38).
        
        First applies the MAD-based Byzantine screen, then calls
        geometric_median on the surviving updates.
        """
        if len(updates) == 0:
            raise ValueError("No updates to aggregate.")
            
        updates_arr = np.stack(updates, axis=0)
        
        # 1. Geometric median for the base aggregate.
        median = geometric_median(updates_arr)
        
        # 2. MAD screen (Byzantine robustness).
        distances = np.linalg.norm(updates_arr - median[None, :], axis=1)
        mad = np.median(np.abs(distances - np.median(distances)))
        if mad > 0:
            # Reject outliers whose distance from the median exceeds
            # mad_multiplier × MAD.
            threshold = self.mad_multiplier * mad
            mask = distances <= threshold
            if mask.sum() == 0:
                log.warning("MAD screen rejected all updates; falling back to geometric median.")
                return median
            if mask.sum() < len(updates):
                log.info("MAD screen rejected %d/%d updates.", len(updates) - mask.sum(), len(updates))
            return geometric_median(updates_arr[mask])
            
        return median

    def dp_step(self, weights: np.ndarray, gradients: Sequence[np.ndarray], learning_rate: float) -> np.ndarray:
        """One DP-SGD step (Eq. 36).
        
        Delegates to `math_utils.dp_sgd_step` for the actual math.
        """
        return dp_sgd_step(
            weights=weights,
            gradients=gradients,
            learning_rate=learning_rate,
            clip_norm=self.clip_norm,
            noise_multiplier=self.noise_multiplier,
            batch_size=self.batch_size,
            num_clients=self.num_clients,
            rng=self.rng,
        )
