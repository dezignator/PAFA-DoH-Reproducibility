# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# src/models/tda_detector.py — PAFA-DoH Reproducibility Artifact
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
# Topological data analysis detector. Builds persistence diagrams from
# flow point clouds, compares test diagrams to a benign reference using
# the p-Wasserstein distance, and extracts Betti-1 and persistence-
# entropy features for Table 17.
#
# ───────────────────────────────────────────────────────────────────────────
# Where the pieces come from in Chapter 4
# ───────────────────────────────────────────────────────────────────────────
#   Eq. 40 — Persistence diagram     → §4.7.1
#   Eq. 41 — Wasserstein distance    → §4.7.2
#   Eq. 42 — Hypothesis test         → §4.8.1
#   Table 17 — Reported values       → §4.28.2
#   Config   — tda_params.yaml       → File 9d
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import logging
import numpy as np
from typing import Any

from ..utils.math_utils import (
    betti_number,
    hypothesis_test_p_value,
    persistence_diagram,
    persistence_entropy,
    wasserstein_distance_persistence,
)

log = logging.getLogger(__name__)


class TDADetector:
    """Topological detector for encrypted DoH traffic."""
    
    def __init__(
        self,
        max_edge: float = 100.0,
        max_dim: int = 1,
        p_wasserstein: int = 2,
        backend: str = "ripser",
    ):
        self.max_edge = max_edge
        self.max_dim = max_dim
        self.p_wasserstein = p_wasserstein
        self.backend = backend
        self.reference_dgm: np.ndarray | None = None

    def fit_reference(self, benign_point_cloud: np.ndarray):
        """Build the benign reference diagram from a sample of benign flows."""
        dgms = persistence_diagram(
            benign_point_cloud,
            max_edge=self.max_edge,
            max_dim=self.max_dim,
            backend=self.backend,
        )
        self.reference_dgm = dgms[1]  # Use H_1 (loops) as the reference.
        log.info("Fitted reference diagram with %d features.", len(self.reference_dgm))

    def compute_diagram(self, point_cloud: np.ndarray) -> list[np.ndarray]:
        """Compute persistence diagrams for a point cloud (Eq. 40)."""
        return persistence_diagram(
            point_cloud,
            max_edge=self.max_edge,
            max_dim=self.max_dim,
            backend=self.backend,
        )

    def anomaly_score(self, test_point_cloud: np.ndarray) -> float:
        """Compute the Wasserstein-based anomaly score (Eq. 41)."""
        if self.reference_dgm is None:
            raise RuntimeError("Call fit_reference() before anomaly_score().")
            
        dgms = self.compute_diagram(test_point_cloud)
        test_dgm = dgms[1]  # H_1
        
        return wasserstein_distance_persistence(
            test_dgm,
            self.reference_dgm,
            p=self.p_wasserstein,
        )
