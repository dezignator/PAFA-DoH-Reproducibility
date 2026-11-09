# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# src/models/self_healing.py — PAFA-DoH Reproducibility Artifact
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
# Self-healing trust fabric. Implements the entropy-based trust
# score (Eq. 53), resolver blacklisting, and reintegration. This is
# the component that autonomously reacts when a resolver's behavior
# becomes anomalous.
#
# ───────────────────────────────────────────────────────────────────────────
# Where the pieces come from in Chapter 4
# ───────────────────────────────────────────────────────────────────────────
#   Eq. 53 — Entropy trust score    → §4.19.3
#   §4.19.1–§4.19.2 — SNN forensics + blacklisting → §4.19
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import logging
import numpy as np
from typing import Any

from ..utils.math_utils import entropy_trust_score

log = logging.getLogger(__name__)


class SelfHealingTrustFabric:
    """Entropy-based trust scoring and blacklist management."""
    
    def __init__(self, trust_threshold: float = 0.5):
        self.trust_threshold = trust_threshold
        self.blacklist: set[str] = set()

    def detect_behavioral_anomaly(
        self,
        resolver_id: str,
        observed_entropy: float,
        expected_entropy: float,
        max_entropy: float,
    ) -> bool:
        """Compute the trust score (Eq. 53) and return True if below threshold."""
        score = entropy_trust_score(observed_entropy, expected_entropy, max_entropy)
        if score < self.trust_threshold:
            log.warning("Resolver %s trust score %.3f below threshold %.3f",
                        resolver_id, score, self.trust_threshold)
            return True
        return False

    def blacklist_resolver(self, resolver_id: str):
        """Add the resolver to the blacklist."""
        self.blacklist.add(resolver_id)
        log.warning("Blacklisted resolver: %s", resolver_id)

    def reintegrate_resolver(self, resolver_id: str):
        """Remove the resolver from the blacklist."""
        if resolver_id in self.blacklist:
            self.blacklist.discard(resolver_id)
            log.info("Reintegrated resolver: %s", resolver_id)
        else:
            log.info("Resolver %s not in blacklist; nothing to reintegrate.", resolver_id)
