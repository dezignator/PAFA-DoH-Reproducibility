# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# src/models/ai_obfuscated_crypto.py — PAFA-DoH Reproducibility Artifact
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
# AI-obfuscated cryptography: SNN-guided noise injection for side-
# channel hardening (Eq. 47), adaptive threshold for the noise
# controller (Eq. 48), and dynamic key obfuscation via STDP (Eq. 49).
# The goal is to raise the cost of side-channel observation without
# changing the underlying cryptographic hardness.
#
# ───────────────────────────────────────────────────────────────────────────
# Where the pieces come from in Chapter 4
# ───────────────────────────────────────────────────────────────────────────
#   Eq. 47 — SNR ratio                        → §4.13.1
#   Eq. 48 — Adaptive threshold               → §4.13.2
#   Eq. 49 — STDP-modulated key obfuscation   → §4.13.3
#   Table 14 — Side-channel 100 % (categorical) → §4.26.3
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import logging
import numpy as np
from typing import Any

from ..utils.math_utils import (
    adaptive_threshold,
    snr_ratio,
    stdp_dynamic_key_obfuscation,
)

log = logging.getLogger(__name__)


class SNNGuidedNoiseInjector:
    """SNN-guided noise injector for side-channel hardening.
    
    The injector's design goal (Eq. 47) is to keep the leaked-emission
    SNR below a threshold δ that the adversary cannot separate from
    measurement noise. The adaptive threshold (Eq. 48) concentrates
    the injection during high-threat intervals.
    """
    
    def __init__(
        self,
        V_th0: float = 0.5,
        beta: float = 1.0,
        sigma_range: tuple[float, float] = (0.5, 2.0),
        seed: int = 42,
    ):
        self.V_th0 = V_th0
        self.beta = beta
        self.sigma_min, self.sigma_max = sigma_range
        self.rng = np.random.default_rng(seed)
        self.theta_trace: dict[str, float] = {}

    def compute_snr(self, signal: np.ndarray, noise: np.ndarray) -> float:
        """Compute the L2 SNR (Eq. 47)."""
        return snr_ratio(signal, noise)

    def inject_noise(
        self,
        signal: np.ndarray,
        threat_level: float,
    ) -> np.ndarray:
        """Inject adaptive noise into a signal (Eqs. 47–48)."""
        # 1. Adaptive threshold from threat level.
        threshold = adaptive_threshold(self.V_th0, self.beta, threat_level)
        
        # 2. Noise magnitude scaled by threat level.
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * threat_level
        noise = self.rng.normal(0, sigma, size=signal.shape)
        
        # 3. Inject (additive noise, matching the Eq. 47 model).
        return signal + noise

    def obfuscate_key(
        self,
        synapse_id: str,
        delta_w: float,
        xi: float,
    ) -> float:
        """Update the STDP trace and return the per-step obfuscation (Eq. 49)."""
        theta_prev = self.theta_trace.get(synapse_id, 0.0)
        contribution, theta_new = stdp_dynamic_key_obfuscation(delta_w, xi, theta_prev)
        self.theta_trace[synapse_id] = theta_new
        return contribution
