# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# src/models/homomorphic_inference.py — PAFA-DoH Reproducibility Artifact
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
# Ring-LWE homomorphic encryption layer and differential-privacy
# inference for PAFA-DoH. Implements Eq. 58 (encryption), and the
# moments accountant (Eq. 60) used for the privacy-utility trade-off.
#
# ───────────────────────────────────────────────────────────────────────────
# Where the pieces come from in Chapter 4
# ───────────────────────────────────────────────────────────────────────────
#   Eq. 58 — Ring-LWE encrypt       → §4.21
#   Eq. 59–60 — Moments accountant  → §4.22
#   Table 16 — Reported values      → §4.28.1
#   Table 17a — n=256, q=4096       → §4.31
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import logging
import numpy as np

from ..utils.math_utils import (
    moments_accountant,
    ring_lwe_decrypt,
    ring_lwe_encrypt,
)

log = logging.getLogger(__name__)


class RingLWE:
    """Ring-LWE encryption (Eq. 58).
    
    Parameters n=256, q=4096 are pinned in Table 17a. The error sigma
    is 3.2, chosen so that the decryption-failure probability is below
    2^-40 at this parameter set.
    """
    
    def __init__(self, n: int = 256, q: int = 4096, sigma: float = 3.2, seed: int = 42):
        self.n = n
        self.q = q
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)
        # Generate a fixed secret and a per-call public vector. In the
        # full pipeline (File 17a), the secret is generated once per
        # session; here it is fixed at construction so the toy demo
        # can round-trip.
        self.s = self.rng.integers(-3, 4, n).astype(np.int64)

    def _sample_error(self) -> np.ndarray:
        return np.round(self.rng.normal(0, self.sigma, self.n)).astype(np.int64)

    def encrypt(self, message_bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Encrypt a binary message (Eq. 58)."""
        a = self.rng.integers(0, self.q, self.n).astype(np.int64)
        e = self._sample_error()
        return ring_lwe_encrypt(message_bits, a, self.s, e, self.q)

    def decrypt(self, c0: np.ndarray, c1: np.ndarray) -> np.ndarray:
        """Decrypt a ciphertext pair."""
        return ring_lwe_decrypt(c0, c1, self.s, self.q)


class DPInference:
    """Differential-privacy wrapper for inference.
    
    The moments accountant (Eq. 60) computes the achieved epsilon for
    a given (sigma, T, delta) triple. The artifact uses sigma=1.1,
    T=100, delta=1e-5, which yields epsilon ≈ 1.198, matching the
    pinned epsilon=1.2 in Table 16.
    """
    
    def __init__(
        self,
        epsilon: float = 1.2,
        delta: float = 1e-5,
        batch_size: int = 64,
        num_clients: int = 100,
        noise_multiplier: float = 1.1,
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.batch_size = batch_size
        self.num_clients = num_clients
        self.noise_multiplier = noise_multiplier

    def moments_accountant(self, num_rounds: int) -> float:
        """Compute epsilon for the given number of rounds (Eq. 60)."""
        return moments_accountant(
            noise_multiplier=self.noise_multiplier,
            num_rounds=num_rounds,
            target_delta=self.delta,
            batch_size=self.batch_size,
            num_clients=self.num_clients,
        )
