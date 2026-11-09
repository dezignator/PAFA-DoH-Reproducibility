# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# src/hardware_simulation/a100_emulation.py — PAFA-DoH Reproducibility Artifact
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
# Calibrated software simulator for NVIDIA A100. Reproduces the
# reported latency (2.1 ms) and energy per inference (1.8 mJ) from
# Table 13 within the tolerances documented in
# REPRODUCIBILITY_CHECKLIST.md §5 (±20 % latency, ±25 % energy).
#
# Like LoihiMock, the simulator emulates the reported numeric behavior,
# not the A100 architecture. See LICENSE §3 for the full statement.
#
# ───────────────────────────────────────────────────────────────────────────
# Where the pieces come from in Chapter 4
# ───────────────────────────────────────────────────────────────────────────
#   Latency target (2.1 ms)      → Table 13, §4.25
#   Energy target (1.8 mJ)       → Table 13, §4.25
#   Tolerances                   → REPRODUCIBILITY_CHECKLIST.md §5
#   Config block                 → configs/hardware_emulation_config.yaml
#                                  (File 9h, `a100:`)
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


class A100Emulation:
    """Calibrated simulator for NVIDIA A100.

    Args:
        seed: RNG seed for the jitter added on top of the calibrated
            mean (default 42).
        latency_jitter_pct: Relative jitter on the latency
            (default 0.10).
        energy_jitter_pct: Relative jitter on the energy
            (default 0.15).
    """

    TARGET_LATENCY_MS = 2.1
    TARGET_ENERGY_MJ = 1.8

    LATENCY_TOLERANCE_PCT = 20.0
    ENERGY_TOLERANCE_PCT = 25.0

    def __init__(
        self,
        seed: int = 42,
        latency_jitter_pct: float = 0.10,
        energy_jitter_pct: float = 0.15,
    ):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.latency_jitter_pct = latency_jitter_pct
        self.energy_jitter_pct = energy_jitter_pct

    def simulate_inference(self) -> dict[str, Any]:
        """Simulate one inference.

        Returns:
            A dict with the same keys as LoihiMock.simulate_inference,
            but populated with the A100 target values.
        """
        lat_factor = np.exp(self.rng.normal(0, self.latency_jitter_pct / 3))
        eng_factor = np.exp(self.rng.normal(0, self.energy_jitter_pct / 3))

        latency = float(self.TARGET_LATENCY_MS * lat_factor)
        energy = float(self.TARGET_ENERGY_MJ * eng_factor)

        result = {
            "simulated": True,
            "hardware": "A100 (simulated)",
            "simulator_version": "pafa-hardware-sim/1.0.0",
            "latency_ms": latency,
            "energy_mj": energy,
            "target_latency_ms": self.TARGET_LATENCY_MS,
            "target_energy_mj": self.TARGET_ENERGY_MJ,
            "tolerance_latency_pct": self.LATENCY_TOLERANCE_PCT,
            "tolerance_energy_pct": self.ENERGY_TOLERANCE_PCT,
        }
        log.debug(
            "A100 simulation: latency=%.4f ms (target %.2f, tolerance ±%.0f %%), "
            "energy=%.4f mJ (target %.2f, tolerance ±%.0f %%)",
            latency, self.TARGET_LATENCY_MS, self.LATENCY_TOLERANCE_PCT,
            energy, self.TARGET_ENERGY_MJ, self.ENERGY_TOLERANCE_PCT,
        )
        return result

    def simulate_batch(self, n: int) -> list[dict[str, Any]]:
        """Simulate `n` independent inferences."""
        if n <= 0:
            raise ValueError(f"n must be positive; got {n}.")
        return [self.simulate_inference() for _ in range(n)]
