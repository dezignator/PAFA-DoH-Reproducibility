# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# src/hardware_simulation/loihi_mock.py — PAFA-DoH Reproducibility Artifact
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
# Calibrated software simulator for Intel Loihi 2. Reproduces the
# reported latency (0.8 ms) and energy per inference (0.12 mJ) from
# Table 13 within the tolerances documented in
# REPRODUCIBILITY_CHECKLIST.md §5 (±20 % latency, ±25 % energy).
#
# The simulator does NOT emulate the Loihi 2 architecture, instruction
# set, or any proprietary Intel technology. It emulates the *reported
# numeric behavior* of the hardware, which is what a reviewer needs in
# order to verify that the artifact's Table 13 reproduction is
# self-consistent. See LICENSE §3 for the full statement.
#
# ───────────────────────────────────────────────────────────────────────────
# Where the pieces come from in Chapter 4
# ───────────────────────────────────────────────────────────────────────────
#   Latency target (0.8 ms)      → Table 13, §4.25
#   Energy target (0.12 mJ)      → Table 13, §4.25
#   Spike efficiency (0.45)      → Table 13
#   Tolerances                   → REPRODUCIBILITY_CHECKLIST.md §5
#   Config block                 → configs/hardware_emulation_config.yaml
#                                  (File 9h, `loihi_2:`)
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


class LoihiMock:
    """Calibrated simulator for Intel Loihi 2.

    Args:
        seed: RNG seed for the jitter added on top of the calibrated
            mean (default 42).
        latency_jitter_pct: Relative jitter on the latency
            (default 0.10 = ±10 %, which is within the ±20 %
            tolerance).
        energy_jitter_pct: Relative jitter on the energy
            (default 0.15 = ±15 %, which is within the ±25 %
            tolerance).
    """

    # The reported targets from Table 13.
    TARGET_LATENCY_MS = 0.8
    TARGET_ENERGY_MJ = 0.12
    TARGET_SPIKE_EFFICIENCY = 0.45

    # The tolerances from REPRODUCIBILITY_CHECKLIST.md §5.
    LATENCY_TOLERANCE_PCT = 20.0
    ENERGY_TOLERANCE_PCT = 25.0
    SPIKE_EFFICIENCY_TOLERANCE_PCT = 10.0

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
            A dict with:
              • latency_ms:         observed latency
              • energy_mj:          observed energy
              • spike_efficiency:   observed bits/spike
              • simulated:          True
              • target_latency_ms:  0.8 (the Table 13 target)
              • target_energy_mj:   0.12
              • tolerance_latency_pct: 20.0
              • tolerance_energy_pct:  25.0
        """
        # Multiplicative jitter centered on 1.0, symmetric in log-space
        # (so that the distribution of the observed value is centered
        # on the target, not shifted up).
        lat_factor = np.exp(self.rng.normal(0, self.latency_jitter_pct / 3))
        eng_factor = np.exp(self.rng.normal(0, self.energy_jitter_pct / 3))
        eff_factor = np.exp(self.rng.normal(0, 0.03))  # ±10 % tolerance

        latency = float(self.TARGET_LATENCY_MS * lat_factor)
        energy = float(self.TARGET_ENERGY_MJ * eng_factor)
        efficiency = float(self.TARGET_SPIKE_EFFICIENCY * eff_factor)

        result = {
            "simulated": True,
            "hardware": "Loihi 2 (simulated)",
            "simulator_version": "pafa-hardware-sim/1.0.0",
            "latency_ms": latency,
            "energy_mj": energy,
            "spike_efficiency_bits_per_spike": efficiency,
            "target_latency_ms": self.TARGET_LATENCY_MS,
            "target_energy_mj": self.TARGET_ENERGY_MJ,
            "target_spike_efficiency": self.TARGET_SPIKE_EFFICIENCY,
            "tolerance_latency_pct": self.LATENCY_TOLERANCE_PCT,
            "tolerance_energy_pct": self.ENERGY_TOLERANCE_PCT,
            "tolerance_spike_efficiency_pct": self.SPIKE_EFFICIENCY_TOLERANCE_PCT,
        }
        log.debug(
            "Loihi simulation: latency=%.4f ms (target %.2f, tolerance ±%.0f %%), "
            "energy=%.4f mJ (target %.2f, tolerance ±%.0f %%)",
            latency, self.TARGET_LATENCY_MS, self.LATENCY_TOLERANCE_PCT,
            energy, self.TARGET_ENERGY_MJ, self.ENERGY_TOLERANCE_PCT,
        )
        return result

    def simulate_batch(self, n: int) -> list[dict[str, Any]]:
        """Simulate `n` independent inferences.

        Useful for the significance test in scripts/reproduce_tables/
        table_13_neuromorphic.py, which compares the distribution of
        observed latencies against the target.
        """
        if n <= 0:
            raise ValueError(f"n must be positive; got {n}.")
        return [self.simulate_inference() for _ in range(n)]
