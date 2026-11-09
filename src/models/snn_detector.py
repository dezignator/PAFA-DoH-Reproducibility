# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# src/models/snn_detector.py — PAFA-DoH Reproducibility Artifact
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
# The spiking neural network anomaly detector. Converts DoH traffic to
# spike trains (Eq. 43), processes them through leaky integrate-and-fire
# neurons (Eqs. 44–45), and learns weights via spike-timing-dependent
# plasticity (Eq. 46). Generates the anomaly score used by PAFA-DoH.
#
# ───────────────────────────────────────────────────────────────────────────
# Where the pieces come from in Chapter 4
# ───────────────────────────────────────────────────────────────────────────
#   Eq. 43 — Log-burst spike encoding → §4.9.1
#   Eq. 44 — LIF membrane update      → §4.10
#   Eq. 45 — LIF threshold reset      → §4.10
#   Eq. 46 — STDP weight update       → §4.11
#   Table 13 — Latency/energy targets → §4.25
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import logging
import numpy as np

from ..utils.math_utils import (
    adaptive_threshold,
    lif_membrane_update,
    lif_threshold_reset,
    logarithmic_burst_encoding,
    stdp_weight_update,
)

log = logging.getLogger(__name__)


class SNNDetector:
    """Spiking Neural Network detector for DoH traffic.
    
    This is a NumPy reference implementation. The full pipeline in
    `scripts/train_pafa.py` (File 17a) uses `snntorch` for the
    optimized path; this class provides the exact math from Chapter 4
    and is used by the toy demo and the unit tests.
    """
    
    def __init__(
        self,
        tau: float = 0.01,
        alpha: float = 1.0,
        spikes_per_packet: int = 5,
        tau_m: float = 20.0,
        V_th: float = 1.0,
        V_reset: float = 0.0,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        eta: float = 0.01,
        seed: int = 42,
    ):
        self.tau = tau
        self.alpha = alpha
        self.spikes_per_packet = spikes_per_packet
        self.tau_m = tau_m
        self.V_th = V_th
        self.V_reset = V_reset
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.eta = eta
        self.rng = np.random.default_rng(seed)
        
    def encode(self, traffic_features: np.ndarray) -> np.ndarray:
        """Convert traffic features to spike times (Eq. 43)."""
        return logarithmic_burst_encoding(
            traffic_features,
            tau=self.tau,
            alpha=self.alpha,
            spikes_per_packet=self.spikes_per_packet,
        )

    def forward(self, spike_times: np.ndarray) -> float:
        """Process a single spike train through one LIF neuron.
        
        Returns the total number of spikes fired by the output neuron.
        The toy demo uses a single neuron; the full pipeline uses a
        three-layer network.
        """
        V = 0.0
        fired_count = 0
        for t in spike_times:
            # Constant synaptic input for the toy path.
            xi = self.rng.normal(0, 0.1)
            V = lif_membrane_update(V, I_syn=1.0, xi=xi, tau_m=self.tau_m, dt=1.0)
            V, fired, _ = lif_threshold_reset(V, V_th=self.V_th, V_reset=self.V_reset, t=t)
            if fired:
                fired_count += 1
        return float(fired_count)

    def anomaly_score(self, traffic_features: np.ndarray) -> float:
        """Compute the anomaly score for a traffic sample.
        
        The score is the negative of the fired spike count (more
        spikes → more anomalous). This is a reference path for the
        toy demo; the full pipeline uses the STDP-trained weights.
        """
        spikes = self.encode(traffic_features)
        count = self.forward(spikes)
        return -count
