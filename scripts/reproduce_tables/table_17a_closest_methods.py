#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# scripts/reproduce_tables/table_17a_closest_methods.py
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
# Reproduces Table 17a (Closest Published Methods).
# This is a configuration-only table: it maps each PAFA-DoH component
# to its closest published method and the specific configuration used
# in this artifact.
#
# Output: outputs/tables/table_17a_closest_methods.json
# ───────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pafa.utils.logger import get_logger, resolve_artifact_root

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce Table 17a.")
    args = parser.parse_args(argv)

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)

    root = resolve_artifact_root()
    logger = get_logger("table_17a", log_dir=root / "outputs" / "logs")
    logger.info("Reproducing Table 17a...")

    table_17a = {
        "federated_aggregation": {
            "closest_method": "Federated averaging with DP-SGD (Abadi et al., CCS 2016)",
            "configuration": "DP-SGD with epsilon=1.2, clipping C=3.5 and Gaussian noise sigma calibrated through the moments accountant"
        },
        "byzantine_robust_aggregation": {
            "closest_method": "Geometric median with iteratively reweighted least squares",
            "configuration": "Weighted geometric median with a cross-check against the topological detector outputs"
        },
        "homomorphic_inference": {
            "closest_method": "Ring-LWE encryption",
            "configuration": "Parameter selection n=256, q=4096 targeting 128-bit post-quantum security; application to anomaly-score evaluation"
        },
        "spiking_neural_network_detector": {
            "closest_method": "Leaky integrate-and-fire neuron with pair-based STDP",
            "configuration": "Logarithmic burst encoding (Eq. 43), three-layer LIF network, 20 ms STDP window, implementation on Intel Loihi"
        },
        "topological_anomaly_detection": {
            "closest_method": "Persistent homology with Vietoris-Rips filtration and p-Wasserstein distance",
            "configuration": "Application to encrypted-DNS flow point clouds; Betti-1, persistence-entropy and Wasserstein comparison with Bonferroni correction"
        },
        "zero_knowledge_attestation": {
            "closest_method": "Groth16 zk-SNARK",
            "configuration": "Attestation circuit of ~10^5 constraints for resolver policy; both the SNARK and the STARK path evaluated"
        },
        "side_channel_noise_injection": {
            "closest_method": "Adaptive noise injection (standard countermeasure)",
            "configuration": "SNN-guided amplitude and spectral shaping of the injected noise; adaptive threshold from Eq. 48"
        },
        "adversarial_training_data": {
            "closest_method": "GAN-generated synthetic queries",
            "configuration": "Total-variation regulariser on the generator, applied to encrypted-DNS query generation"
        }
    }

    out_dir = root / "outputs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "table_17a_closest_methods.json"
    out_path.write_text(json.dumps(table_17a, indent=2))
    logger.info("Wrote %s", out_path)

    print(f"\nTable 17a reproduced: {out_path}")
    print(f"  Mapped {len(table_17a)} components to their closest published methods.")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
