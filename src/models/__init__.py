# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# src/models/__init__.py — PAFA-DoH Reproducibility Artifact
# ═══════════════════════════════════════════════════════════════════════════
#
# Thesis chapter: Chapter 4 (PAFA-DoH)
# Citation DOI:   10.1016/j.neunet.2025.108343
# License:        MIT
# Artifact version: 1.0.0
#
# ───────────────────────────────────────────────────────────────────────────
# Lazy exports for the seven component models of PAFA-DoH. Importing this
# module does NOT pull in torch, gudhi, or py_ecc; each class is loaded
# only when first accessed. This allows `import pafa.models` in a minimal
# environment (e.g., a CI job that only checks the schema).
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

__all__ = [
    "PAFAFederatedLearning",
    "SNNDetector",
    "TDADetector",
    "Groth16Attestation",
    "FRI_STARKAttestation",
    "RingLWE",
    "DPInference",
    "SelfHealingTrustFabric",
    "SNNGuidedNoiseInjector",
]


def __getattr__(name: str):
    if name == "PAFAFederatedLearning":
        from .federated_learning import PAFAFederatedLearning
        return PAFAFederatedLearning
    if name == "SNNDetector":
        from .snn_detector import SNNDetector
        return SNNDetector
    if name == "TDADetector":
        from .tda_detector import TDADetector
        return TDADetector
    if name == "Groth16Attestation":
        from .zk_attestation import Groth16Attestation
        return Groth16Attestation
    if name == "FRI_STARKAttestation":
        from .zk_attestation import FRI_STARKAttestation
        return FRI_STARKAttestation
    if name == "RingLWE":
        from .homomorphic_inference import RingLWE
        return RingLWE
    if name == "DPInference":
        from .homomorphic_inference import DPInference
        return DPInference
    if name == "SelfHealingTrustFabric":
        from .self_healing import SelfHealingTrustFabric
        return SelfHealingTrustFabric
    if name == "SNNGuidedNoiseInjector":
        from .ai_obfuscated_crypto import SNNGuidedNoiseInjector
        return SNNGuidedNoiseInjector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
