# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# src/hardware_simulation/__init__.py — PAFA-DoH Reproducibility Artifact
# ═══════════════════════════════════════════════════════════════════════════
#
# Thesis chapter: Chapter 4 (PAFA-DoH)
# Citation DOI:   10.1016/j.neunet.2025.108343
# License:        MIT
# Artifact version: 1.0.0
#
# ───────────────────────────────────────────────────────────────────────────
# Lazy exports for the two hardware simulators.
#
# These simulators are pure NumPy and do not import torch, gudhi, or
# any other heavy dependency. They are imported only when the caller
# actually wants to run a simulation, so that `import pafa` and
# `import pafa.hardware_simulation` remain cheap.
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

__all__ = [
    "LoihiMock",
    "A100Emulation",
]


def __getattr__(name: str):
    if name == "LoihiMock":
        from .loihi_mock import LoihiMock
        return LoihiMock
    if name == "A100Emulation":
        from .a100_emulation import A100Emulation
        return A100Emulation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
