# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# src/attacks/__init__.py — PAFA-DoH Reproducibility Artifact
# ═══════════════════════════════════════════════════════════════════════════
#
# Thesis chapter: Chapter 4 (PAFA-DoH)
# Citation DOI:   10.1016/j.neunet.2025.108343
# License:        MIT
# Artifact version: 1.0.0
#
# ───────────────────────────────────────────────────────────────────────────
# Lazy exports for the four attack generators used in Chapter 4:
#
#   DNSTunnelingAttack     — §4.26.1  (dnscat2-style subdomain exfiltration)
#   ZeroDayPhishingAttack  — §4.26.2  (GAN-generated domains)
#   SideChannelAttack      — §4.26.3  (ChipWhisperer-style timing / power)
#   C2ExfiltrationAttack   — §4.26.4  (multi-resolver periodic beaconing)
#
# Every generator exposes a `generate(n: int) -> pd.DataFrame` method
# returning exactly the 47-column schema defined in `data/data_utils.py`
# (File 10a), so the attack DataFrames plug directly into the training
# and evaluation scripts.
#
# The reported per-attack detection rates in Table 14 (99.5 % DNS
# tunnelling, 98.7 % C2 heartbeat, 100 % side-channel) are the
# evaluation outputs of these generators combined with the PAFA-DoH
# detector — not properties of the generators themselves. The
# generators produce traffic; the detectors classify it; Table 14
# records the classification rates.
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

__all__ = [
    "DNSTunnelingAttack",
    "ZeroDayPhishingAttack",
    "SideChannelAttack",
    "C2ExfiltrationAttack",
]


def __getattr__(name: str):
    if name == "DNSTunnelingAttack":
        from .dns_tunneling import DNSTunnelingAttack
        return DNSTunnelingAttack
    if name == "ZeroDayPhishingAttack":
        from .zero_day_phishing import ZeroDayPhishingAttack
        return ZeroDayPhishingAttack
    if name == "SideChannelAttack":
        from .side_channel import SideChannelAttack
        return SideChannelAttack
    if name == "C2ExfiltrationAttack":
        from .c2_exfiltration import C2ExfiltrationAttack
        return C2ExfiltrationAttack
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
