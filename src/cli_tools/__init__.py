# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# src/cli_tools/__init__.py — PAFA-DoH Reproducibility Artifact
# ═══════════════════════════════════════════════════════════════════════════
#
# Thesis chapter: Chapter 4 (PAFA-DoH)
# Citation DOI:   10.1016/j.neunet.2025.108343
# License:        MIT
# Artifact version: 1.0.0
#
# ───────────────────────────────────────────────────────────────────────────
# The entry points for the four console scripts registered in setup.py
# (File 7):
#
#   pafa-fl  → federated_cli.py:main
#   pafa-snn → snn_cli.py:main
#   pafa-tda → tda_cli.py:main
#   pafa-zk  → zk_cli.py:main
#
# Every CLI in this subpackage uses the standard `argparse` module and
# exposes a `main(argv: list[str] | None = None) -> int` function.
# Returning an integer (rather than calling sys.exit directly) makes
# the CLIs testable from pytest without having to intercept SystemExit.
#
# Importing this module does NOT pull in torch or py_ecc; each CLI is
# loaded only when its `main()` is invoked by the console-script
# wrapper.
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

__all__ = [
    "federated_cli",
    "snn_cli",
    "tda_cli",
    "zk_cli",
]
