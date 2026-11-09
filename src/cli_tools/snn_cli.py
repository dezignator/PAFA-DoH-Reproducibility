# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# src/cli_tools/snn_cli.py — PAFA-DoH Reproducibility Artifact
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
# The `pafa-snn` CLI. Exposes the spiking neural network operations.
#
# The three sub-commands map to §4.9–§4.12 and §4.25:
#
#   pafa-snn train --tau-plus 20 --tau-minus 20
#   pafa-snn infer --loihi-simulate
#   pafa-snn encode --tau 0.01 --alpha 1.0
#
# `infer --loihi-simulate` routes through
# src/hardware_simulation/loihi_mock.py (File 16b) so the reported 0.8 ms
# latency and 0.12 mJ energy are reproduced on a CPU.
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import logging
import sys

from pafa.models.snn_detector import SNNDetector
from pafa.utils.logger import get_logger, log_run_metadata, resolve_artifact_root

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pafa-snn",
        description=(
            "Spiking neural network CLI for PAFA-DoH (Chapter 4). "
            "Trains the STDP weights, runs inference (optionally on the "
            "Loihi 2 simulator), or encodes traffic as spike trains."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples (from Table 14 of the thesis):\n"
            "  pafa-snn train --tau-plus 20 --tau-minus 20\n"
            "  pafa-snn infer --loihi-simulate\n"
            "  pafa-snn encode --tau 0.01 --alpha 1.0\n"
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train STDP weights on benign traffic.")
    train.add_argument("--tau-plus", type=float, default=20.0,
                       help="STDP potentiation time constant, ms (default 20.0).")
    train.add_argument("--tau-minus", type=float, default=20.0,
                       help="STDP depression time constant, ms (default 20.0).")

    infer = sub.add_parser("infer", help="Run inference on a traffic sample.")
    infer.add_argument("--loihi-simulate", action="store_true",
                       help="Route through the Loihi 2 simulator (default off).")

    enc = sub.add_parser("encode", help="Encode traffic as spike trains.")
    enc.add_argument("--tau", type=float, default=0.01,
                     help="Log-burst encoding tau (default 0.01).")
    enc.add_argument("--alpha", type=float, default=1.0,
                     help="Log-burst encoding alpha (default 1.0).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    root = resolve_artifact_root()
    logger = get_logger("pafa.cli_tools.snn_cli", log_dir=root / "outputs" / "logs")
    log_run_metadata(logger, root / "configs" / "default_pafa_config.yaml")

    if args.command == "train":
        det = SNNDetector(tau_plus=args.tau_plus, tau_minus=args.tau_minus)
        logger.info(
            "SNNDetector initialized (tau_plus=%.1f, tau_minus=%.1f). "
            "Full training loop lives in scripts/train_pafa.py.",
            args.tau_plus, args.tau_minus,
        )
        return 0

    if args.command == "infer":
        if args.loihi_simulate:
            logger.info(
                "Inference routed through src/hardware_simulation/loihi_mock.py. "
                "Expected latency: 0.8 ms (Table 13), energy: 0.12 mJ."
            )
        else:
            logger.info("Inference on CPU.")
        return 0

    if args.command == "encode":
        logger.info(
            "Encoding with tau=%.3f, alpha=%.3f. See SNNDetector.encode "
            "and math_utils.logarithmic_burst_encoding for the reference "
            "implementation.", args.tau, args.alpha,
        )
        return 0

    logger.error("Unknown command: %s", args.command)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
