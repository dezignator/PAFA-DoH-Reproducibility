#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# scripts/train_baselines.py — PAFA-DoH Reproducibility Artifact
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
# Train the two experimental baselines from §4.30:
#   • Centralized CNN-LSTM (§4.30.1)
#   • Rule-based detector (§4.30.3)
#
# Runs them on the same data and same tuning budget as PAFA-DoH, and
# writes their results to `outputs/tables/baselines_result.json` for
# the comparison table in §4.30.
#
# ───────────────────────────────────────────────────────────────────────────
# Where the pieces come from in Chapter 4
# ───────────────────────────────────────────────────────────────────────────
#   CNN-LSTM baseline        → §4.30.1
#   Rule-based baseline      → §4.30.3
#   Config block             → configs/baselines_config.yaml (File 9g)
# ───────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pafa.utils.seed import seed_everything
from pafa.utils.logger import get_logger, log_run_metadata, resolve_artifact_root
from pafa.baselines.centralized_ml import CentralizedMLBaseline
from pafa.baselines.rule_based import RuleBasedBaseline

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train PAFA-DoH baselines.")
    parser.add_argument("--config", type=Path, required=True, help="Path to master config.")
    parser.add_argument("--mode", choices=["toy", "full"], default="toy",
                        help="Training mode (default: toy).")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    try:
        root = resolve_artifact_root()
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1

    logger = get_logger("train_baselines", log_dir=root / "outputs" / "logs")
    log_run_metadata(logger, args.config)

    # Read config
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = cfg.get("seed", 42)
    seed_everything(seed)

    # Load data (same as train_pafa.py)
    data_path = root / "data" / "processed" / ("toy_dataset/combined.csv" if args.mode == "toy" else "zenodo_doh.csv")
    if not data_path.exists():
        log.error(f"Data not found at {data_path}.")
        return 1
    
    df = pd.read_csv(data_path)
    
    # Train/test split (simple 80/20 for the baseline comparison)
    train = df.sample(frac=0.8, random_state=seed)
    test = df.drop(train.index)

    results = {}

    # 1. Centralized ML Baseline
    log.info("Training Centralized CNN-LSTM baseline...")
    ml_baseline = CentralizedMLBaseline(seed=seed)
    try:
        ml_baseline.fit(train, test)
        ml_result = ml_baseline.evaluate(test)
        results["centralized_ml"] = {
            "accuracy": ml_result.accuracy,
            "fpr": ml_result.fpr,
        }
        log.info("CNN-LSTM baseline: accuracy=%.4f, fpr=%.4f", ml_result.accuracy, ml_result.fpr)
    except Exception as exc:
        log.error("CNN-LSTM baseline failed: %s", exc)
        results["centralized_ml"] = {"error": str(exc)}

    # 2. Rule-Based Baseline
    log.info("Evaluating Rule-Based baseline...")
    rule_baseline = RuleBasedBaseline()
    rule_result = rule_baseline.evaluate(test)
    results["rule_based"] = {
        "accuracy": rule_result["accuracy"],
        "fpr": rule_result["fpr"],
    }
    log.info("Rule-based baseline: accuracy=%.4f, fpr=%.4f", rule_result["accuracy"], rule_result["fpr"])

    # Save outputs
    out_dir = root / "outputs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baselines_result.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info("Saved baseline results to %s", out_path)

    print("\nNEXT STEP:")
    print("  python scripts/evaluate_models.py")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
