#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# scripts/reproduce_tables/table_13_neuromorphic.py
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
# Reproduces Table 13 (Neuromorphic Computing Efficiency) of Chapter 4.
# Uses the calibrated hardware simulators (`LoihiMock` and `A100Emulation`)
# to generate the latency, energy, and spike-efficiency metrics, and
# calculates the derived speedup and energy-efficiency ratios.
#
# Output: outputs/tables/table_13_neuromorphic.json
# ───────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pafa.hardware_simulation import LoihiMock, A100Emulation
from pafa.utils.logger import get_logger, resolve_artifact_root

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce Table 13.")
    parser.add_argument("--simulate", action="store_true", default=True,
                        help="Use calibrated simulators (default).")
    parser.add_argument("--runs", type=int, default=100,
                        help="Number of simulation runs to average (default 100).")
    args = parser.parse_args(argv)

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)

    root = resolve_artifact_root()
    logger = get_logger("table_13", log_dir=root / "outputs" / "logs")
    logger.info("Reproducing Table 13 with %d runs per simulator...", args.runs)

    # 1. Initialize simulators
    loihi = LoihiMock(seed=42)
    a100 = A100Emulation(seed=42)

    # 2. Run batch simulations
    loihi_results = loihi.simulate_batch(args.runs)
    a100_results = a100.simulate_batch(args.runs)

    # 3. Aggregate results (mean)
    loihi_lat = float(np.mean([r["latency_ms"] for r in loihi_results]))
    loihi_eng = float(np.mean([r["energy_mj"] for r in loihi_results]))
    loihi_eff = float(np.mean([r["spike_efficiency_bits_per_spike"] for r in loihi_results]))

    a100_lat = float(np.mean([r["latency_ms"] for r in a100_results]))
    a100_eng = float(np.mean([r["energy_mj"] for r in a100_results]))

    # 4. Calculate derived metrics
    speedup = a100_lat / loihi_lat if loihi_lat > 0 else 0.0
    efficiency_gain = a100_eng / loihi_eng if loihi_eng > 0 else 0.0

    # 5. Assemble the output
    table_13 = {
        "metric": {
            "loihi_inference_latency_ms": loihi_lat,
            "gpu_inference_latency_ms": a100_lat,
            "loihi_energy_per_inference_mj": loihi_eng,
            "gpu_energy_per_inference_mj": a100_eng,
            "speedup_latency": speedup,
            "efficiency_gain_energy": efficiency_gain,
            "spike_encoding_efficiency_bits_per_spike": loihi_eff,
        },
        "simulated": True,
        "simulator_version": "pafa-hardware-sim/1.0.0",
        "targets": {
            "loihi_inference_latency_ms": 0.8,
            "gpu_inference_latency_ms": 2.1,
            "loihi_energy_per_inference_mj": 0.12,
            "gpu_energy_per_inference_mj": 1.8,
            "speedup_latency": 2.6,
            "efficiency_gain_energy": 15.0,
            "spike_encoding_efficiency_bits_per_spike": 0.45,
        },
        "tolerances": {
            "latency_pct": 20.0,
            "energy_pct": 25.0,
            "spike_efficiency_pct": 10.0,
        }
    }

    # 6. Write output
    out_dir = root / "outputs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "table_13_neuromorphic.json"
    out_path.write_text(json.dumps(table_13, indent=2))
    logger.info("Wrote %s", out_path)

    print(f"\nTable 13 reproduced: {out_path}")
    print(f"  Loihi latency: {loihi_lat:.3f} ms (target: 0.8)")
    print(f"  Speedup:       {speedup:.2f}x (target: 2.6x)")
    print(f"  Energy gain:   {efficiency_gain:.2f}x (target: 15.0x)")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
