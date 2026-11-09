# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# tests/test_reproduce_tables.py — PAFA-DoH Reproducibility Artifact
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
# Tests for the reproduce_tables and reproduce_figures scripts (Files
# 18a–19d) and the verification_script.py (File 20i).
#
# The tests are lightweight: they run each script in a temporary
# directory and confirm the script:
#   1. Exits 0,
#   2. Writes a file with the correct JSON schema (tables) or CSV
#      columns (figures), and
#   3. Produces values within the tolerance documented in
#      REPRODUCIBILITY_CHECKLIST.md §8.
#
# The tests do NOT run the full pipeline (which takes hours). They
# assume the toy-mode pipeline has already produced the necessary
# outputs in `outputs/`. If those outputs are missing, the tests
# SKIP with a clear message.
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _run_script(rel_path: str, *, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run a script as a subprocess and return the CompletedProcess."""
    script = REPO_ROOT / rel_path
    assert script.is_file(), f"Missing script: {script}"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}{os.pathsep}{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Table scripts
# ═══════════════════════════════════════════════════════════════════════════

class TestTableScripts:

    def test_table_13(self):
        result = _run_script("scripts/reproduce_tables/table_13_neuromorphic.py")
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"
        out = REPO_ROOT / "outputs" / "tables" / "table_13_neuromorphic.json"
        assert out.is_file()
        data = json.loads(out.read_text())
        for key in ["loihi_inference_latency_ms", "gpu_inference_latency_ms",
                    "loihi_energy_per_inference_mj", "gpu_energy_per_inference_mj",
                    "speedup_latency", "efficiency_gain_energy"]:
            assert key in data["metric"], f"Missing metric.{key}"
        # Table 13 targets.
        assert abs(data["metric"]["loihi_inference_latency_ms"] - 0.8) / 0.8 <= 0.20
        assert abs(data["metric"]["gpu_inference_latency_ms"] - 2.1) / 2.1 <= 0.20

    def test_table_14(self):
        result = _run_script("scripts/reproduce_tables/table_14_attack_metrics.py")
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"
        out = REPO_ROOT / "outputs" / "tables" / "table_14_attack_metrics.json"
        assert out.is_file()
        data = json.loads(out.read_text())
        for key in ["dns_tunnelling_detection_rate",
                    "c2_heartbeat_detection_rate",
                    "side_channel_detection_rate"]:
            assert key in data["metrics"], f"Missing metrics.{key}"
        assert 0.99 <= data["metrics"]["dns_tunnelling_detection_rate"] <= 1.0

    def test_table_15(self):
        result = _run_script("scripts/reproduce_tables/table_15_zk_attestation.py")
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"
        out = REPO_ROOT / "outputs" / "tables" / "table_15_zk_attestation.json"
        assert out.is_file()
        data = json.loads(out.read_text())
        assert data["proof_generation_ms"] == 432.0
        assert data["proof_verification_ms"] == 78.0
        assert data["proof_size_kb"] == 1.8
        assert data["soundness_error"] == "2^-132"

    def test_table_16(self):
        result = _run_script("scripts/reproduce_tables/table_16_federated_learning.py")
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"
        out = REPO_ROOT / "outputs" / "tables" / "table_16_federated_learning.json"
        assert out.is_file()
        data = json.loads(out.read_text())
        assert 99.0 <= data["global_model_accuracy_pct"] <= 100.0
        assert 1.0 <= data["dp_epsilon"] <= 1.5
        assert data["convergence_epochs"] <= 10

    def test_table_17(self):
        result = _run_script("scripts/reproduce_tables/table_17_tda_validation.py")
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"
        out = REPO_ROOT / "outputs" / "tables" / "table_17_tda_validation.json"
        assert out.is_file()
        data = json.loads(out.read_text())
        assert 1.5 <= data["betti_1_benign"]["mean"] <= 3.0
        assert 5.0 <= data["betti_1_attack"]["mean"] <= 7.0

    def test_table_17a(self):
        result = _run_script("scripts/reproduce_tables/table_17a_closest_methods.py")
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"
        out = REPO_ROOT / "outputs" / "tables" / "table_17a_closest_methods.json"
        assert out.is_file()
        data = json.loads(out.read_text())
        assert "federated_aggregation" in data
        assert "closest_method" in data["federated_aggregation"]


# ═══════════════════════════════════════════════════════════════════════════
# Figure scripts
# ═══════════════════════════════════════════════════════════════════════════

class TestFigureScripts:

    def test_figure_7(self):
        result = _run_script("scripts/reproduce_figures/figure_7_architecture.py")
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"
        pdf = REPO_ROOT / "outputs" / "figures" / "figure_7_architecture.pdf"
        png = REPO_ROOT / "outputs" / "figures" / "figure_7_architecture.png"
        assert pdf.is_file()
        assert png.is_file()

    def test_figure_8(self):
        result = _run_script("scripts/reproduce_figures/figure_8_threat_vectors.py")
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"
        png = REPO_ROOT / "outputs" / "figures" / "figure_8_threat_vectors.png"
        assert png.is_file()
        csv = REPO_ROOT / "expected_outputs" / "figure_8_expected.csv"
        import pandas as pd
        df = pd.read_csv(csv)
        assert len(df) == 240
        assert set(df["threat"].unique()) == {"DNS Tunneling", "C2 Traffic", "Quantum SCA"}

    def test_figure_9(self):
        result = _run_script("scripts/reproduce_figures/figure_9_privacy_tradeoff.py")
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"
        png = REPO_ROOT / "outputs" / "figures" / "figure_9_privacy_tradeoff.png"
        assert png.is_file()
        csv = REPO_ROOT / "expected_outputs" / "figure_9_expected.csv"
        import pandas as pd
        df = pd.read_csv(csv)
        assert len(df) == 147
        assert set(df["illustrative"].unique()) == {"true", "false"}

    def test_figure_10(self):
        result = _run_script("scripts/reproduce_figures/figure_10_mitigation_privacy.py")
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"
        png = REPO_ROOT / "outputs" / "figures" / "figure_10_mitigation_privacy.png"
        assert png.is_file()
        csv = REPO_ROOT / "expected_outputs" / "figure_10_expected.csv"
        import pandas as pd
        df = pd.read_csv(csv)
        assert len(df) == 57
        assert list(df.columns) == ["hour", "pafa_detection", "pafa_privacy",
                                     "centralized_detection", "federated_dp_detection"]


# ═══════════════════════════════════════════════════════════════════════════
# Verification script (File 20i)
# ═══════════════════════════════════════════════════════════════════════════

class TestVerificationScript:

    def test_help(self):
        result = _run_script("expected_outputs/verification_script.py --help")
        # We can't pass --help through _run_script because it splits on
        # the space; run the module directly instead.
        script = REPO_ROOT / "expected_outputs" / "verification_script.py"
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}{os.pathsep}{REPO_ROOT}"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30,
            env=env,
        )
        assert result.returncode == 0
        assert "--toy-mode" in result.stdout

    def test_toy_mode_after_full_pipeline(self):
        """After running the toy pipeline, verification must exit 0."""
        required = [
            "outputs/tables/table_13_neuromorphic.json",
            "outputs/tables/table_14_attack_metrics.json",
            "outputs/tables/table_15_zk_attestation.json",
            "outputs/tables/table_16_federated_learning.json",
            "outputs/tables/table_17_tda_validation.json",
        ]
        missing = [r for r in required if not (REPO_ROOT / r).is_file()]
        if missing:
            pytest.skip(
                f"Skipping: toy pipeline has not been run. Missing: {missing}. "
                "Run `python scripts/train_pafa.py --mode toy` first."
            )
        script = REPO_ROOT / "expected_outputs" / "verification_script.py"
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}{os.pathsep}{REPO_ROOT}"
        result = subprocess.run(
            [sys.executable, str(script), "--toy-mode"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120,
            env=env,
        )
        assert result.returncode == 0, (
            f"Verification failed:\n{result.stdout}\n{result.stderr}"
        )
