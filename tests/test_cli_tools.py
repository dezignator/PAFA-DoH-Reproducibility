# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# tests/test_cli_tools.py — PAFA-DoH Reproducibility Artifact
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
# Unit tests for the four command-line tools in src/cli_tools/
# (Files 15b–15e) and the shell replay script (File 15f).
#
# Each CLI is tested at three levels:
#   1. The module exposes a `build_parser()` and `main(argv)` function.
#   2. `main(["--help"])` exits via argparse (raises SystemExit with code 0).
#   3. Each sub-command runs against synthetic input and returns 0.
#
# The shell replay script is tested at the syntax level (`bash -n`).
# Running it end-to-end requires the toy dataset to exist, which the
# artifact deliberately does not generate during `pytest`.
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


# ═══════════════════════════════════════════════════════════════════════════
# Generic helpers
# ═══════════════════════════════════════════════════════════════════════════

def _help_raises_systemexit(module_name: str) -> None:
    """`main(["--help"])` must exit via argparse with code 0."""
    mod = __import__(module_name, fromlist=["main"])
    with pytest.raises(SystemExit) as exc_info:
        mod.main(["--help"])
    assert exc_info.value.code == 0


# ═══════════════════════════════════════════════════════════════════════════
# 1. federated_cli (File 15b)
# ═══════════════════════════════════════════════════════════════════════════

class TestFederatedCLI:

    def test_help(self):
        _help_raises_systemexit("pafa.cli_tools.federated_cli")

    def test_train(self):
        from pafa.cli_tools.federated_cli import main
        rc = main([
            "train", "--clients", "10", "--epsilon", "1.2", "--rounds", "2",
        ])
        assert rc == 0

    def test_aggregate(self):
        from pafa.cli_tools.federated_cli import main
        rc = main(["aggregate", "--method", "geometric_median"])
        assert rc == 0

    def test_detect_poison(self):
        from pafa.cli_tools.federated_cli import main
        rc = main(["detect-poison", "--mads", "2.5"])
        assert rc == 0

    def test_no_args_requires_subcommand(self):
        from pafa.cli_tools.federated_cli import main
        with pytest.raises(SystemExit):
            main([])


# ═══════════════════════════════════════════════════════════════════════════
# 2. snn_cli (File 15c)
# ═══════════════════════════════════════════════════════════════════════════

class TestSNNCLI:

    def test_help(self):
        _help_raises_systemexit("pafa.cli_tools.snn_cli")

    def test_train(self):
        from pafa.cli_tools.snn_cli import main
        rc = main(["train", "--tau-plus", "20", "--tau-minus", "20"])
        assert rc == 0

    def test_infer_loihi(self):
        from pafa.cli_tools.snn_cli import main
        rc = main(["infer", "--loihi-simulate"])
        assert rc == 0

    def test_encode(self):
        from pafa.cli_tools.snn_cli import main
        rc = main(["encode", "--tau", "0.01", "--alpha", "1.0"])
        assert rc == 0


# ═══════════════════════════════════════════════════════════════════════════
# 3. tda_cli (File 15d)
# ═══════════════════════════════════════════════════════════════════════════

class TestTDACLI:

    def test_help(self):
        _help_raises_systemexit("pafa.cli_tools.tda_cli")

    def test_analyze(self):
        from pafa.cli_tools.tda_cli import main
        rc = main(["analyze", "--homology", "0,1"])
        assert rc == 0

    def test_analyze_bad_homology(self):
        from pafa.cli_tools.tda_cli import main
        rc = main(["analyze", "--homology", "0,x"])
        assert rc == 1

    def test_betti(self):
        from pafa.cli_tools.tda_cli import main
        rc = main(["betti", "--scale", "50"])
        assert rc == 0

    def test_significance(self):
        from pafa.cli_tools.tda_cli import main
        rc = main(["significance", "--resamples", "100"])
        assert rc == 0


# ═══════════════════════════════════════════════════════════════════════════
# 4. zk_cli (File 15e)
# ═══════════════════════════════════════════════════════════════════════════

class TestZKCLI:

    def test_help(self):
        _help_raises_systemexit("pafa.cli_tools.zk_cli")

    def test_prove(self):
        from pafa.cli_tools.zk_cli import main
        rc = main(["prove", "--constraints", "100000"])
        assert rc == 0

    def test_verify_missing_file(self, tmp_path):
        from pafa.cli_tools.zk_cli import main
        rc = main(["verify", "--proof-file", str(tmp_path / "nonexistent.bin")])
        assert rc == 1

    def test_stark_prove(self):
        from pafa.cli_tools.zk_cli import main
        rc = main(["stark-prove", "--blowup", "4", "--queries", "20"])
        assert rc == 0


# ═══════════════════════════════════════════════════════════════════════════
# 5. run_all_scenarios.sh (File 15f) — syntax check only
# ═══════════════════════════════════════════════════════════════════════════

class TestRunAllScenarios:

    def test_shell_syntax(self):
        script = REPO_ROOT / "src" / "cli_tools" / "run_all_scenarios.sh"
        assert script.is_file(), f"Missing shell script at {script}."
        result = subprocess.run(
            ["bash", "-n", str(script)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"bash -n reported a syntax error:\n{result.stderr}"
        )
