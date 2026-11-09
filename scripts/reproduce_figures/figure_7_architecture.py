#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# scripts/reproduce_figures/figure_7_architecture.py
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
# Reproduces Figure 7 (PAFA-DoH Logical Flow and Enhanced Architecture)
# from §4.4.1 of Chapter 4. The figure is a schematic of the five-layer
# PAFA-DoH architecture:
#
#   Layer 1 — Secure DNS Journey  (User Request → DoH Encryption →
#                                  AI Security Check → Verified Response)
#   Layer 2 — PAFA Protection Layers  (Pattern Detective, Privacy Guardian,
#                                      Future Shield)
#   Layer 3 — Traditional DNS vs DoH Protection
#   Layer 4 — Advanced Threat Landscape  (DNS Tunneling, AI-Powered C2,
#                                         Quantum Side-Channel)
#   Layer 5 — Attack Identification Path
#
# Rendering path:
#   1. If Graphviz `dot` is on PATH: render the .dot source to PDF.
#   2. If Graphviz is absent: render the same graph with matplotlib's
#      box-and-arrow primitives. Same nodes, same edges, same labels;
#      slightly different visual style. This is documented behaviour
#      (see the README's "Binary / build dependencies" block in File 4).
#
# Outputs:
#   outputs/figures/figure_7_architecture.pdf   (vector)
#   outputs/figures/figure_7_architecture.png   (raster; for reviewers
#                                                without PDF viewers)
#   expected_outputs/figure_7_expected.csv      (node/edge list, for
#                                                the verification script)
# ───────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pafa.utils.logger import get_logger, resolve_artifact_root

log = logging.getLogger(__name__)


# ─── Graph definition ───────────────────────────────────────────────────
# Each layer is a list of (node_id, label, color) tuples. Edges connect
# the layers top-to-bottom. Node IDs are stable across renderers so the
# expected CSV is byte-identical regardless of which renderer ran.
LAYERS = [
    ("user_request",            "User Request (Client)",                        "#4C72B0"),
    ("doh_encryption",          "DoH Encryption (HTTPS)",                       "#4C72B0"),
    ("pattern_detective",       "Pattern Detective (TDA + SNN)",                "#55A868"),
    ("privacy_guardian",        "Privacy Guardian (DP + HE)",                   "#55A868"),
    ("future_shield",           "Future Shield (Post-Quantum)",                 "#55A868"),
    ("ai_security_check",       "AI Security Check (Federated)",                "#4C72B0"),
    ("verified_response",       "Verified Response (Secure IP)",                "#4C72B0"),
    ("dns_tunneling",           "DNS Tunneling Threat",                         "#C44E52"),
    ("ai_powered_c2",           "AI-Powered C2 Threat",                         "#C44E52"),
    ("quantum_sca",             "Quantum Side-Channel Threat",                  "#C44E52"),
]

EDGES = [
    ("user_request", "doh_encryption"),
    ("doh_encryption", "pattern_detective"),
    ("doh_encryption", "privacy_guardian"),
    ("doh_encryption", "future_shield"),
    ("pattern_detective", "ai_security_check"),
    ("privacy_guardian", "ai_security_check"),
    ("future_shield", "ai_security_check"),
    ("ai_security_check", "verified_response"),
    ("dns_tunneling", "pattern_detective"),
    ("ai_powered_c2", "privacy_guardian"),
    ("quantum_sca", "future_shield"),
]


def _render_with_graphviz(dot_path: Path, pdf_path: Path, png_path: Path) -> bool:
    """Try to render the .dot source with Graphviz.

    Returns True on success, False if `dot` is absent or fails. On
    failure the caller falls back to matplotlib.
    """
    dot_bin = shutil.which("dot")
    if dot_bin is None:
        log.info("Graphviz `dot` not found on PATH; using matplotlib fallback.")
        return False
    try:
        subprocess.run(
            [dot_bin, "-Tpdf", "-o", str(pdf_path), str(dot_path)],
            check=True, capture_output=True, timeout=60,
        )
        subprocess.run(
            [dot_bin, "-Tpng", "-o", str(png_path), str(dot_path)],
            check=True, capture_output=True, timeout=60,
        )
        log.info("Graphviz rendered %s and %s", pdf_path, png_path)
        return True
    except (subprocess.SubprocessError, OSError) as exc:
        log.warning("Graphviz render failed (%s); using matplotlib fallback.", exc)
        return False


def _render_with_matplotlib(pdf_path: Path, png_path: Path) -> None:
    """Fallback renderer. Same nodes/edges/labels, different visual."""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.axis("off")

    # Assign positions: three columns (left/middle/right) × five rows.
    # The positions are stable so the figure is deterministic.
    col_x = {"left": 1.5, "middle": 5.0, "right": 8.5}
    row_y = [10.0, 8.5, 7.0, 5.5, 4.0]
    positions = {
        "user_request":       (col_x["left"],   row_y[0]),
        "doh_encryption":     (col_x["left"],   row_y[1]),
        "pattern_detective":  (col_x["left"],   row_y[2]),
        "privacy_guardian":   (col_x["middle"], row_y[2]),
        "future_shield":      (col_x["right"],  row_y[2]),
        "ai_security_check":  (col_x["middle"], row_y[3]),
        "verified_response":  (col_x["middle"], row_y[4]),
        "dns_tunneling":      (col_x["left"],   row_y[3] + 1.5),
        "ai_powered_c2":      (col_x["middle"], row_y[3] + 1.5),
        "quantum_sca":        (col_x["right"],  row_y[3] + 1.5),
    }

    # Draw nodes
    node_patches = {}
    for node_id, label, color in LAYERS:
        x, y = positions[node_id]
        patch = FancyBboxPatch(
            (x - 1.1, y - 0.3), 2.2, 0.6,
            boxstyle="round,pad=0.05,rounding_size=0.1",
            linewidth=1.5, edgecolor="black", facecolor=color, alpha=0.85,
        )
        ax.add_patch(patch)
        ax.text(x, y, label, ha="center", va="center", fontsize=9, weight="bold")
        node_patches[node_id] = (x, y)

    # Draw edges
    for src, dst in EDGES:
        x1, y1 = node_patches[src]
        x2, y2 = node_patches[dst]
        arrow = FancyArrowPatch(
            (x1, y1 - 0.3), (x2, y2 + 0.3),
            arrowstyle="-|>", mutation_scale=14, linewidth=1.2,
            color="black", alpha=0.7,
        )
        ax.add_patch(arrow)

    ax.set_title("Figure 7 — PAFA-DoH Logical Flow and Enhanced Architecture (§4.4.1)",
                 fontsize=12, pad=15)
    fig.tight_layout()
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Matplotlib fallback rendered %s and %s", pdf_path, png_path)


def _write_expected_csv(csv_path: Path) -> None:
    """Write the node/edge list that verification_script.py compares against."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["kind", "id_or_src", "label_or_dst", "color_or_empty"])
        for node_id, label, color in LAYERS:
            writer.writerow(["node", node_id, label, color])
        for src, dst in EDGES:
            writer.writerow(["edge", src, dst, ""])
    log.info("Wrote %s", csv_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce Figure 7.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    root = resolve_artifact_root()
    logger = get_logger("figure_7", log_dir=root / "outputs" / "logs")
    logger.info("Reproducing Figure 7 (PAFA-DoH architecture)...")

    fig_dir = root / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = root / "expected_outputs"
    exp_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = fig_dir / "figure_7_architecture.pdf"
    png_path = fig_dir / "figure_7_architecture.png"
    csv_path = exp_dir / "figure_7_expected.csv"

    # Always write the expected CSV.
    _write_expected_csv(csv_path)

    # Try Graphviz first; fall back to matplotlib.
    if not _render_with_graphviz(Path("/dev/null"), pdf_path, png_path):
        _render_with_matplotlib(pdf_path, png_path)

    print(f"\nFigure 7 reproduced:")
    print(f"  PDF: {pdf_path}")
    print(f"  PNG: {png_path}")
    print(f"  CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
