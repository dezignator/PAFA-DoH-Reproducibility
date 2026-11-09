#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# setup.py — PAFA-DoH Reproducibility Artifact
# ═══════════════════════════════════════════════════════════════════════════
#
# Companion to:
#   README.md                       (File 1 of 74)
#   REPRODUCIBILITY_CHECKLIST.md    (File 2 of 74)
#   LICENSE                         (File 3 of 74)
#   requirements.txt                (File 4 of 74)
#   environment.yml                 (File 5 of 74)
#   Dockerfile                      (File 6 of 74)
#
# Thesis chapter: Chapter 4 (PAFA-DoH)
# Citation DOI:   10.1016/j.neunet.2025.108343   (Neural Networks, April 2026)
# License:        MIT (this file, as part of the artifact's source code)
# Artifact version: 1.0.0
#
# ───────────────────────────────────────────────────────────────────────────
# What this file does
# ───────────────────────────────────────────────────────────────────────────
# 1. Exposes the source tree under `src/` as an installable Python package
#    named `pafa`, so that `pip install -e .` makes
#
#        from pafa.models.snn_detector import SNNDetector
#        from pafa.utils.math_utils   import dp_sgd_step
#        from pafa.attacks.dns_tunneling import generate
#
#    work from any directory on the machine.
#
# 2. Registers four console scripts — `pafa-fl`, `pafa-snn`, `pafa-tda`,
#    `pafa-zk` — pointing at the `main()` entry points in
#    src/cli_tools/federated_cli.py, snn_cli.py, tda_cli.py and zk_cli.py
#    (Files 15b–15e of the 74-file plan). After `pip install -e .`, these
#    scripts are on the PATH and can be invoked directly:
#
#        pafa-fl  train --clients 100 --epsilon 1.2 --rounds 100
#        pafa-snn infer --loihi-simulate
#        pafa-tda betti --scale 50
#        pafa-zk  prove --constraints 100000
#
#    The exact command strings are the ones in Table 14 of the thesis and
#    in src/cli_tools/run_all_scenarios.sh (File 15f).
#
# 3. Declares install-time dependencies by *reading* requirements.txt, so
#    that the two files cannot drift. Editing a pin in one place
#    (requirements.txt) is enough; setup.py does not duplicate the pins.
#
# 4. Installs the repo-root YAML configs (Files 9–9h) and the two header-
#    only toy-dataset CSVs (Files 10h–10i) into `<sys.prefix>/share/pafa-doh/`
#    so that a wheel install still has access to them. The resolution
#    contract that lets the runtime find these files is documented at the
#    bottom of this file.
#
# ───────────────────────────────────────────────────────────────────────────
# What this file deliberately does NOT do
# ───────────────────────────────────────────────────────────────────────────
# • It does not declare `setup_requires` (deprecated since setuptools 46;
#   build-time dependencies belong in the Docker builder stage, which is
#   already set up for them — see File 6).
# • It does not include a `pyproject.toml`. This file is a classical
#   `setuptools` setup.py driven entirely by the pins in requirements.txt.
#   A `pyproject.toml` would be a separate file outside the 74-file plan;
#   if the user adds one later, `build-system.requires = ["setuptools",
#   "wheel"]` is sufficient and no other change to this file is needed.
# • It does not declare `python_requires='>=3.11'` without an upper bound.
#   The pins were validated against 3.11.9 only; 3.12 changes the ABI of
#   several compiled extensions (ripser, gudhi) and is intentionally
#   excluded.
# • It does not upload to PyPI. This artifact is meant to be installed
#   from the repository tree (`pip install -e .`), from a GitHub Release
#   tarball, or built as a wheel by CI. The metadata below is complete
#   enough that a future `twine upload` would work, but uploading is a
#   maintainer decision, not an artifact capability.
#
# ───────────────────────────────────────────────────────────────────────────
# Blind-review safety — placeholders to replace before publishing
# ───────────────────────────────────────────────────────────────────────────
# The thesis is currently under blind review. The four PLACEHOLDER_*
# constants below are deliberately written as `<...>` tokens so that no
# author identity leaks into the package metadata. Before flipping the
# repository to public, replace each one:
#
#   PLACEHOLDER_AUTHOR       → legal author name
#   PLACEHOLDER_AUTHOR_EMAIL → contact email
#   PLACEHOLDER_ORG          → GitHub org / user name
#   PLACEHOLDER_YEAR         → copyright year (currently hard-coded to 2026
#                              in LICENSE; this constant is only used in
#                              project_urls and MUST match LICENSE)
#
# A grep for "PLACEHOLDER_" in this file should return exactly the four
# assignments below. The repository's CI (once configured) should fail
# a release build if any `PLACEHOLDER_` token is still present in the
# built wheel metadata.
# ═══════════════════════════════════════════════════════════════════════════

import os
import pathlib
import re
import sys

from setuptools import find_packages, setup


# ─── Blind-review placeholders (replace before publishing) ──────────────────
PLACEHOLDER_AUTHOR       = "<Author Name — To Be Filled In>"
PLACEHOLDER_AUTHOR_EMAIL = "<author@example.com>"
PLACEHOLDER_ORG          = "<your-org>"
PLACEHOLDER_YEAR         = "2026"


# ─── Constants ──────────────────────────────────────────────────────────────
HERE              = pathlib.Path(__file__).parent.resolve()
ARTIFACT_NAME     = "pafa-doh"
ARTIFACT_VERSION  = "1.0.0"
PYTHON_REQUIRES   = ">=3.11,<3.12"
CITATION_DOI      = "10.1016/j.neunet.2025.108343"
THESIS_CHAPTER    = "Chapter 4 (PAFA-DoH)"


# ─── Helpers ────────────────────────────────────────────────────────────────
def read_requirements(path: pathlib.Path = HERE / "requirements.txt") -> list[str]:
    """
    Read `requirements.txt` and return the list of pinned package
    specifiers, stripped of comments and blank lines.

    This is how `install_requires` is populated. The design goal is that
    `requirements.txt` (File 4 of 74) is the single source of truth for
    every pinned version, and that this file never has to be edited when
    a pin changes.

    Rules applied to each line:
      * Strip anything after `#`.
      * Skip blank lines.
      * Skip pip-only directives (`--extra-index-url`, `-r ...`, `-e ...`).
      * Keep everything else verbatim, including `==` pins and the
        optional `py-solc-x==2.0.3` line if the user has uncommented it
        for the Solidity circuit path.

    Raises:
        RuntimeError: if `requirements.txt` is missing from the source
            tree. A missing requirements.txt in an artifact repository is
            a packaging bug; failing loudly here is safer than silently
            building a wheel with no dependencies.
    """
    if not path.is_file():
        raise RuntimeError(
            f"requirements.txt not found at {path}. "
            "setup.py must be run from the repository root (or from a "
            "source tree that includes requirements.txt)."
        )

    requires: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = re.sub(r"#.*$", "", raw_line).strip()
        if not stripped:
            continue
        if stripped.startswith("--") or stripped.startswith("-r") or stripped.startswith("-e"):
            # pip-only directives are not install_requires entries.
            continue
        requires.append(stripped)
    return requires


def read_long_description() -> str:
    """Return README.md content as the PyPI long_description."""
    readme = HERE / "README.md"
    if not readme.is_file():
        return (
            f"Reproducibility artifact for {THESIS_CHAPTER}. "
            f"See the repository's README.md for the full mapping of "
            f"tables and figures to scripts."
        )
    return readme.read_text(encoding="utf-8")


def discover_packages():
    """
    Discover packages under `src/` and rename each one to its `pafa.*`
    equivalent, so that the installed layout exposes `pafa` as the top-
    level package even though the on-disk layout has no `src/pafa/`
    directory.

    Input layout:
        src/__init__.py
        src/utils/__init__.py
        src/models/__init__.py
        src/baselines/__init__.py
        src/attacks/__init__.py
        src/cli_tools/__init__.py
        src/hardware_simulation/__init__.py

    Output layout (in site-packages):
        pafa/__init__.py
        pafa/utils/__init__.py
        pafa/models/__init__.py
        pafa/baselines/__init__.py
        pafa/attacks/__init__.py
        pafa/cli_tools/__init__.py
        pafa/hardware_simulation/__init__.py

    Returns:
        (packages, package_dir) — the two arguments to pass to setup().
    """
    found = find_packages(where="src")
    packages = ["pafa"] + [f"pafa.{pkg}" for pkg in found]
    package_dir = {"pafa": "src"}
    for pkg in found:
        package_dir[f"pafa.{pkg}"] = os.path.join("src", *pkg.split("."))
    return packages, package_dir


def discover_data_files():
    """
    Build the `data_files` list. Places the YAML configs and the two
    header-only toy-dataset CSVs under `<sys.prefix>/share/pafa-doh/` so
    that a wheel install still has access to them. The on-disk filenames
    are preserved so that the runtime resolution contract in
    src/utils/logger.py (File 11d) can use `path.name` without a lookup
    table.

    The `data_files` mechanism is used (rather than `package_data`)
    because these files live at the repository root, not under `src/`.
    For an `-e` install or a run from the repo tree, the on-disk paths
    are found directly and `data_files` is only a fallback. See the
    footer for the resolution order.
    """
    data_files = []

    configs_dir = HERE / "configs"
    if configs_dir.is_dir():
        yaml_files = sorted(str(p.relative_to(HERE)) for p in configs_dir.glob("*.yaml"))
        if yaml_files:
            data_files.append((os.path.join("share", "pafa-doh", "configs"), yaml_files))

    toy_dir = HERE / "data" / "toy_dataset"
    if toy_dir.is_dir():
        toy_files = sorted(str(p.relative_to(HERE)) for p in toy_dir.glob("*.csv"))
        if toy_files:
            data_files.append((os.path.join("share", "pafa-doh", "data", "toy_dataset"), toy_files))

    # CHECKSUMS.txt and data/README*.md, if present.
    checksums = HERE / "data" / "CHECKSUMS.txt"
    if checksums.is_file():
        data_files.append(
            (os.path.join("share", "pafa-doh", "data"), [str(checksums.relative_to(HERE))])
        )

    return data_files


# ─── Package discovery (computed before setup()) ────────────────────────────
PACKAGES, PACKAGE_DIR = discover_packages()
DATA_FILES = discover_data_files()
INSTALL_REQUIRES = read_requirements()
LONG_DESCRIPTION = read_long_description()


# ─── Setup ──────────────────────────────────────────────────────────────────
setup(
    # ── Identity ────────────────────────────────────────────────────────────
    name=ARTIFACT_NAME,
    version=ARTIFACT_VERSION,
    description=(
        f"Reproducibility artifact for {THESIS_CHAPTER} of the thesis "
        "'Next-Generation Secure Encrypted DNS Architectures: Four "
        "Complementary Frameworks for Adaptive, Privacy-Preserving, and "
        "Quantum-Resilient Defense'."
    ),
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",

    # ── Authorship (blind-review placeholders — replace before publishing) ──
    author=PLACEHOLDER_AUTHOR,
    author_email=PLACEHOLDER_AUTHOR_EMAIL,
    license="MIT",

    # ── Source layout ───────────────────────────────────────────────────────
    packages=PACKAGES,
    package_dir=PACKAGE_DIR,
    include_package_data=True,
    # The runtime reads files from the filesystem (configs, toy CSVs,
    # expected outputs); a zip install would break those lookups.
    zip_safe=False,

    # ── Python version ──────────────────────────────────────────────────────
    python_requires=PYTHON_REQUIRES,

    # ── Dependencies (single source of truth = requirements.txt) ────────────
    install_requires=INSTALL_REQUIRES,
    # setup_requires is deprecated (setuptools >= 46) and is intentionally
    # left out. Build-time requirements are handled by the Docker builder
    # stage (File 6) and, for a bare-metal install, by `pip install
    # setuptools wheel` before `pip install -e .`.

    # ── Console scripts (Files 15b–15e of the 74-file plan) ─────────────────
    entry_points={
        "console_scripts": [
            # Federated learning coordinator (Table 16 of the thesis).
            "pafa-fl  = pafa.cli_tools.federated_cli:main",
            # Spiking-neural-network detector (Table 13 / §4.9–4.12).
            "pafa-snn = pafa.cli_tools.snn_cli:main",
            # Topological data analysis detector (Table 17 / §4.7–4.8).
            "pafa-tda = pafa.cli_tools.tda_cli:main",
            # Zero-knowledge attestation (Table 15 / §4.15–4.18).
            "pafa-zk  = pafa.cli_tools.zk_cli:main",
        ],
    },

    # ── Repo-root data files (see footer for the resolution contract) ───────
    data_files=DATA_FILES,

    # ── Discovery metadata ──────────────────────────────────────────────────
    keywords=[
        "network security",
        "encrypted DNS",
        "DNS-over-HTTPS",
        "DoH",
        "federated learning",
        "differential privacy",
        "spiking neural network",
        "topological data analysis",
        "zero-knowledge",
        "reproducibility",
        "PAFA-DoH",
    ],

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Topic :: System :: Networking",
    ],

    # ── Project URLs (blind-review placeholders — replace before publishing) ─
    project_urls={
        "Source":         f"https://github.com/{PLACEHOLDER_ORG}/PAFA-DoH-Reproducibility",
        "Documentation":  f"https://github.com/{PLACEHOLDER_ORG}/PAFA-DoH-Reproducibility/blob/main/README.md",
        "Reproducibility Checklist": (
            f"https://github.com/{PLACEHOLDER_ORG}/PAFA-DoH-Reproducibility/"
            "blob/main/REPRODUCIBILITY_CHECKLIST.md"
        ),
        "Thesis Citation (DOI)": f"https://doi.org/{CITATION_DOI}",
        "Dataset (Zenodo)":      "https://doi.org/10.5281/zenodo.5957677",
        "Dataset (CIC-DoH2023)": "https://www.unb.ca/cic/datasets/dohbrw-2023.html",
    },
)


# ═══════════════════════════════════════════════════════════════════════════
# Runtime file-resolution contract (implemented in src/utils/logger.py,
# File 11d of the 74-file plan)
# ═══════════════════════════════════════════════════════════════════════════
#
# The artifact needs to locate three kinds of resource at run time:
#
#   configs/<name>.yaml          — the eight YAML configs (Files 9–9h)
#   data/toy_dataset/*.csv       — the two header-only toy CSVs (Files 10h–10i)
#   data/CHECKSUMS.txt           — dataset checksum manifest (File 10j)
#
# Three install topologies are supported, and the resolution order is
# the same for all three resource kinds:
#
#   1. Environment variable override. If `$PAFA_ROOT` is set, it is
#      treated as the repository root and every path is resolved
#      relative to it. This is the mechanism that Docker users will
#      use when they mount `data/raw`, `data/processed` and `outputs`
#      as volumes (see File 6 §“Full-mode reproduction”).
#
#   2. Repository-tree install (`pip install -e .`). When the package
#      is installed in editable mode, `pathlib.Path(__file__).parent`
#      of any `pafa.*` module resolves to `<repo>/src/<...>`, so
#      `<repo>` is the parent of `src/` and the resource paths are
#      resolved directly. This is the common case for artifact
#      reviewers.
#
#   3. Wheel install (`pip install pafa-doh`). The wheel is built with
#      `data_files` entries that place the same files under
#      `<sys.prefix>/share/pafa-doh/{configs,data}/`. The resolver
#      checks this location as the last fallback.
#
# A resource is considered present at the first path in the list that
# exists; if none exists, the resolver raises a `FileNotFoundError`
# whose message names the resource and every path it tried. There is
# no silent fallback: a missing config is a bug, not a degraded mode.
#
# Cross-reference for reviewers:
#   File 11d (src/utils/logger.py)   — implements the resolver
#   File 9  (configs/default_pafa_config.yaml) — the master config
#   File 10h–10i (data/toy_dataset/*.csv)     — toy-demo inputs
#   File 10j (data/CHECKSUMS.txt)             — download integrity
#
# ───────────────────────────────────────────────────────────────────────────
# Development-install quick-check (run after upload)
# ───────────────────────────────────────────────────────────────────────────
# From the repository root:
#
#     python -m venv .venv
#     source .venv/bin/activate
#     pip install -e .
#     python -c "import pafa; print(pafa.__version__ if hasattr(pafa, '__version__') else 'pafa import ok')"
#     python -c "from pafa.models.snn_detector import SNNDetector; print('SNN import ok')"
#     which pafa-fl pafa-snn pafa-tda pafa-zk
#
# The last line should print four paths under `.venv/bin/` (or the
# equivalent on Windows). If any of the four imports fails, the most
# likely cause is a missing `__init__.py` in one of the `src/` sub-
# directories listed in File 11a–16a of the 74-file plan; add the
# missing file and re-run `pip install -e .`.
#
# Full pipeline check (toy mode, no network):
#
#     python data/make_toy_dataset.py --mode synthetic --seed 42
#     python scripts/train_pafa.py --config configs/default_pafa_config.yaml --mode toy
#     python expected_outputs/verification_script.py --toy-mode
#
# Exit code 0 on the last command is the acceptance criterion for the
# artifact on a fresh machine. See README.md §3 and §4 for what the
# toy-mode numbers mean (approximately 85–90 % accuracy; NOT the
# Chapter 4 headline figures).
#
# ───────────────────────────────────────────────────────────────────────────
# Maintenance note
# ───────────────────────────────────────────────────────────────────────────
# When a new YAML file is added under configs/, or a new CSV is added
# under data/toy_dataset/, no change to this file is required: the
# `discover_data_files()` helper globs both directories at build time.
# When a new sub-package is added under src/, no change is required
# either: `discover_packages()` re-runs `find_packages(where='src')`
# at build time. The only edits that require touching this file are
# (a) adding a new console-script entry point, (b) adding a new
# top-level data directory that is not configs/ or data/, or (c)
# changing the package name, version or Python-version band.
#
# The artifact version in this file (ARTIFACT_VERSION = "1.0.0") must
# stay in sync with:
#   • README.md §Status banner
#   • REPRODUCIBILITY_CHECKLIST.md §12 (Sign-off)
#   • environment.yml header comment
#   • Dockerfile LABEL org.opencontainers.image.version
#   • setup.py (this constant)
#
# If the five disagree, the repository is in an inconsistent state
# and any release tag built from it should be rejected. A future
# `scripts/check_version_consistency.py` (not in the 74-file plan;
# optional) can automate this check.
# ═══════════════════════════════════════════════════════════════════════════
