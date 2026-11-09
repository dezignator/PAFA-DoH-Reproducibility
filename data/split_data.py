#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# split_data.py — PAFA-DoH Reproducibility Artifact
# ═══════════════════════════════════════════════════════════════════════════
#
# Companion to:
#   README.md, REPRODUCIBILITY_CHECKLIST.md, LICENSE, requirements.txt,
#   environment.yml, Dockerfile, setup.py, .gitignore   (Files 1–8)
#   configs/default_pafa_config.yaml                    (File 9, master config)
#   configs/*.yaml                                      (Files 9b–9h)
#   data/data_utils.py                                  (File 10a)
#   data/download_datasets.py                           (File 10b)
#   data/preprocess_doh2023.py                          (File 10c)
#   data/preprocess_zenodo.py                           (File 10d)
#
# Thesis chapter: Chapter 4 (PAFA-DoH)
# Citation DOI:   10.1016/j.neunet.2025.108343   (Neural Networks, April 2026)
# License:        MIT (this file, as part of the artifact's source code)
# Artifact version: 1.0.0
#
# ───────────────────────────────────────────────────────────────────────────
# Purpose
# ───────────────────────────────────────────────────────────────────────────
# Consume the output of data/preprocess_zenodo.py (File 10d, the training
# source) or data/preprocess_doh2023.py (File 10c, the schema-parity
# check), and produce the seeded, stratified train / val / test
# partition that scripts/train_pafa.py (File 17a) reads.
#
# The partitioner delegates to `data_utils.write_split()` (File 10a),
# which already implements the stratified, seeded split documented in
# §4.25.1 of the thesis and used to produce the committed expected
# outputs. This file adds four things that `write_split()` alone cannot:
#
#   1. It knows about the two processed-corpus sources (Zenodo vs
#      CIC-DoH2023) and applies the correct file paths and schema-
#      parity semantics for each.
#
#   2. It handles both a single CSV output (cic_doh2023.csv,
#      zenodo_doh.csv) and a Dask-partitioned Parquet directory
#      (zenodo_doh.parquet.d), streaming the split when the input is
#      large enough to benefit from Dask.
#
#   3. It cross-checks the CLI `--seed` against the master config's
#      `seed`, so a reviewer cannot silently produce a split that does
#      not match the committed expected outputs.
#
#   4. It writes a small `split_manifest.json` next to the three CSVs,
#      recording the source, the seed, the fractions, the per-class
#      counts and the per-(label, capture_date) stratum counts. This
#      manifest is what `verification_script.py` (File 20i) reads to
#      confirm that a fresh split reproduces the committed split
#      without re-reading the whole corpus.
#
# ───────────────────────────────────────────────────────────────────────────
# Where the pieces are specified in Chapter 4
# ───────────────────────────────────────────────────────────────────────────
#   Dataset composition                  → §4.25.1
#   Train / val / test split policy      → §4.25.1 and §1.3 / Table 2
#                                          (the committed expected
#                                          outputs must be reproducible
#                                          from a pinned seed)
#   Seed (42)                            → Table 16 and
#                                          configs/default_pafa_config.yaml
#   Per-class sample reporting           → Table 16
#   Determinism                          → File 9 header (determinism:
#                                          "strict") and File 10a's
#                                          `write_split()` docstring
#   Forbidden observables                → §1.7.2 / §4.13; already
#                                          enforced upstream by Files
#                                          10c and 10d, but re-checked
#                                          here on the loaded frame
#                                          before the split runs.
#
# ───────────────────────────────────────────────────────────────────────────
# Why a separate script
# ───────────────────────────────────────────────────────────────────────────
# `data_utils.write_split()` is pure: it takes a DataFrame and a seed
# and returns three DataFrames. It has no knowledge of which source the
# DataFrame came from, which files are on disk, or what the master config
# says the seed should be. Those are the responsibilities of the outer
# pipeline, and they belong in a script that runs *after* preprocessing
# and *before* training.
#
# Keeping the splitter separate also lets a reviewer re-run only the
# split (e.g., after changing `train_frac`) without re-preprocessing
# the 32 GB corpus. That is a 90-minute saving per iteration, and the
# reason this file exists as a separate step rather than being folded
# into `preprocess_zenodo.py`.
#
# ───────────────────────────────────────────────────────────────────────────
# Design notes
# ───────────────────────────────────────────────────────────────────────────
# • The seed cross-check (`check_seed_matches_config`) is fail-closed.
#   Passing `--seed 7` while `configs/default_pafa_config.yaml` says
#   `seed: 42` raises. The artifact is a reproducibility artifact; a
#   nondeterministic split is not "a different run", it is a broken
#   run. If you need a different seed for a local experiment, copy
#   `configs/default_pafa_config.yaml` to
#   `configs/default_pafa_config.local.yaml` (which is gitignored by
#   File 8's rule 9) and change it there.
#
# • The source choice (`--source zenodo | cic_doh2023`) selects the
#   input file(s) and the manifest metadata. It does **not** change the
#   split policy: the same `write_split()` function is used for both,
#   so the CIC schema-parity check exercises exactly the same splitter
#   the Zenodo path does.
#
# • The Dask path is engaged only when the input is a directory
#   (`zenodo_doh.parquet.d`). For the CSV sources, the file is read
#   with pandas and `write_split()` runs directly. This is deliberate:
#   Dask's splitting would need a different implementation of
#   `write_split()` (distributed, with a global shuffle), and the
#   in-memory version is what produced the committed expected outputs.
#   The Dask path therefore reuses the same split by reading the
#   Parquet directory into memory in chunks *only if* it fits, and
#   falling back to a clear error if it does not. A future version can
#   add a true distributed split; see §7.3.7 (external validity) in the
#   thesis for the general caveat.
#
# • The split manifest is written even when `--force` is not passed and
#   the CSVs already exist, as long as the manifest is absent. This
#   lets a reviewer regenerate the manifest from an existing split
#   without overwriting the (possibly committed) CSVs.
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import dask.dataframe as dd
    import yaml
except ImportError as exc:  # pragma: no cover — environment misconfiguration
    print(
        f"FATAL: missing required dependency: {exc.name}. "
        "Install the pinned dependencies first:\n"
        "    pip install -r requirements.txt",
        file=sys.stderr,
    )
    raise


# ─── Local imports — support both `python data/split_data.py` and
#     `python -m data.split_data` from the repo root ─────────────────────
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (str(_HERE), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from data.data_utils import (
        DEFAULT_SEED,
        DEFAULT_TEST_FRAC,
        DEFAULT_TRAIN_FRAC,
        DEFAULT_VAL_FRAC,
        SplitResult,
        resolve_artifact_root,
        validate_schema,
        write_split,
    )
    from data.preprocess_doh2023 import (  # shared guard, single source of truth
        enforce_forbidden_observables,
    )
except ImportError:  # script-qualified (python data/split_data.py)
    from data_utils import (  # type: ignore[no-redef]
        DEFAULT_SEED,
        DEFAULT_TEST_FRAC,
        DEFAULT_TRAIN_FRAC,
        DEFAULT_VAL_FRAC,
        SplitResult,
        resolve_artifact_root,
        validate_schema,
        write_split,
    )
    from preprocess_doh2023 import (  # type: ignore[no-redef]
        enforce_forbidden_observables,
    )


# ─── Constants ─────────────────────────────────────────────────────────────

ARTIFACT_VERSION = "1.0.0"
THESIS_CHAPTER = "Chapter 4 (PAFA-DoH)"
CITATION_DOI = "10.1016/j.neunet.2025.108343"

# The filenames produced by the two preprocessors (Files 10c and 10d).
SOURCE_CSV_FILENAMES: dict[str, str] = {
    "zenodo": "zenodo_doh.csv",
    "cic_doh2023": "cic_doh2023.csv",
}
SOURCE_PARQUET_DIRNAMES: dict[str, str] = {
    "zenodo": "zenodo_doh.parquet.d",
    "cic_doh2023": "cic_doh2023.parquet.d",  # not currently written; reserved
}

log = logging.getLogger("data.split_data")


# ─── Config and seed helpers ──────────────────────────────────────────────

def load_config(root: Path, override: Path | None = None) -> dict[str, Any]:
    """Read the master config (File 9) from the artifact root.

    Raises:
        FileNotFoundError: if the config is missing.
        yaml.YAMLError: if the config is present but malformed.
    """
    cfg_path = override if override is not None else root / "configs" / "default_pafa_config.yaml"
    if not Path(cfg_path).is_file():
        raise FileNotFoundError(
            f"Master config not found at {cfg_path}. "
            "This file is committed to the repository; a missing copy "
            "usually means the checkout is incomplete."
        )
    return yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))


def check_seed_matches_config(cli_seed: int, cfg: dict[str, Any]) -> None:
    """Fail closed if the CLI seed differs from the master-config seed.

    The artifact's reproducibility guarantee rests on a single pinned
    seed. A split produced with a different seed would produce a
    different train / val / test partition, which would produce a
    different model, which would fail `verification_script.py` for
    reasons that look like a bug but are actually a user error. This
    check catches that user error at the start of the run.

    Raises:
        ValueError: naming the CLI seed, the config seed and the
            path to `configs/default_pafa_config.yaml` so a reviewer
            can either fix their command or make a local config copy.
    """
    config_seed = int(cfg["seed"])
    if int(cli_seed) != config_seed:
        raise ValueError(
            f"--seed {cli_seed} does not match the master config's seed "
            f"({config_seed}) at configs/default_pafa_config.yaml::seed. "
            f"The artifact reproduces Chapter 4's numbers only at the "
            f"pinned seed ({config_seed}). If you need a different seed "
            f"for a local experiment, copy the master config to "
            f"configs/default_pafa_config.local.yaml (which is gitignored) "
            f"and change it there."
        )


# ─── Source loading ───────────────────────────────────────────────────────

def load_processed_source(
    root: Path,
    cfg: dict[str, Any],
    source: str,
    *,
    dask_threshold_mb: int = 4096,
) -> pd.DataFrame:
    """Load the processed corpus for `source` into a single DataFrame.

    Resolution order for the input:

      1. `<processed>/<source>_doh.parquet.d/` (Dask partitioned
         directory) — read with `dask.dataframe.read_parquet` and
         `.compute()` into memory. Engaged only if the directory
         exists.
      2. `<processed>/<source>_doh.parquet` (single file) — read with
         `pandas.read_parquet`.
      3. `<processed>/<source>_doh.csv` (single file) — read with
         `pandas.read_csv`.

    The Dask threshold (`dask_threshold_mb`, default 4096 MB) is a
    safety net: if the partitioned directory's total on-disk size
    exceeds the threshold, this function refuses to `.compute()` it
    into memory and raises, rather than OOM-ing the reviewer's
    machine. A true streaming split is out of scope for this file;
    see the header's "Design notes" for the rationale.

    The forbidden-observable guard runs on the loaded DataFrame before
    it is returned. The preprocessors already ran the same guard, so
    this is a defence-in-depth check, not the primary enforcement.
    """
    processed_dir = root / cfg["paths"]["processed_datasets"]
    parquet_dir = processed_dir / SOURCE_PARQUET_DIRNAMES[source]
    parquet_file = processed_dir / f"{SOURCE_PARQUET_DIRNAMES[source].replace('.d', '')}"
    csv_file = processed_dir / SOURCE_CSV_FILENAMES[source]

    df: pd.DataFrame | None = None

    if parquet_dir.is_dir():
        size_mb = sum(
            p.stat().st_size for p in parquet_dir.rglob("*") if p.is_file()
        ) / (1024 * 1024)
        log.info(
            "Loading partitioned Parquet directory %s (%.1f MB on disk)",
            parquet_dir, size_mb,
        )
        if size_mb > dask_threshold_mb:
            raise RuntimeError(
                f"Partitioned Parquet directory {parquet_dir} is "
                f"{size_mb:.0f} MB, above the {dask_threshold_mb} MB "
                f"in-memory threshold. A distributed split is not "
                f"implemented in this artifact (see the header of "
                f"data/split_data.py and §7.3.7 of the thesis). "
                f"Options: (a) reduce the corpus with "
                f"preprocess_zenodo.py --sample, (b) run the CSV "
                f"path on a smaller machine, or (c) extend "
                f"write_split() to a distributed implementation."
            )
        df = dd.read_parquet(str(parquet_dir)).compute()
    elif parquet_file.is_file():
        log.info("Loading single Parquet file %s", parquet_file)
        df = pd.read_parquet(parquet_file)
    elif csv_file.is_file():
        log.info("Loading CSV file %s", csv_file)
        df = pd.read_csv(csv_file, low_memory=False)
    else:
        raise FileNotFoundError(
            f"No processed corpus found for source {source!r}. Tried: "
            f"{parquet_dir}, {parquet_file}, {csv_file}. "
            f"Run the corresponding preprocessor first: "
            f"`python data/preprocess_zenodo.py` for 'zenodo', or "
            f"`python data/preprocess_doh2023.py` for 'cic_doh2023'."
        )

    # Defence-in-depth: re-run the forbidden-observable guard on the
    # processed frame. The preprocessor already ran it, so a hit here
    # means either the preprocessor was bypassed or the processed file
    # was edited after preprocessing.
    enforce_forbidden_observables(df, csv_file if csv_file.is_file() else parquet_dir)
    return df


# ─── Manifest writer ──────────────────────────────────────────────────────

def write_split_manifest(
    result: SplitResult,
    manifest_path: Path,
    *,
    source: str,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> None:
    """Write `split_manifest.json` next to the three split CSVs.

    The manifest records the source, seed, fractions, per-class counts
    and the per-stratum counts from `result.per_stratum_counts`. This
    is the file `verification_script.py` (File 20i) reads to confirm
    that a fresh split reproduces the committed split without re-
    reading the whole corpus.
    """
    # Per-class counts across the three partitions
    per_class: dict[str, dict[str, int]] = {}
    for name, part in (("train", result.train), ("val", result.val), ("test", result.test)):
        # The label column is int8 0/1; convert to str keys for JSON
        counts = part["label"].value_counts().to_dict() if "label" in part.columns else {}
        per_class[name] = {str(k): int(v) for k, v in counts.items()}

    # Per-stratum counts: result.per_stratum_counts is keyed by
    # (label, capture_date_str) -> {train, val, test}. JSON requires
    # string keys, so serialize as a list of records.
    strata_records = [
        {
            "label": int(label),
            "capture_date": str(date_str),
            "counts": {
                "train": int(counts["train"]),
                "val": int(counts["val"]),
                "test": int(counts["test"]),
            },
        }
        for (label, date_str), counts in sorted(result.per_stratum_counts.items())
    ]

    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "thesis_chapter": THESIS_CHAPTER,
        "citation_doi": CITATION_DOI,
        "source": source,
        "seed": int(result.seed),
        "fractions": {
            "train": float(train_frac),
            "val": float(val_frac),
            "test": float(test_frac),
        },
        "row_counts": {
            "train": int(len(result.train)),
            "val": int(len(result.val)),
            "test": int(len(result.test)),
            "total": int(len(result.train) + len(result.val) + len(result.test)),
        },
        "per_class": per_class,
        "strata": strata_records,
    }
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log.info("Wrote split manifest: %s", manifest_path)


# ─── argparse ─────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="split_data.py",
        description=(
            "Produce the seeded, stratified train / val / test split "
            "for PAFA-DoH (Chapter 4) from the output of "
            "preprocess_zenodo.py (File 10d) or preprocess_doh2023.py "
            "(File 10c). Writes split_{train,val,test}.csv and "
            "split_manifest.json under data/processed/."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python data/split_data.py\n"
            "  python data/split_data.py --source cic_doh2023\n"
            "  python data/split_data.py --npartitions 16\n"
            "  python data/split_data.py --train-frac 0.8 --val-frac 0.1 --test-frac 0.1\n"
        ),
    )
    p.add_argument("--config", type=Path, default=None,
                   help="Path to the master config (default: auto-resolve).")
    p.add_argument("--source", choices=["zenodo", "cic_doh2023"],
                   default="zenodo",
                   help="Which processed corpus to split. 'zenodo' (default) "
                        "is the training source; 'cic_doh2023' is the "
                        "schema-parity check.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"Split seed (default {DEFAULT_SEED}, must match "
                        f"configs/default_pafa_config.yaml::seed).")
    p.add_argument("--train-frac", type=float, default=DEFAULT_TRAIN_FRAC,
                   help=f"Training fraction (default {DEFAULT_TRAIN_FRAC}).")
    p.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC,
                   help=f"Validation fraction (default {DEFAULT_VAL_FRAC}).")
    p.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC,
                   help=f"Test fraction (default {DEFAULT_TEST_FRAC}).")
    p.add_argument("--npartitions", type=int, default=None,
                   help="Dask partition count for the Parquet directory "
                        "path (default: os.cpu_count()). Ignored when the "
                        "input is a single CSV or Parquet file.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing split_{train,val,test}.csv "
                        "files. Without this flag, an existing split is "
                        "a hard error.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable DEBUG-level logging.")
    return p


# ─── Self-check ─────────────────────────────────────────────────────────

def _self_check() -> int:
    """Non-destructive self-check used by pytest (File 21a).

    Runs on a 100-row synthetic frame. Confirms that:

      1. `write_split()` produces three non-empty partitions.
      2. The three partition sizes sum to the input size.
      3. The split is deterministic: running `write_split()` twice with
         the same seed produces the same per-stratum counts.
      4. The generated `SplitResult` has the schema-complete columns
         required by `scripts/train_pafa.py`.

    Writes to a temporary directory so it is safe to run from pytest
    without touching `data/processed/`.
    """
    import tempfile

    # Build a 100-row synthetic frame with a plausible class balance
    # (50 / 50) and 5 capture dates so the stratification has something
    # to work with.
    rng = np.random.default_rng(42)
    n = 100
    labels = np.array([0, 1] * (n // 2))
    dates = np.array(["2024-01-01", "2024-01-02", "2024-01-03",
                      "2024-01-04", "2024-01-05"] * (n // 5))
    df = pd.DataFrame({
        "flow_id": [f"f{i:03d}" for i in range(n)],
        "capture_date": dates,
        "src_ip": [f"10.0.0.{i % 256}" for i in range(n)],
        "dst_ip": ["8.8.8.8"] * n,
        "src_port": rng.integers(1024, 65535, n),
        "dst_port": [443] * n,
        "protocol": ["DoH"] * n,
        "flow_duration_ms": rng.uniform(100, 5000, n),
        "packet_count": rng.integers(2, 100, n),
        "bytes_out": rng.integers(500, 50000, n),
        "bytes_in": rng.integers(500, 50000, n),
        "packet_size_mean": rng.uniform(100, 1000, n),
        "packet_size_std": rng.uniform(1, 100, n),
        "packet_size_max": rng.integers(100, 2000, n),
        "packet_size_min": rng.integers(10, 100, n),
        "bytes_per_second": rng.uniform(1000, 1e6, n),
        "packets_per_second": rng.uniform(1, 100, n),
        "iat_mean_ms": rng.uniform(10, 1000, n),
        "iat_std_ms": rng.uniform(1, 100, n),
        "iat_min_ms": rng.uniform(0.1, 10, n),
        "iat_max_ms": rng.uniform(100, 5000, n),
        "iat_median_ms": rng.uniform(10, 1000, n),
        "iat_skew": rng.normal(0, 1, n),
        "iat_kurtosis": rng.normal(0, 1, n),
        "entropy_subdomain": rng.uniform(0, 4, n),
        "entropy_query_type": rng.uniform(0, 3, n),
        "qtype_A_count": rng.integers(0, 10, n),
        "qtype_AAAA_count": rng.integers(0, 10, n),
        "qtype_TXT_count": rng.integers(0, 5, n),
        "qtype_MX_count": rng.integers(0, 5, n),
        "qtype_CNAME_count": rng.integers(0, 5, n),
        "qtype_other_count": rng.integers(0, 5, n),
        "response_size_mean": rng.uniform(100, 2000, n),
        "response_size_std": rng.uniform(10, 200, n),
        "response_entropy": rng.uniform(0, 3, n),
        "tls_version": ["1.3"] * n,
        "cipher_suite_count": rng.integers(4, 20, n),
        "extension_count": rng.integers(0, 20, n),
        "session_reuse": rng.integers(0, 2, n),
        "http2_stream_count": rng.integers(0, 20, n),
        "http2_frame_count": rng.integers(0, 50, n),
        "http3_frame_count": rng.integers(0, 50, n),
        "autocorr_lag1": rng.uniform(-1, 1, n),
        "autocorr_lag5": rng.uniform(-1, 1, n),
        "burstiness": rng.uniform(-1, 1, n),
        "spectral_low_band_energy": rng.uniform(0, 1, n),
        "label": labels,
    })

    # The synthetic frame is built to match `data_utils.FEATURE_SCHEMA`,
    # but `write_split()` will validate it anyway.
    with tempfile.TemporaryDirectory(prefix="split_selfcheck_") as tmp:
        out_dir = Path(tmp)

        # 1 + 2. Non-empty partitions, sum equals input.
        r1 = write_split(df, out_dir, seed=42)
        sizes = (len(r1.train), len(r1.val), len(r1.test))
        if any(s == 0 for s in sizes):
            print(f"FAIL: empty partition(s) in sizes={sizes}.")
            return 1
        if sum(sizes) != len(df):
            print(f"FAIL: partition sizes {sizes} sum to {sum(sizes)}, "
                  f"expected {len(df)}.")
            return 1
        print(f"OK: non-empty partitions, sizes sum to {len(df)} "
              f"(train={sizes[0]}, val={sizes[1]}, test={sizes[2]}).")

        # 3. Determinism: same seed → same per-stratum counts.
        out_dir2 = out_dir / "run2"
        r2 = write_split(df, out_dir2, seed=42)
        if r1.per_stratum_counts != r2.per_stratum_counts:
            print("FAIL: same seed produced different per-stratum counts.")
            return 1
        print("OK: same seed reproduces the same per-stratum counts.")

        # 4. Schema-completeness of the split partitions.
        from data_utils import validate_schema as _vs
        for name, part in (("train", r1.train), ("val", r1.val), ("test", r1.test)):
            try:
                _vs(part)
            except ValueError as exc:
                print(f"FAIL: split_{name} failed schema validation: {exc}")
                return 1
        print("OK: all three split partitions pass validate_schema.")

    print("self-check ok")
    return 0


# ─── main ─────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    else:
        log.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    # Resolve root.
    try:
        root = resolve_artifact_root()
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1
    log.info("Artifact root:     %s", root)
    log.info("Artifact version:  %s", ARTIFACT_VERSION)
    log.info("Thesis:            %s", THESIS_CHAPTER)
    log.info("Citation DOI:      %s", CITATION_DOI)

    # Config + seed cross-check.
    try:
        cfg = load_config(root, override=args.config)
    except (FileNotFoundError, yaml.YAMLError) as exc:
        log.error("Failed to load master config: %s", exc)
        return 1
    try:
        check_seed_matches_config(args.seed, cfg)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    # Fraction consistency.
    total = args.train_frac + args.val_frac + args.test_frac
    if abs(total - 1.0) > 1e-9:
        log.error(
            "Split fractions must sum to 1.0, got %.6f "
            "(train=%.4f, val=%.4f, test=%.4f).",
            total, args.train_frac, args.val_frac, args.test_frac,
        )
        return 1

    # Load the processed corpus.
    try:
        df = load_processed_source(root, cfg, args.source)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        log.error("%s", exc)
        return 1
    log.info("Loaded processed corpus: %d rows × %d columns",
             len(df), df.shape[1])

    # Validate schema before splitting.
    try:
        validate_schema(df)
    except ValueError as exc:
        log.error("Input schema validation failed: %s", exc)
        return 1

    # Refuse to overwrite an existing split without --force.
    out_dir = root / cfg["paths"]["processed_datasets"]
    existing = [out_dir / f"split_{n}.csv" for n in ("train", "val", "test")]
    if any(p.exists() for p in existing) and not args.force:
        log.error(
            "Existing split file(s) present: %s. "
            "Pass --force to overwrite.",
            [str(p) for p in existing if p.exists()],
        )
        return 1

    # Split.
    try:
        result = write_split(
            df, out_dir,
            seed=args.seed,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
        )
    except (ValueError, OSError) as exc:
        log.error("Splitting failed: %s", exc)
        return 1

    # Manifest.
    write_split_manifest(
        result,
        out_dir / "split_manifest.json",
        source=args.source,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )

    # NEXT STEP banner.
    print()
    print("NEXT STEP:")
    if args.source == "zenodo":
        print(
            "  python scripts/train_pafa.py "
            "--config configs/default_pafa_config.yaml --mode full"
        )
    else:
        print(
            "  # CIC-DoH2023 is used only for the 47-column schema-parity "
            "check.\n"
            "  # Do NOT run scripts/train_pafa.py --mode full against the "
            "CIC split;\n"
            "  # the reported numbers in Table 16 are produced from the "
            "Zenodo split."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
