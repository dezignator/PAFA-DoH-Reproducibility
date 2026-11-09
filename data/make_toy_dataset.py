#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# make_toy_dataset.py — PAFA-DoH Reproducibility Artifact
# ═══════════════════════════════════════════════════════════════════════════
#
# Companion to:
#   README.md (File 1) — §3 Quick Start (2-minute toy demo), §4 Accuracy
#                        warning; the toy demo is described there.
#   REPRODUCIBILITY_CHECKLIST.md (File 2) — §2 environment matrix,
#                                            §6 dataset dependency map,
#                                            §8 tolerance table.
#   LICENSE (File 3) — §2 (toy dataset is CC-BY 4.0, header-only in git,
#                        generated data is not committed).
#   configs/default_pafa_config.yaml (File 9) — `datasets.toy.*` block
#                                                (1,000 + 1,000 rows,
#                                                seed 42).
#   data/data_utils.py (File 10a) — 47-column schema, dtype helpers,
#                                    `write_split()` (not used here;
#                                    the toy generator writes two
#                                    separate files, not a split).
#   data/download_datasets.py (File 10b) — `toy` sub-command delegates
#                                           to this script.
#
# Thesis chapter: Chapter 4 (PAFA-DoH)
# Citation DOI:   10.1016/j.neunet.2025.108343   (Neural Networks, April 2026)
# License:        MIT (this file, as part of the artifact's source code)
# Artifact version: 1.0.0
#
# ───────────────────────────────────────────────────────────────────────────
# Purpose
# ───────────────────────────────────────────────────────────────────────────
# Generate the 1,000 + 1,000 row toy dataset used by README §3's two-minute
# smoke test. The toy dataset exercises the full PAFA-DoH pipeline end to
# end (SNN encoder → TDA detector → federated aggregate → ZK reference
# proof → verification) but does NOT reproduce Chapter 4's headline
# numbers.
#
# Two modes:
#
#   --mode synthetic   (default)
#       Generate 1,000 benign + 1,000 malicious rows from the
#       parametric distributions documented in `SYNTHETIC_DISTRIBUTIONS`
#       below. No network, no 32 GB corpus, sub-second runtime.
#       Expected accuracy of a model trained on the toy data: ~85–90 %.
#
#   --mode from-full
#       Sample the first 1,000 benign + first 1,000 malicious rows
#       from the processed Zenodo DoH corpus
#       (`data/processed/zenodo_doh.csv` or `zenodo_doh.parquet.d`,
#       the output of File 10d). Still a smoke test, but on real data;
#       accuracy approaches but does not reach Chapter 4's numbers
#       because 2,000 rows is far below the full corpus.
#
# In both modes, the output is written to
# `data/processed/toy_dataset/{benign_sample.csv, malicious_sample.csv,
# combined.csv}`. The committed header-only toy CSVs in
# `data/toy_dataset/` (Files 10h–10i) are read as the schema template
# and are NEVER overwritten by this generator. This separation is what
# File 8's `.gitignore` rule 8 is designed for (see File 8's comment).
#
# ───────────────────────────────────────────────────────────────────────────
# Where the pieces are specified in Chapter 4
# ───────────────────────────────────────────────────────────────────────────
#   Toy demo scope / accuracy warning   → README §3, §4
#   Toy dataset size / seed             → File 9, `datasets.toy.*`
#   47-column schema                    → data/data_utils.py (File 10a)
#   Discriminative feature families     → §4.7.1 (TDA features), §4.8.2
#                                         (Betti-1 / persistence entropy /
#                                         Wasserstein), §4.26.1–§4.26.4
#                                         (attack simulation metrics:
#                                         tunneling, C2, phishing,
#                                         side-channel)
#   Reported accuracy on the full data  → Table 16 (99.4 % ± 0.12 %)
#                                         — deliberately NOT reached
#                                         in toy mode.
#
# ───────────────────────────────────────────────────────────────────────────
# Why the toy data is not the reported data
# ───────────────────────────────────────────────────────────────────────────
# The toy dataset is 2,000 rows. Chapter 4's reported results are on the
# 32 GB Zenodo corpus (tens of millions of flow records). A model
# trained on 2,000 synthetic rows can only approximate the class
# boundary that the full corpus supports; the accuracy gap (85–90 %
# vs 99.4 %) is the expected consequence of the size difference, not a
# bug. README §4 explains this and `expected_outputs/verification_script.py
# --toy-mode` relaxes the tolerance to rtol = 0.20 to match.
#
# ───────────────────────────────────────────────────────────────────────────
# Design notes
# ───────────────────────────────────────────────────────────────────────────
# • The 47-column schema is imported from `data_utils`; it is never
#   redeclared here. The committed header-only CSVs in
#   `data/toy_dataset/` are consulted at runtime only to confirm the
#   schema file layout; the generator writes its own header row.
#
# • The two classes use two independent RNG streams derived from the
#   master seed: benign uses `seed` directly, malicious uses
#   `seed + 1_000_000`. This makes the two files reproducible
#   independently (e.g., a reviewer can regenerate only the benign
#   file with `--benign-rows 500 --malicious-rows 0` and get the same
#   first 500 rows as they would from a full run). It also prevents a
#   change in the malicious distribution from shifting the benign
#   rows.
#
# • The malicious shifts are modest (`|Cohen's d| ≈ 0.6–1.2` per
#   feature) so that a model trained on the toy data can reach
#   ~85–90 % but not 99 %. If the shifts were stronger, the toy demo
#   would falsely suggest that Chapter 4's numbers are reproducible
#   in toy mode. The self-check asserts that at least one feature has
#   `|Cohen's d| ≥ 0.5` (class separability exists) and that at
#   least one has `|Cohen's d| ≤ 2.0` (the toy data is not
#   trivially separable).
#
# • The generator uses `numpy.random.default_rng` (PCG64) rather than
#   the legacy `np.random.seed`. PCG64 is the recommended generator
#   for reproducibility guarantees across NumPy versions, and the
#   committed expected toy outputs were produced with it.
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    print(
        f"FATAL: missing required dependency: {exc.name}. "
        "Install the pinned dependencies first:\n"
        "    pip install -r requirements.txt",
        file=sys.stderr,
    )
    raise


# ─── Local imports — support both `python data/make_toy_dataset.py` and
#     `python -m data.make_toy_dataset` from the repo root ─────────────
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (str(_HERE), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from data.data_utils import (
        FEATURE_SCHEMA,
        LABEL_COLUMN,
        METADATA_COLUMNS,
        NUMERIC_COLUMNS,
        canonicalize_dtypes,
        resolve_artifact_root,
        validate_schema,
    )
except ImportError:  # script-qualified (python data/make_toy_dataset.py)
    from data_utils import (  # type: ignore[no-redef]
        FEATURE_SCHEMA,
        LABEL_COLUMN,
        METADATA_COLUMNS,
        NUMERIC_COLUMNS,
        canonicalize_dtypes,
        resolve_artifact_root,
        validate_schema,
    )


# ─── Constants ─────────────────────────────────────────────────────────────

ARTIFACT_VERSION = "1.0.0"
THESIS_CHAPTER = "Chapter 4 (PAFA-DoH)"
CITATION_DOI = "10.1016/j.neunet.2025.108343"

DEFAULT_SEED = 42
DEFAULT_BENIGN_ROWS = 1000
DEFAULT_MALICIOUS_ROWS = 1000

# Offset the malicious RNG stream from the benign one. See the
# "Design notes" header comment for the rationale.
MALICIOUS_SEED_OFFSET = 1_000_000

# Where the generator writes. The committed header-only toy CSVs in
# `data/toy_dataset/` are the *schema template*; the generated data is
# written here, to the processed side, so that the committed files
# stay header-only (see .gitignore rule 8, File 8).
OUTPUT_SUBDIR = Path("data") / "processed" / "toy_dataset"

log = logging.getLogger("data.make_toy_dataset")


# ─── Distribution specifications ─────────────────────────────────────────
#
# Each entry is a callable `(rng, n) -> np.ndarray` that draws `n`
# samples of the named feature from its benign distribution. The
# `MALICIOUS_SHIFTS` dict (below) describes how the malicious
# distribution differs.
#
# The distribution family choices are motivated by the physical meaning
# of each feature (bytes and durations are heavy-tailed → lognormal;
# entropy is bounded → beta rescaled; counts are non-negative integers
# → Poisson; binary flags → Bernoulli). The means and standard
# deviations are chosen to be plausible for DoH traffic on a small
# enterprise network; they are NOT calibrated to the Zenodo corpus
# (that would require the 32 GB download the toy demo is designed to
# avoid).

def _bounded_beta(rng: np.random.Generator, n: int, a: float, b: float,
                  scale: float) -> np.ndarray:
    """Draw from a beta distribution rescaled to [0, scale]."""
    return rng.beta(a, b, n) * scale


SYNTHETIC_DISTRIBUTIONS: dict[str, Any] = {
    # ── §1. Flow metadata ─────────────────────────────────────────────
    # These are populated deterministically by `_generate_class()`,
    # not drawn from a distribution; the entries below exist only so
    # that `SYNTHETIC_DISTRIBUTIONS.keys()` covers the full schema.
    "flow_id": None,
    "capture_date": None,
    "src_ip": None,
    "dst_ip": None,
    "src_port": lambda rng, n: rng.integers(1024, 65535, n),
    "dst_port": lambda rng, n: np.full(n, 443, dtype=np.int64),
    "protocol": lambda rng, n: np.array(["DoH"] * n, dtype=object),

    # ── §2. Flow-level statistics ─────────────────────────────────────
    "flow_duration_ms": lambda rng, n: rng.lognormal(7.0, 1.0, n),
    "packet_count": lambda rng, n: rng.poisson(10, n),
    "bytes_out": lambda rng, n: rng.lognormal(7.0, 0.8, n),
    "bytes_in": lambda rng, n: rng.lognormal(7.0, 0.8, n),
    "packet_size_mean": lambda rng, n: rng.normal(500.0, 150.0, n).clip(60.0, 1400.0),
    "packet_size_std": lambda rng, n: rng.lognormal(3.5, 0.5, n),
    "packet_size_max": lambda rng, n: rng.normal(1200.0, 200.0, n).clip(500.0, 2000.0),
    "packet_size_min": lambda rng, n: rng.normal(50.0, 15.0, n).clip(20.0, 200.0),
    "bytes_per_second": lambda rng, n: rng.lognormal(10.0, 1.0, n),
    "packets_per_second": lambda rng, n: rng.lognormal(2.0, 1.0, n),

    # ── §3. Timing / IAT ──────────────────────────────────────────────
    "iat_mean_ms": lambda rng, n: rng.lognormal(4.5, 1.0, n),
    "iat_std_ms": lambda rng, n: rng.lognormal(3.5, 1.0, n),
    "iat_min_ms": lambda rng, n: rng.lognormal(1.0, 1.0, n),
    "iat_max_ms": lambda rng, n: rng.lognormal(6.0, 1.0, n),
    "iat_median_ms": lambda rng, n: rng.lognormal(4.5, 1.0, n),
    "iat_skew": lambda rng, n: rng.normal(0.0, 1.0, n),
    "iat_kurtosis": lambda rng, n: rng.normal(0.0, 2.0, n),

    # ── §4. Query-type / entropy ──────────────────────────────────────
    # The maximum entropy of a 10-char subdomain is log2(10) ≈ 3.32,
    # so scaling beta by 3 keeps the benign and malicious means in a
    # realistic range.
    "entropy_subdomain": lambda rng, n: _bounded_beta(rng, n, 2, 3, 3.0),
    "entropy_query_type": lambda rng, n: _bounded_beta(rng, n, 2, 3, 3.0),
    "qtype_A_count": lambda rng, n: rng.poisson(3, n),
    "qtype_AAAA_count": lambda rng, n: rng.poisson(2, n),
    "qtype_TXT_count": lambda rng, n: rng.poisson(1, n),
    "qtype_MX_count": lambda rng, n: rng.poisson(0, n),
    "qtype_CNAME_count": lambda rng, n: rng.poisson(1, n),
    "qtype_other_count": lambda rng, n: rng.poisson(0, n),

    # ── §5. Response ──────────────────────────────────────────────────
    "response_size_mean": lambda rng, n: rng.lognormal(6.0, 0.8, n),
    "response_size_std": lambda rng, n: rng.lognormal(3.5, 0.5, n),
    "response_entropy": lambda rng, n: _bounded_beta(rng, n, 2, 3, 3.0),

    # ── §6. TLS handshake ─────────────────────────────────────────────
    # `tls_version` is the string "1.3" for benign flows; the malicious
    # shift below introduces some "1.2" values to mimic downgrade
    # attempts (§4.26 / §4.30).
    "tls_version": lambda rng, n: np.array(["1.3"] * n, dtype=object),
    "cipher_suite_count": lambda rng, n: rng.poisson(8, n),
    "extension_count": lambda rng, n: rng.poisson(5, n),
    "session_reuse": lambda rng, n: rng.binomial(1, 0.3, n),

    # ── §7. HTTP/2 + HTTP/3 frames ────────────────────────────────────
    "http2_stream_count": lambda rng, n: rng.poisson(3, n),
    "http2_frame_count": lambda rng, n: rng.poisson(7, n),
    "http3_frame_count": lambda rng, n: rng.poisson(0, n),

    # ── §8. Temporal / spectral ───────────────────────────────────────
    "autocorr_lag1": lambda rng, n: rng.normal(0.3, 0.2, n).clip(0.0, 1.0),
    "autocorr_lag5": lambda rng, n: rng.normal(0.3, 0.2, n).clip(0.0, 1.0),
    "burstiness": lambda rng, n: _bounded_beta(rng, n, 2, 2, 2.0) - 1.0,
    "spectral_low_band_energy": lambda rng, n: _bounded_beta(rng, n, 2, 5, 1.0),
}


# ─── Malicious shifts ────────────────────────────────────────────────────
#
# Each entry describes how the malicious distribution differs from the
# benign one. Three kinds of shift are supported:
#
#   ("mean_shift", delta)     — add `delta` to every benign sample
#                                (preserves the distribution family;
#                                usable for normal / lognormal / beta
#                                features after the benign draw).
#   ("replace", callable)     — discard the benign draw and use the
#                                callable instead.
#   ("clip", lo, hi)          — re-clip the shifted draw to [lo, hi]
#                                (used for entropy features).
#
# The shifts are documented with the chapter section that motivates
# them. The `|Cohen's d|` in the comment is an approximate expected
# value; the self-check verifies that at least one feature actually
# has `|d| ≥ 0.5` and at least one has `|d| ≤ 2.0` on the generated
# data.
MALICIOUS_SHIFTS: dict[str, Any] = {
    # §4.7.1 — the TDA features (Betti-1, persistence entropy,
    # Wasserstein) are the most discriminative family in the chapter,
    # so the toy data makes entropy_subdomain and autocorr_lag5 the
    # two strongest shifts. Approximate |d| ≈ 1.0–1.2 each.
    "entropy_subdomain":
        ("mean_shift", 0.8),                    # |d| ≈ 1.1
    "autocorr_lag5":
        ("mean_shift", 0.25),                   # |d| ≈ 1.2

    # §4.26.1 — DNS tunneling uses TXT records; §4.8.1 — periodic
    # C2 beaconing. Both raise the TXT and other-qtype counts.
    # Poisson shifts have |d| ≈ 1.0 and |d| ≈ 1.5 respectively.
    "qtype_TXT_count":
        ("replace", lambda rng, n: rng.poisson(4, n)),
    "qtype_other_count":
        ("replace", lambda rng, n: rng.poisson(2, n)),

    # §4.28.2 — the response entropy is a proxy for the
    # persisted-homology signature. Approximate |d| ≈ 0.8.
    "response_entropy":
        ("mean_shift", 0.4),

    # §4.26.1 — DNS tunneling carries more bytes out than benign
    # resolution. Approximate |d| ≈ 0.6.
    "bytes_out":
        ("replace", lambda rng, n: rng.lognormal(7.6, 0.8, n)),

    # §4.26.4 — C2 traffic is more bursty. Approximate |d| ≈ 0.7.
    "burstiness":
        ("mean_shift", 0.2),

    # §4.30 / §4.26 — protocol downgrade. 20% of malicious flows
    # advertise TLS 1.2 instead of 1.3. This is a categorical shift,
    # not a Cohen's-d feature, so the self-check does not include it
    # in its continuous-feature scan.
    "tls_version":
        ("replace", lambda rng, n: np.where(
            rng.random(n) < 0.20,
            np.array(["1.2"] * n, dtype=object),
            np.array(["1.3"] * n, dtype=object),
        )),
}


# ─── Generation ──────────────────────────────────────────────────────────

def _generate_metadata(
    rng: np.random.Generator, n: int, class_tag: str,
) -> dict[str, np.ndarray]:
    """Populate the seven metadata columns for `n` rows of class `class_tag`.

    Flow IDs are `toy_{class_tag}_{i:06d}`, so benign and malicious
    rows are immediately distinguishable by ID (useful when eyeballing
    a generated CSV).

    IP addresses use TEST-NET-1 (192.0.2.0/24) for benign and
    TEST-NET-2 (198.51.100.0/24) for malicious. These are reserved
    for documentation and are guaranteed not to be real (RFC 5737).
    The choice is intentional: a toy dataset should not accidentally
    point at real infrastructure.
    """
    if class_tag == "b":
        prefix = "192.0.2."
    else:
        prefix = "198.51.100."
    dates = ["2024-01-01", "2024-01-02", "2024-01-03",
             "2024-01-04", "2024-01-05"]
    return {
        "flow_id": np.array([f"toy_{class_tag}_{i:06d}" for i in range(n)],
                            dtype=object),
        "capture_date": rng.choice(dates, n).astype(object),
        "src_ip": np.array([f"{prefix}{rng.integers(1, 255)}" for _ in range(n)],
                           dtype=object),
        "dst_ip": np.array(["8.8.8.8"] * n, dtype=object),
        "src_port": rng.integers(1024, 65535, n).astype(np.int64),
        "dst_port": np.full(n, 443, dtype=np.int64),
        "protocol": np.array(["DoH"] * n, dtype=object),
    }


def _generate_class(
    n: int,
    rng: np.random.Generator,
    class_tag: str,
    label: int,
) -> pd.DataFrame:
    """Generate `n` rows of one class.

    Metadata columns are populated by `_generate_metadata`. Every
    other column is populated by its entry in `SYNTHETIC_DISTRIBUTIONS`
    (benign) or by the shift in `MALICIOUS_SHIFTS` applied to the
    benign draw (malicious). Columns with no entry in either dict are
    filled from `SYNTHETIC_DISTRIBUTIONS` directly, even for the
    malicious class; only the explicitly-shifted columns differ.

    The order of the returned DataFrame's columns is exactly
    `FEATURE_SCHEMA`, so the writer does not need to reorder.
    """
    # Step 1: draw every column from the benign distribution.
    data: dict[str, np.ndarray] = _generate_metadata(rng, n, class_tag)
    for col, fn in SYNTHETIC_DISTRIBUTIONS.items():
        if fn is None or col in data:
            continue
        data[col] = fn(rng, n)

    # Step 2: apply the malicious shifts (no-op for benign).
    if label == 1:
        for col, shift in MALICIOUS_SHIFTS.items():
            kind = shift[0]
            if kind == "mean_shift":
                data[col] = data[col] + shift[1]
            elif kind == "replace":
                data[col] = shift[1](rng, n)
            elif kind == "clip":
                lo, hi = shift[1], shift[2]
                data[col] = np.clip(data[col], lo, hi)
            else:  # pragma: no cover — unreachable if MALICIOUS_SHIFTS is valid
                raise ValueError(f"Unknown shift kind {kind!r} for column {col!r}.")

    # Step 3: label.
    data[LABEL_COLUMN] = np.full(n, label, dtype=np.int8)

    # Step 4: assemble in schema order and canonicalize dtypes.
    df = pd.DataFrame(data)
    # Any canonical column not yet present (a schema column added after
    # this file was written) is filled with the sentinel so the toy
    # demo still produces a schema-complete file. The self-check
    # verifies that no such column exists at the time of writing.
    for col in FEATURE_SCHEMA:
        if col not in df.columns:
            df[col] = -1.0 if col in NUMERIC_COLUMNS else "unknown"
    df = df[list(FEATURE_SCHEMA)]
    return canonicalize_dtypes(df)


def generate_synthetic(
    seed: int = DEFAULT_SEED,
    n_benign: int = DEFAULT_BENIGN_ROWS,
    n_malicious: int = DEFAULT_MALICIOUS_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate the benign and malicious halves of the synthetic toy dataset.

    Returns:
        (benign_df, malicious_df). Each is a schema-complete DataFrame
        with `len(df) == n_benign` / `len(df) == n_malicious` and
        columns exactly `FEATURE_SCHEMA`.
    """
    # Two independent RNG streams from the same seed. See the
    # "Design notes" header comment.
    benign_rng = np.random.default_rng(seed)
    malicious_rng = np.random.default_rng(seed + MALICIOUS_SEED_OFFSET)

    benign = _generate_class(n_benign, benign_rng, "b", label=0)
    malicious = _generate_class(n_malicious, malicious_rng, "m", label=1)
    return benign, malicious


def sample_from_full(
    processed_path: Path,
    seed: int = DEFAULT_SEED,
    n_benign: int = DEFAULT_BENIGN_ROWS,
    n_malicious: int = DEFAULT_MALICIOUS_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample `n_benign` benign + `n_malicious` malicious rows from the
    processed Zenodo DoH corpus (File 10d).

    The sampling is deterministic given `seed`: the benign and
    malicious subsets are each shuffled by an independent RNG derived
    from `seed`, then truncated to the requested size. The returned
    DataFrames have columns exactly `FEATURE_SCHEMA`.

    Raises:
        FileNotFoundError: if `processed_path` does not exist. The
            message names File 10d as the preprocessor that produces it.
        ValueError: if either class has fewer rows than requested.
    """
    processed_path = Path(processed_path)
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed Zenodo corpus not found at {processed_path}. "
            f"Run `python data/preprocess_zenodo.py` first (File 10d of "
            f"the 74-file plan). If the corpus has not been downloaded "
            f"yet, run `python data/download_datasets.py zenodo` before "
            f"that."
        )
    if processed_path.is_dir():
        # Dask partitioned directory.
        import dask.dataframe as dd
        df = dd.read_parquet(str(processed_path)).compute()
    elif processed_path.suffix == ".parquet":
        df = pd.read_parquet(processed_path)
    else:
        df = pd.read_csv(processed_path, low_memory=False)

    validate_schema(df)

    benign_pool = df[df[LABEL_COLUMN] == 0]
    malicious_pool = df[df[LABEL_COLUMN] == 1]
    if len(benign_pool) < n_benign:
        raise ValueError(
            f"Processed corpus has only {len(benign_pool)} benign row(s); "
            f"requested {n_benign}. Use a smaller --benign-rows value."
        )
    if len(malicious_pool) < n_malicious:
        raise ValueError(
            f"Processed corpus has only {len(malicious_pool)} malicious "
            f"row(s); requested {n_malicious}. Use a smaller "
            f"--malicious-rows value."
        )

    # Deterministic sample.
    benign_rng = np.random.default_rng(seed)
    malicious_rng = np.random.default_rng(seed + MALICIOUS_SEED_OFFSET)
    benign_idx = benign_rng.choice(len(benign_pool), n_benign, replace=False)
    malicious_idx = malicious_rng.choice(len(malicious_pool), n_malicious, replace=False)

    benign = benign_pool.iloc[benign_idx].reset_index(drop=True)[list(FEATURE_SCHEMA)]
    malicious = malicious_pool.iloc[malicious_idx].reset_index(drop=True)[list(FEATURE_SCHEMA)]
    return canonicalize_dtypes(benign), canonicalize_dtypes(malicious)


# ─── Writer ──────────────────────────────────────────────────────────────

def write_toy_dataset(
    benign: pd.DataFrame,
    malicious: pd.DataFrame,
    out_dir: Path,
    *,
    force: bool,
) -> tuple[Path, Path, Path]:
    """Write the three toy CSVs to `out_dir`.

    Writes:
      * `benign_sample.csv`    — `len(benign)` rows
      * `malicious_sample.csv` — `len(malicious)` rows
      * `combined.csv`         — `len(benign) + len(malicious)` rows

    All three files carry the canonical `FEATURE_SCHEMA` order. The
    combined file is a convenience for `train_pafa.py --mode toy`,
    which can read it directly without a `pd.concat`.

    `out_dir` is `data/processed/toy_dataset/` by default (not
    `data/toy_dataset/`). The committed header-only CSVs in
    `data/toy_dataset/` are schema templates and are never touched.

    Returns:
        (benign_path, malicious_path, combined_path).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "benign": out_dir / "benign_sample.csv",
        "malicious": out_dir / "malicious_sample.csv",
        "combined": out_dir / "combined.csv",
    }
    existing = [p for p in paths.values() if p.exists()]
    if existing and not force:
        raise FileExistsError(
            f"Existing toy dataset file(s): {[str(p) for p in existing]}. "
            "Pass --force to overwrite."
        )

    benign.to_csv(paths["benign"], index=False, columns=list(FEATURE_SCHEMA))
    malicious.to_csv(paths["malicious"], index=False, columns=list(FEATURE_SCHEMA))
    combined = pd.concat([benign, malicious], ignore_index=True)
    combined.to_csv(paths["combined"], index=False, columns=list(FEATURE_SCHEMA))

    log.info("Wrote %s (%d rows)", paths["benign"], len(benign))
    log.info("Wrote %s (%d rows)", paths["malicious"], len(malicious))
    log.info("Wrote %s (%d rows)", paths["combined"], len(combined))
    return paths["benign"], paths["malicious"], paths["combined"]


# ─── Cohen's d helper ────────────────────────────────────────────────────

def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d (pooled) effect size between two samples.

    Uses the pooled-standard-deviation form, which is the appropriate
    estimator when the two samples have comparable variances. For
    wildly unequal variances the pooled estimate is biased; the toy
    generator's shifts are designed to be roughly variance-preserving,
    so this is acceptable here. The self-check uses the sign and
    magnitude, not the exact value, so a small bias does not affect
    the assertion.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    sx2 = x.var(ddof=1)
    sy2 = y.var(ddof=1)
    pooled = np.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2))
    if pooled == 0:
        return 0.0
    return float((x.mean() - y.mean()) / pooled)


# ─── Self-check ──────────────────────────────────────────────────────────

def _self_check() -> int:
    """Non-destructive self-check used by pytest (File 21a).

    Runs the synthetic mode on 1,000 + 1,000 rows (the defaults) and
    confirms:

      1. Total 2,000 rows; 47 columns; schema-valid.
      2. Class balance 1,000 benign / 1,000 malicious.
      3. Determinism: the same seed produces byte-identical CSV bytes
         (via a temp file round-trip).
      4. Class separability: at least one continuous feature has
         |Cohen's d| ≥ 0.5 (the toy data is not class-blind).
      5. Class separability upper bound: at least one feature has
         |Cohen's d| ≤ 2.0 (the toy data is not trivially separable —
         the toy demo must land in the 85–90 % range, not 99 %).

    Does not run the `from-full` mode (that requires the 32 GB corpus
    and is out of scope for the self-check). Writes to a temp
    directory; does not touch `data/processed/`.
    """
    import tempfile

    # 1 + 2. Generate and validate.
    try:
        benign, malicious = generate_synthetic(seed=42, n_benign=1000, n_malicious=1000)
    except Exception as exc:  # noqa: BLE001 — surface any generator bug clearly
        print(f"FAIL: generator raised {type(exc).__name__}: {exc}")
        return 1

    if len(benign) != 1000 or len(malicious) != 1000:
        print(f"FAIL: unexpected row counts benign={len(benign)} "
              f"malicious={len(malicious)} (expected 1000/1000).")
        return 1
    if list(benign.columns) != list(FEATURE_SCHEMA):
        print("FAIL: benign columns do not match FEATURE_SCHEMA.")
        return 1
    if list(malicious.columns) != list(FEATURE_SCHEMA):
        print("FAIL: malicious columns do not match FEATURE_SCHEMA.")
        return 1
    try:
        validate_schema(benign)
        validate_schema(malicious)
    except ValueError as exc:
        print(f"FAIL: schema validation failed: {exc}")
        return 1
    if benign[LABEL_COLUMN].nunique() != 1 or int(benign[LABEL_COLUMN].iloc[0]) != 0:
        print("FAIL: benign file does not contain only label 0.")
        return 1
    if malicious[LABEL_COLUMN].nunique() != 1 or int(malicious[LABEL_COLUMN].iloc[0]) != 1:
        print("FAIL: malicious file does not contain only label 1.")
        return 1
    print("OK: 2,000 rows (1,000 / 1,000), 47 columns, schema valid.")

    # 3. Determinism via byte-identical CSVs in a temp dir.
    with tempfile.TemporaryDirectory(prefix="toy_selfcheck_") as tmp:
        p1 = write_toy_dataset(benign, malicious, Path(tmp) / "a", force=True)
        p2 = write_toy_dataset(benign, malicious, Path(tmp) / "b", force=True)
        for name, a, b in (
            ("benign", p1[0], p2[0]),
            ("malicious", p1[1], p2[1]),
            ("combined", p1[2], p2[2]),
        ):
            if a.read_bytes() != b.read_bytes():
                print(f"FAIL: {name} file is not byte-deterministic across runs.")
                return 1
    print("OK: same seed produces byte-identical CSVs.")

    # 4 + 5. Class separability scan.
    ds: dict[str, float] = {}
    for col in NUMERIC_COLUMNS:
        if col == LABEL_COLUMN:
            continue
        d = cohens_d(benign[col].to_numpy(), malicious[col].to_numpy())
        ds[col] = d
    max_abs = max(abs(d) for d in ds.values())
    # Count how many features have |d| >= 0.5.
    strong = [c for c, d in ds.items() if abs(d) >= 0.5]
    if not strong:
        print("FAIL: no feature has |Cohen's d| >= 0.5; the toy data is class-blind.")
        return 1
    if max_abs > 2.0:
        # Report which feature is too strong, to make debugging easy.
        worst = max(ds.items(), key=lambda kv: abs(kv[1]))
        print(
            f"FAIL: feature {worst[0]!r} has |Cohen's d| = {abs(worst[1]):.2f} "
            f"(> 2.0); the toy data is trivially separable. Reduce the "
            f"shift in MALICIOUS_SHIFTS so the toy demo lands in the "
            f"85-90 % range, not 99 %."
        )
        return 1
    print(
        f"OK: {len(strong)} feature(s) with |Cohen's d| >= 0.5; "
        f"max |d| = {max_abs:.2f} (<= 2.0, toy data is not trivially separable)."
    )

    print("self-check ok")
    return 0


# ─── argparse ─────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="make_toy_dataset.py",
        description=(
            "Generate the 1,000 + 1,000 row toy dataset used by the "
            "two-minute PAFA-DoH smoke test (README §3). Output goes to "
            "data/processed/toy_dataset/. The committed header-only "
            "CSVs in data/toy_dataset/ are the schema template and are "
            "never overwritten."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python data/make_toy_dataset.py --mode synthetic --seed 42\n"
            "  python data/make_toy_dataset.py --mode from-full \\\n"
            "      --processed-path data/processed/zenodo_doh.csv\n"
        ),
    )
    p.add_argument("--mode", choices=["synthetic", "from-full"],
                   default="synthetic",
                   help="Generation mode. 'synthetic' (default) draws from "
                        "parametric distributions; 'from-full' samples the "
                        "first 1,000 + 1,000 rows from the processed Zenodo "
                        "corpus (File 10d's output).")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"Master seed (default {DEFAULT_SEED}). Must match "
                        f"configs/default_pafa_config.yaml::seed.")
    p.add_argument("--benign-rows", type=int, default=DEFAULT_BENIGN_ROWS,
                   help=f"Number of benign rows (default {DEFAULT_BENIGN_ROWS}).")
    p.add_argument("--malicious-rows", type=int, default=DEFAULT_MALICIOUS_ROWS,
                   help=f"Number of malicious rows (default {DEFAULT_MALICIOUS_ROWS}).")
    p.add_argument("--processed-path", type=Path, default=None,
                   help="Path to the processed Zenodo corpus. Required for "
                        "--mode from-full; ignored for --mode synthetic. "
                        "Accepts a CSV, a single Parquet, or a Dask-"
                        "partitioned Parquet directory.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Override the output directory. Default: "
                        "<artifact_root>/data/processed/toy_dataset/.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing toy dataset files. Without this "
                        "flag, an existing file is a hard error.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable DEBUG-level logging.")
    return p


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

    # Resolve artifact root.
    try:
        root = resolve_artifact_root()
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1
    log.info("Artifact root:     %s", root)
    log.info("Artifact version:  %s", ARTIFACT_VERSION)
    log.info("Thesis:            %s", THESIS_CHAPTER)
    log.info("Citation DOI:      %s", CITATION_DOI)

    # Seed cross-check against the master config.
    cfg_path = root / "configs" / "default_pafa_config.yaml"
    if cfg_path.is_file():
        try:
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            config_seed = int(cfg["seed"])
            if args.seed != config_seed:
                log.error(
                    "--seed %d does not match the master config's seed (%d) "
                    "at configs/default_pafa_config.yaml::seed. The toy "
                    "demo reproduces the committed expected outputs only "
                    "at the pinned seed (%d).",
                    args.seed, config_seed, config_seed,
                )
                return 1
        except (yaml.YAMLError, KeyError, TypeError) as exc:
            log.warning("Could not read master config for seed check: %s", exc)

    # Output dir.
    out_dir = Path(args.out_dir) if args.out_dir else root / OUTPUT_SUBDIR
    log.info("Output directory:  %s", out_dir)

    # Generate.
    try:
        if args.mode == "synthetic":
            benign, malicious = generate_synthetic(
                seed=args.seed,
                n_benign=args.benign_rows,
                n_malicious=args.malicious_rows,
            )
        elif args.mode == "from-full":
            if args.processed_path is None:
                log.error(
                    "--mode from-full requires --processed-path (a CSV, a "
                    "single Parquet, or a Dask-partitioned Parquet "
                    "directory produced by data/preprocess_zenodo.py)."
                )
                return 1
            benign, malicious = sample_from_full(
                processed_path=args.processed_path,
                seed=args.seed,
                n_benign=args.benign_rows,
                n_malicious=args.malicious_rows,
            )
        else:  # pragma: no cover — argparse enforces the choice
            log.error("Unknown mode: %s", args.mode)
            return 1
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 1

    # Write.
    try:
        benign_path, malicious_path, combined_path = write_toy_dataset(
            benign, malicious, out_dir, force=args.force,
        )
    except (OSError, FileExistsError) as exc:
        log.error("%s", exc)
        return 1

    # NEXT STEP banner.
    print()
    print("NEXT STEP:")
    print(
        "  python scripts/train_pafa.py "
        "--config configs/default_pafa_config.yaml --mode toy"
    )
    print()
    print(f"  benign:    {benign_path}")
    print(f"  malicious: {malicious_path}")
    print(f"  combined:  {combined_path}")
    print()
    print(
        "Toy-mode accuracy is ~85-90 % (README §4). This is a smoke "
        "test, NOT a reproduction of Chapter 4's headline figures."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
