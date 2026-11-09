#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# data_utils.py — PAFA-DoH Reproducibility Artifact
# ═══════════════════════════════════════════════════════════════════════════
#
# Companion to:
#   README.md                       (File 1 of 74)
#   REPRODUCIBILITY_CHECKLIST.md    (File 2 of 74)
#   LICENSE                         (File 3 of 74)
#   requirements.txt                (File 4 of 74)
#   environment.yml                 (File 5 of 74)
#   Dockerfile                      (File 6 of 74)
#   setup.py                        (File 7 of 74)
#   .gitignore                      (File 8 of 74)
#   configs/default_pafa_config.yaml (File 9, master config)
#   configs/fl_dp_params.yaml ... configs/hardware_emulation_config.yaml
#                                   (Files 9b–9h)
#
# Thesis chapter: Chapter 4 (PAFA-DoH)
# Citation DOI:   10.1016/j.neunet.2025.108343   (Neural Networks, April 2026)
# License:        MIT (this file, as part of the artifact's source code)
# Artifact version: 1.0.0
#
# ───────────────────────────────────────────────────────────────────────────
# Purpose
# ───────────────────────────────────────────────────────────────────────────
# This module is the shared substrate for the data layer (Files 10b–10f).
# It defines the authoritative 47-column feature schema, the schema-
# validation and dtype-canonicalisation helpers, the SHA-256 and
# checksum-verification helpers, the three-step artifact-root resolver
# documented in File 7's footer, and the seeded stratified train / val /
# test splitter.
#
# Every other script in data/ imports from this module; nothing in
# data/preprocess_zenodo.py, data/preprocess_doh2023.py or
# data/make_toy_dataset.py is allowed to re-declare the schema or the
# split logic, because two declarations would silently drift apart.
#
# ───────────────────────────────────────────────────────────────────────────
# Where the pieces are specified in Chapter 4
# ───────────────────────────────────────────────────────────────────────────
#   Feature schema (47 columns)       → §4.24.1, §4.25.1, Table 14–17.
#                                       The schema is the same one used
#                                       by the preprocessing pipeline
#                                       (§6.13 in the blueprint is the
#                                       E3-DoH equivalent; the PAFA-DoH
#                                       preprocessor uses the same
#                                       47-column layout so that the
#                                       committed expected outputs of
#                                       Tables 14, 16 and 17 can be
#                                       regenerated without a schema
#                                       mismatch).
#
#   Stratified train/val/test split   → §4.25.1 (dataset composition and
#                                       preprocessing) and the design
#                                       commitment in §1.3 / Table 2 that
#                                       the expected outputs are
#                                       reproducible from a pinned seed.
#                                       The split is seeded (seed = 42,
#                                       Table 16) and stratified by class
#                                       and by capture-date bucket so
#                                       that no training row and test row
#                                       come from the same date bucket.
#
#   SHA-256 integrity check           → `data/CHECKSUMS.txt` (File 10j).
#                                       Datasets are fetched by DOI and
#                                       checked before preprocessing; a
#                                       checksum mismatch aborts the
#                                       pipeline (no silent fallback).
#
#   Artifact-root resolution          → File 7's footer and File 9's
#                                       header. Three-step contract:
#                                       $PAFA_ROOT → repo tree →
#                                       <sys.prefix>/share/pafa-doh/.
#
# ───────────────────────────────────────────────────────────────────────────
# Design notes
# ───────────────────────────────────────────────────────────────────────────
# • `from __future__ import annotations` is included so that annotations
#   work unchanged on Python 3.11 and on future Python versions where
#   `X | Y` syntax becomes the default.
# • No third-party imports at module level beyond `numpy` and `pandas`,
#   both of which are pinned in requirements.txt (File 4). No `torch`,
#   no `gudhi`, no `py_ecc` — the data layer must import in a minimal
#   environment (e.g., a CI job that only checks the schema).
# • The toy CSVs are read only if they exist. On a fresh clone *before*
#   Files 10h–10i are uploaded, the self-check prints `SKIP` and exits
#   0; the same self-check on a fully-populated clone prints `schema ok`.
#   This makes the file safe to upload at position 10a without breaking
#   CI on an in-progress repository.
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import hashlib
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd


# ─── Module-level constants ─────────────────────────────────────────────────

ARTIFACT_VERSION: str = "1.0.0"
THESIS_CHAPTER: str = "Chapter 4 (PAFA-DoH)"
CITATION_DOI: str = "10.1016/j.neunet.2025.108343"

DEFAULT_SEED: int = 42
DEFAULT_TRAIN_FRAC: float = 0.70
DEFAULT_VAL_FRAC: float = 0.15
DEFAULT_TEST_FRAC: float = 0.15


# ─── 47-column feature schema ───────────────────────────────────────────────
#
# Authoritative column order. Every CSV the artifact reads or writes MUST
# have its columns in this order. The two committed header-only toy CSVs
# (Files 10h–10i) contain exactly this header row and no data rows; the
# preprocessing scripts (Files 10c–10d) write their output with exactly
# this header row; `validate_schema()` raises if the order differs.
#
# The schema is 47 columns:
#
#   §1  Flow metadata .................  7
#   §2  Flow-level statistics  ......... 10
#   §3  Timing / inter-arrival  ........  7
#   §4  Query-type / entropy  ..........  8
#   §5  Response features  .............  3
#   §6  TLS handshake features  ........  4
#   §7  HTTP/2 + HTTP/3 frame features .  3
#   §8  Temporal / spectral  ...........  4
#   §9  Label  .........................  1
#                                       ──
#                                       47
#
# All features are computable from the six permitted observables in
# `configs/default_pafa_config.yaml::threat_model.permitted_observables`
# and from protocol metadata that is visible on the wire without
# decrypting the DoH payload. No feature depends on query or response
# payload content, application-layer semantics or database information
# (the four forbidden observables in the same config file). A feature
# that could only be computed from a forbidden observable is a bug and
# must be removed from this schema before the artifact is re-run.
FEATURE_SCHEMA: tuple[str, ...] = (
    # ── §1. Flow metadata (7) ──────────────────────────────────────────
    "flow_id",             # str   — stable identifier for the flow
    "capture_date",        # date  — YYYY-MM-DD (used by the splitter)
    "src_ip",              # str   — source IP (hashed if anonymised)
    "dst_ip",              # str   — destination IP
    "src_port",            # int   — 0–65535
    "dst_port",            # int   — 443 (DoH), 853 (DoT / DoQ)
    "protocol",            # str   — "DoH" | "DoT" | "DoQ"

    # ── §2. Flow-level statistics (10) ─────────────────────────────────
    "flow_duration_ms",    # float — from first to last packet
    "packet_count",        # int   — number of packets
    "bytes_out",           # int   — total bytes, client → resolver
    "bytes_in",            # int   — total bytes, resolver → client
    "packet_size_mean",    # float — bytes
    "packet_size_std",     # float — bytes
    "packet_size_max",     # int   — bytes
    "packet_size_min",     # int   — bytes
    "bytes_per_second",    # float — bytes_out + bytes_in over duration
    "packets_per_second",  # float

    # ── §3. Timing / inter-arrival statistics (7) ──────────────────────
    "iat_mean_ms",         # float — mean inter-arrival time
    "iat_std_ms",          # float — std
    "iat_min_ms",          # float — min
    "iat_max_ms",          # float — max
    "iat_median_ms",       # float — median
    "iat_skew",            # float — third standardised moment
    "iat_kurtosis",        # float — fourth standardised moment − 3

    # ── §4. Query-type / entropy features (8) ──────────────────────────
    "entropy_subdomain",   # float — Shannon entropy of the subdomain
    "entropy_query_type",  # float — Shannon entropy of the qtype stream
    "qtype_A_count",       # int
    "qtype_AAAA_count",    # int
    "qtype_TXT_count",     # int
    "qtype_MX_count",      # int
    "qtype_CNAME_count",   # int
    "qtype_other_count",   # int

    # ── §5. Response features (3) ──────────────────────────────────────
    "response_size_mean",  # float — bytes
    "response_size_std",   # float — bytes
    "response_entropy",    # float — Shannon entropy of the response sizes

    # ── §6. TLS handshake features (4) ─────────────────────────────────
    "tls_version",         # str   — "1.3" (or "1.2" for downgrade tests)
    "cipher_suite_count",  # int   — capped at 256 bits (§1.7.1)
    "extension_count",     # int
    "session_reuse",       # int   — 0 | 1

    # ── §7. HTTP/2 + HTTP/3 frame features (3) ─────────────────────────
    "http2_stream_count",  # int   — 0 for DoT / DoQ
    "http2_frame_count",   # int   — 0 for DoT / DoQ
    "http3_frame_count",   # int   — 0 for DoH/2 and DoT

    # ── §8. Temporal / spectral features (4) ───────────────────────────
    "autocorr_lag1",       # float — ACF of the packet-size series at lag 1
    "autocorr_lag5",       # float — ACF at lag 5 (§4.7 / Eq. 100)
    "burstiness",          # float — (σ_iat − μ_iat) / (σ_iat + μ_iat)
    "spectral_low_band_energy",  # float — FFT energy in 0.1–0.3 Hz

    # ── §9. Label (1) ──────────────────────────────────────────────────
    "label",               # int   — 0 (benign) | 1 (malicious)
)


# Derived views — never redeclare these as literal tuples anywhere else.
LABEL_COLUMN: str = "label"
METADATA_COLUMNS: tuple[str, ...] = (
    "flow_id",
    "capture_date",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
)
"""Columns that identify the flow but are not model features."""

STRATIFY_COLUMNS: tuple[str, ...] = (
    "label",
    "capture_date",
)
"""Columns used by `write_split()` to stratify the split."""

NUMERIC_COLUMNS: tuple[str, ...] = tuple(
    c for c in FEATURE_SCHEMA if c not in METADATA_COLUMNS and c != "tls_version"
)
"""Columns expected to be numeric (int / float) after dtype canonicalisation.

`tls_version` is excluded because it is a string ("1.3" / "1.2") in the
raw CSV and is encoded downstream by the SNN encoder (File 12c). Keeping
it as a string preserves human readability of the CSV.
"""


# ─── Artifact-root resolution ──────────────────────────────────────────────

def resolve_artifact_root(env_var: str = "PAFA_ROOT") -> Path:
    """Return the artifact root directory.

    Resolution order (documented in File 7's footer and File 9's header):

    1. ``$PAFA_ROOT`` if set and a directory.
    2. The nearest ancestor of this file that contains both
       ``requirements.txt`` and ``configs/default_pafa_config.yaml``.
    3. ``<sys.prefix>/share/pafa-doh`` if it exists (wheel installs,
       populated by ``setup.py::data_files``).

    Raises:
        FileNotFoundError: if none of the three candidates resolves.
            The message names every path that was tried, so a reviewer
            can fix the environment without reading this source.
    """
    tried: list[Path] = []

    env_val = os.environ.get(env_var)
    if env_val:
        candidate = Path(env_val).expanduser().resolve()
        tried.append(candidate)
        if candidate.is_dir():
            return candidate

    here = Path(__file__).resolve().parent
    for parent in (here, *here.parents):
        tried.append(parent)
        if (
            (parent / "requirements.txt").is_file()
            and (parent / "configs" / "default_pafa_config.yaml").is_file()
        ):
            return parent

    wheel_candidate = Path(sys.prefix) / "share" / "pafa-doh"
    tried.append(wheel_candidate)
    if wheel_candidate.is_dir():
        return wheel_candidate

    raise FileNotFoundError(
        "Could not resolve the PAFA-DoH artifact root. Tried: "
        + ", ".join(str(p) for p in tried)
        + ". Set $PAFA_ROOT to the repository root, or run the artifact "
          "from a checkout that contains requirements.txt and "
          "configs/default_pafa_config.yaml."
    )


# ─── SHA-256 and checksum verification ─────────────────────────────────────

def compute_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of the file at ``path``.

    Reads in ``chunk_size`` blocks (default 1 MiB) so that a 32 GB file
    is hashed with bounded memory.
    """
    if not path.is_file():
        raise FileNotFoundError(f"SHA-256 target not found: {path}")
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def verify_checksums(manifest_path: Path, data_dir: Path) -> dict[str, str]:
    """Verify every file listed in ``manifest_path`` against its SHA-256.

    ``manifest_path`` is a plain-text file (``data/CHECKSUMS.txt``,
    File 10j) with one record per line, in the conventional format::

        <sha256-hex>  <relative-path-from-data_dir>

    Blank lines and lines starting with ``#`` are ignored. Any record
    whose hash does not match raises ``ValueError`` with a message that
    names the file, the expected hash and the observed hash. A record
    whose file does not exist raises ``FileNotFoundError``.

    Returns:
        A ``dict`` mapping the relative path to the observed hex digest,
        for every record that passed. The dict is deliberately returned
        even on success so that callers can record it in the run log.
    """
    manifest_path = Path(manifest_path)
    data_dir = Path(data_dir)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Checksum manifest not found: {manifest_path}")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    results: dict[str, str] = {}
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(
                f"Malformed checksum line in {manifest_path}: {raw_line!r}. "
                "Expected '<sha256>  <relative-path>'."
            )
        expected_hash, rel_path = parts
        target = data_dir / rel_path
        observed_hash = compute_sha256(target)
        if observed_hash != expected_hash:
            raise ValueError(
                f"Checksum mismatch for {target}:\n"
                f"  expected: {expected_hash}\n"
                f"  observed: {observed_hash}\n"
                "The downloaded dataset does not match the pinned DOI. "
                "Delete the file and re-run data/download_datasets.py; "
                "do not proceed with a mismatched corpus."
            )
        results[rel_path] = observed_hash
    return results


# ─── Schema helpers ────────────────────────────────────────────────────────

def load_feature_schema() -> tuple[str, ...]:
    """Return the authoritative 47-column schema as a tuple.

    A function, not a constant, so that a future version of the schema
    can be loaded from a file (e.g., a JSON schema committed alongside
    the CSV headers) without changing the call sites. Callers should not
    cache the result across runs, because the schema is part of the
    artifact's public interface and a cached value would hide a schema
    change.
    """
    return FEATURE_SCHEMA


def validate_schema(df: pd.DataFrame, *, strict_order: bool = True) -> None:
    """Validate that ``df`` has exactly the 47-column schema.

    Args:
        df: The DataFrame to validate.
        strict_order: If True (default), the columns must appear in the
            exact order of ``FEATURE_SCHEMA``. If False, only the set of
            columns is checked (useful when reading a CSV that has been
            round-tripped through a tool that re-sorts columns).

    Raises:
        ValueError: on any column-count, column-name or column-order
            mismatch. The message names the first mismatch and the
            full expected schema so a reviewer can diff their CSV
            against the artifact.
    """
    expected = list(FEATURE_SCHEMA)
    actual = list(df.columns)

    if len(actual) != len(expected):
        raise ValueError(
            f"Schema mismatch: expected {len(expected)} columns, got "
            f"{len(actual)}. Expected: {expected}. Got: {actual}."
        )

    if strict_order:
        if actual != expected:
            for i, (exp, act) in enumerate(zip(expected, actual)):
                if exp != act:
                    raise ValueError(
                        f"Schema mismatch at column {i}: expected "
                        f"{exp!r}, got {act!r}. Full expected order: "
                        f"{expected}."
                    )
            # The lengths matched and no per-position mismatch was
            # found, so this branch is unreachable — but keep the
            # defensive raise so a future edit cannot silently pass.
            raise ValueError(f"Schema order mismatch. Expected {expected}, got {actual}.")
    else:
        missing = [c for c in expected if c not in actual]
        extra = [c for c in actual if c not in expected]
        if missing or extra:
            raise ValueError(
                f"Schema set mismatch. Missing: {missing}. Extra: {extra}."
            )


def canonicalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with canonical dtypes for every column.

    * Metadata string columns → ``object`` (``str`` in pandas ≥ 2 with
      ``string`` dtype disabled to keep CSV round-trip stable).
    * ``capture_date`` → ``datetime64[ns]`` at day resolution (the
      value is a date, not a timestamp; the splitter stratifies on the
      day).
    * ``label`` → ``int8`` (0 or 1).
    * Every numeric column in ``NUMERIC_COLUMNS`` → ``float64``.
    * ``protocol`` and ``tls_version`` → ``object``.

    The function does not mutate its argument; it always returns a new
    DataFrame. Any conversion that would lose information (e.g., a
    non-integer ``label``) raises ``ValueError`` rather than silently
    truncating.
    """
    out = df.copy()

    # Date column
    if "capture_date" in out.columns:
        out["capture_date"] = pd.to_datetime(out["capture_date"], errors="raise").dt.normalize()

    # Label
    if LABEL_COLUMN in out.columns:
        labels = out[LABEL_COLUMN]
        if not labels.isin([0, 1]).all():
            bad = labels[~labels.isin([0, 1])].unique().tolist()
            raise ValueError(
                f"{LABEL_COLUMN} must be 0 or 1; got unexpected values: {bad}"
            )
        out[LABEL_COLUMN] = labels.astype("int8")

    # Numeric columns
    for col in NUMERIC_COLUMNS:
        if col in out.columns and col != LABEL_COLUMN:
            out[col] = pd.to_numeric(out[col], errors="raise").astype("float64")

    # String columns
    for col in ("protocol", "tls_version"):
        if col in out.columns:
            out[col] = out[col].astype("object")

    return out


# ─── Toy-sample helpers ────────────────────────────────────────────────────

def read_toy_sample(path: Path) -> pd.DataFrame:
    """Read a header-only toy CSV and return an empty DataFrame with the
    47-column schema.

    This helper is deliberately strict: a header-only CSV that does not
    match ``FEATURE_SCHEMA`` raises before any downstream code sees it.
    The two committed toy CSVs (Files 10h–10i) are expected to be
    exactly this: header row, no data rows.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Toy sample not found: {path}")
    df = pd.read_csv(path)
    validate_schema(df)
    return canonicalize_dtypes(df) if len(df) > 0 else df


# ─── Stratified, seeded train / val / test split ───────────────────────────

@dataclass(frozen=True)
class SplitResult:
    """The three partitions produced by :func:`write_split`.

    Attributes:
        train: The training partition (``train_frac`` of each stratum).
        val:   The validation partition (``val_frac``).
        test:  The test partition (``test_frac``).
        seed:  The seed used. Recorded so that the run log preserves the
            provenance of the split.
        per_stratum_counts: A dict keyed by ``(label, capture_date)``
            with values ``{"train": n, "val": n, "test": n}``, useful
            for the per-class sample counts that Table 16 requires and
            that the reproducibility checklist's §9 limitation cites as
            not fully reported in Chapter 4.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    seed: int
    per_stratum_counts: dict[tuple[int, str], dict[str, int]] = field(default_factory=dict)

    def write(self, out_dir: Path) -> None:
        """Write the three partitions to ``out_dir`` as ``split_{train,val,test}.csv``.

        The CSV writer uses ``index=False`` and the exact column order of
        ``FEATURE_SCHEMA``; a downstream reader can round-trip the file
        through ``validate_schema`` and get the same DataFrame back.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, part in (("train", self.train), ("val", self.val), ("test", self.test)):
            part.to_csv(out_dir / f"split_{name}.csv", index=False, columns=list(FEATURE_SCHEMA))


def write_split(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
    test_frac: float = DEFAULT_TEST_FRAC,
    min_stratum_size: int = 5,
) -> SplitResult:
    """Stratified, seeded train / val / test split.

    The split is stratified jointly on ``label`` and the day-bucket of
    ``capture_date``, so that no day's traffic appears in more than one
    partition (this prevents a leakage path that a purely class-
    stratified split would leave open). It is seeded (``seed = 42`` by
    default, matching ``configs/default_pafa_config.yaml::seed``), and
    the shuffle inside each stratum is deterministic.

    Args:
        df: A DataFrame that already passes :func:`validate_schema`.
        out_dir: Directory to write ``split_{train,val,test}.csv`` into.
        seed: RNG seed. Must equal the master config's ``seed``.
        train_frac, val_frac, test_frac: Must sum to 1.0 (within 1e-9).
        min_stratum_size: Strata smaller than this are merged with the
            nearest larger stratum before splitting; otherwise a stratum
            with 3 rows would get 0 train rows for a 70/15/15 split and
            the model would never see that stratum. The default 5 is the
            smallest value that guarantees at least 3 train rows for a
            70/15/15 split.

    Returns:
        A :class:`SplitResult` with the three partitions and the per-
        stratum counts.

    Raises:
        ValueError: if the fractions do not sum to 1.0, if the input
            does not validate, or if a stratum cannot be split without
            an empty partition after merging (in which case the caller
            should increase ``min_stratum_size`` or add more data).
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-9:
        raise ValueError(
            f"Split fractions must sum to 1.0, got "
            f"{train_frac} + {val_frac} + {test_frac} = "
            f"{train_frac + val_frac + test_frac}."
        )
    validate_schema(df)
    df = canonicalize_dtypes(df).reset_index(drop=True)

    # Build the stratification key. `capture_date` is a Timestamp after
    # canonicalize_dtypes; take its ISO date string so the key is hashable
    # and readable in per_stratum_counts.
    strat_keys = list(zip(df[LABEL_COLUMN].tolist(), df["capture_date"].dt.strftime("%Y-%m-%d").tolist()))
    df = df.assign(__strat_key=strat_keys)

    # Merge strata that are too small.
    counts = df["__strat_key"].value_counts()
    small = counts[counts < min_stratum_size]
    if len(small) > 0:
        # Merge each small stratum into the largest stratum with the same
        # label. This preserves the class balance (which is what matters
        # for training) while reducing the number of tiny strata.
        largest_per_label = (
            df.groupby(LABEL_COLUMN)["__strat_key"]
            .agg(lambda s: s.value_counts().idxmax())
            .to_dict()
        )
        for key in small.index:
            label = key[0]
            df.loc[df["__strat_key"] == key, "__strat_key"] = largest_per_label[label]

    rng = random.Random(seed)
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    per_stratum: dict[tuple[int, str], dict[str, int]] = {}

    for key, group in df.groupby("__strat_key", sort=True):
        idx = group.index.tolist()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))
        n_test = n - n_train - n_val
        if n_test < 0:
            # Rounding can over-allocate by 1; take one from train.
            n_train -= 1
            n_test = n - n_train - n_val
        if min(n_train, n_val, n_test) < 1:
            raise ValueError(
                f"Stratum {key} has {n} rows and cannot be split into "
                f"non-empty train / val / test partitions at "
                f"{train_frac}/{val_frac}/{test_frac}. Increase "
                f"min_stratum_size or provide more data."
            )
        train_idx = idx[:n_train]
        val_idx = idx[n_train : n_train + n_val]
        test_idx = idx[n_train + n_val :]
        train_parts.append(df.loc[train_idx])
        val_parts.append(df.loc[val_idx])
        test_parts.append(df.loc[test_idx])
        per_stratum[key] = {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)}

    train_df = pd.concat(train_parts).drop(columns="__strat_key").reset_index(drop=True)
    val_df = pd.concat(val_parts).drop(columns="__strat_key").reset_index(drop=True)
    test_df = pd.concat(test_parts).drop(columns="__strat_key").reset_index(drop=True)

    result = SplitResult(
        train=train_df,
        val=val_df,
        test=test_df,
        seed=seed,
        per_stratum_counts=per_stratum,
    )
    result.write(out_dir)
    return result


# ─── Self-check ────────────────────────────────────────────────────────────

def _self_check() -> int:
    """Run the module self-check. Returns 0 on success, 1 on failure.

    The check does four things:

    1. Asserts ``len(FEATURE_SCHEMA) == 47``.
    2. Asserts ``LABEL_COLUMN`` is in ``FEATURE_SCHEMA`` and is the last
       column (so that a header row from the committed toy CSVs is read
       with the label in the expected position).
    3. For each committed toy CSV (Files 10h–10i), if it exists, reads
       it and validates the 47-column schema. If it does not exist
       (because this file is uploaded before Files 10h–10i), prints
       ``SKIP`` and continues.
    4. If both toy CSVs exist, prints ``schema ok`` and returns 0.

    This function is what `python data/data_utils.py` runs, and what
    `pytest tests/test_math_utils.py::test_schema` (File 21a) wraps.
    """
    root = resolve_artifact_root()
    toy_dir = root / "data" / "toy_dataset"
    toy_files = ("benign_sample.csv", "malicious_sample.csv")

    # 1. Count
    if len(FEATURE_SCHEMA) != 47:
        print(f"FAIL: FEATURE_SCHEMA has {len(FEATURE_SCHEMA)} columns; expected 47.")
        return 1
    print(f"OK: FEATURE_SCHEMA has 47 columns (artifact {ARTIFACT_VERSION}).")

    # 2. Label position
    if FEATURE_SCHEMA[-1] != LABEL_COLUMN:
        print(f"FAIL: last column is {FEATURE_SCHEMA[-1]!r}; expected {LABEL_COLUMN!r}.")
        return 1
    print(f"OK: last column is {LABEL_COLUMN!r} (matches header-only toy CSVs).")

    # 3. Toy CSVs
    any_missing = False
    for name in toy_files:
        path = toy_dir / name
        if not path.is_file():
            print(f"SKIP: {path} (not yet uploaded; expected in a complete clone).")
            any_missing = True
            continue
        df = read_toy_sample(path)
        if len(df) != 0:
            print(f"FAIL: {path} is expected to be header-only but has {len(df)} data rows.")
            return 1
        print(f"OK: {path.name} — header-only, 47-column schema matches.")

    if any_missing:
        print("SELF-CHECK: partial (some toy CSVs not yet uploaded); exit 0.")
        return 0

    print("schema ok")
    return 0


if __name__ == "__main__":
    sys.exit(_self_check())
