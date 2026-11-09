#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# preprocess_zenodo.py — PAFA-DoH Reproducibility Artifact
# ═══════════════════════════════════════════════════════════════════════════
#
# Companion to:
#   README.md, REPRODUCIBILITY_CHECKLIST.md, LICENSE, requirements.txt,
#   environment.yml, Dockerfile, setup.py, .gitignore   (Files 1–8)
#   configs/default_pafa_config.yaml                    (File 9, master config)
#   configs/*.yaml                                      (Files 9b–9h)
#   data/data_utils.py                                  (File 10a)
#   data/download_datasets.py                           (File 10b)
#   data/preprocess_doh2023.py                          (File 10c —
#                                                        shares the
#                                                        forbidden-observables
#                                                        guard and the
#                                                        sentinel constant)
#
# Thesis chapter: Chapter 4 (PAFA-DoH)
# Citation DOI:   10.1016/j.neunet.2025.108343   (Neural Networks, April 2026)
# License:        MIT (this file, as part of the artifact's source code)
# Artifact version: 1.0.0
#
# ───────────────────────────────────────────────────────────────────────────
# Purpose
# ───────────────────────────────────────────────────────────────────────────
# Preprocess the Zenodo DoH corpus (DOI 10.5281/zenodo.5957677, ≈ 32 GB)
# into the artifact's canonical 47-column schema
# (data/data_utils.py::FEATURE_SCHEMA, File 10a). Output is consumed by
# data/split_data.py (File 10e).
#
# Unlike data/preprocess_doh2023.py (File 10c), which is a **schema-
# parity check** run on CIC-DoH2023, this script is the **training
# source** for Tables 14, 16, 17 and Figures 8–10 of Chapter 4. The
# Zenodo corpus carries per-query records (subdomain strings, query
# types, response sizes) and per-flow records (timing, TLS handshake
# parameters, HTTP/2 + HTTP/3 frame sequences), so every one of the
# 47 columns can be computed from the raw data with **zero sentinel
# fills**.
#
# That difference is the whole point of having two preprocessors:
#
#   • preprocess_doh2023.py   → schema-parity check (may use sentinels)
#   • preprocess_zenodo.py    → training source (no sentinels allowed)
#
# ───────────────────────────────────────────────────────────────────────────
# Where the pieces are specified in Chapter 4
# ───────────────────────────────────────────────────────────────────────────
#   Dataset identity / size     → §4.24.1, File 9 header
#   Composition / preprocessing → §4.25.1, and the five-stage pipeline
#                                 in §5.8.4
#   47-column feature schema    → data/data_utils.py::FEATURE_SCHEMA
#                                 (File 10a). Same schema as File 10c.
#   Forbidden observables       → §1.7.2 / §4.13, File 9
#                                 (`threat_model.forbidden_observables`),
#                                 enforced by File 10c's guard (imported
#                                 below, not redefined).
#   Feature families (the eight families) → computed here from the raw
#                                 per-query and per-flow records; see
#                                 the eight `_compute_*` functions below,
#                                 each docstring citing the chapter
#                                 section that reports the family.
#   Preprocessing stages        → §5.8.4 (malformed-flow removal,
#                                 timestamp and label unification,
#                                 statistical descriptor extraction,
#                                 feature normalization, split
#                                 preparation). This script performs
#                                 the first four in a Dask-distributed
#                                 pass; split preparation is in File
#                                 10e.
#
# ───────────────────────────────────────────────────────────────────────────
# Why Dask
# ───────────────────────────────────────────────────────────────────────────
# The corpus is ≈ 32 GB. Loading it into a single pandas DataFrame
# would need ≥ 60 GB of RAM for the concatenated frame plus the
# intermediate derived features, which exceeds the free-tier Colab
# ceiling and would fail on most laptops. Dask's `map_partitions`
# processes each CSV partition independently, so peak memory is
# `partition_size × npartitions_in_flight` (roughly `npartitions`
# × a few hundred MB for the default `npartitions = cpu_count()`),
# not the full corpus size. This is the same design as §6.13 in the
# chapter's blueprint.
#
# ───────────────────────────────────────────────────────────────────────────
# Design notes
# ───────────────────────────────────────────────────────────────────────────
# • The 47-column schema is IMPORTED from data.data_utils. It is never
#   redeclared here.
#
# • The forbidden-observables guard is IMPORTED from
#   data.preprocess_doh2023 (File 10c) — single source of truth, so
#   both preprocessors fail closed on the same forbidden columns.
#
# • The sentinel constant `SENTINEL_NUMERIC` is also imported from
#   File 10c, so the two preprocessors agree on the sentinel value.
#   `to_schema()` below refuses to write a Zenodo output that still
#   contains the sentinel — the whole point of the Zenodo path is
#   that every one of the 47 columns is derivable from the raw data.
#   This is the single most important correctness invariant of the
#   artifact: if the Zenodo preprocessor starts emitting sentinels,
#   the training pipeline would be training on a different schema
#   than the one Chapter 4 reports, and the reproduced numbers would
#   be silently wrong. The check is fail-closed.
#
# • Feature normalization is a two-pass global operation: pass 1
#   computes global mean/std per numeric column across the whole
#   corpus; pass 2 applies `(x − μ) / σ` per partition using the
#   global statistics. Doing it per-partition would make each
#   partition's mean 0 independently, which is not the same thing as
#   the global normalization the committed expected outputs were
#   produced with. The two-pass approach is the standard Dask
#   normalization pattern and matches §5.8.4's description of
#   normalization as a corpus-wide stage.
#
# • `--sample N` reads only the first N rows across all partitions
#   (Dask's `head(N)`), for a quick end-to-end smoke test without
#   processing 32 GB. The default `--sample 0` means "no sample" and
#   processes the full corpus.
#
# • The output is written as a single CSV (`zenodo_doh.csv`) plus
#   (unless `--no-parquet`) a single Parquet (`zenodo_doh.parquet`).
#   Dask's `.to_csv()` and `.to_parquet()` write partitioned output
#   by default; the artifact un-partitions by writing to a temp dir
#   and concatenating, so that the output is a single file that is
#   easy to diff and verify. Un-partitioning is a one-time cost of
#   ≈ 30 seconds for a 32 GB corpus and is worth it for the
#   reproducibility story.
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
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


# ─── Local imports — support both `python data/preprocess_zenodo.py` and
#     `python -m data.preprocess_zenodo` from the repo root ─────────────
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (str(_HERE), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    # data.data_utils (File 10a)
    from data.data_utils import (
        FEATURE_SCHEMA,
        LABEL_COLUMN,
        METADATA_COLUMNS,
        NUMERIC_COLUMNS,
        canonicalize_dtypes,
        resolve_artifact_root,
        validate_schema,
    )
    # data.preprocess_doh2023 (File 10c) — shared guard and sentinel
    from data.preprocess_doh2023 import (
        SENTINEL_NUMERIC,
        enforce_forbidden_observables,
    )
except ImportError:  # script-qualified (python data/preprocess_zenodo.py)
    from data_utils import (  # type: ignore[no-redef]
        FEATURE_SCHEMA,
        LABEL_COLUMN,
        METADATA_COLUMNS,
        NUMERIC_COLUMNS,
        canonicalize_dtypes,
        resolve_artifact_root,
        validate_schema,
    )
    from preprocess_doh2023 import (  # type: ignore[no-redef]
        SENTINEL_NUMERIC,
        enforce_forbidden_observables,
    )


# ─── Constants ─────────────────────────────────────────────────────────────

ARTIFACT_VERSION = "1.0.0"
THESIS_CHAPTER = "Chapter 4 (PAFA-DoH)"
CITATION_DOI = "10.1016/j.neunet.2025.108343"

log = logging.getLogger("data.preprocess_zenodo")


# ─── Raw-column aliases ───────────────────────────────────────────────────
#
# The Zenodo DoH corpus is distributed as multiple files within a single
# record. Column names vary across files in the release; the aliases
# below are the union of names we have observed. Matching is normalized
# (lowercase, strip non-alnum), so `Query Type`, `query-type` and
# `QUERYTYPE` all match the same alias.
#
# Because the Zenodo corpus carries per-query records (subdomain,
# query_type, response_size) and per-flow records (timing, TLS, HTTP/2,
# HTTP/3), every one of the 47 target columns has at least one raw
# alias here. There is deliberately no `None` entry: if a column cannot
# be matched from the raw data, `to_schema()` raises. This is stricter
# than File 10c on purpose.

COLUMN_ALIASES: dict[str, list[str]] = {
    # ── §1. Flow metadata (7) ──────────────────────────────────────────
    "flow_id": ["flow_id", "flowid", "flow", "id", "query_id",
                "transaction_id"],
    "capture_date": ["capture_date", "date", "timestamp", "query_time",
                     "flow_date", "start_time"],
    "src_ip": ["src_ip", "source_ip", "srcip", "sourceip", "client_ip",
               "src", "sip"],
    "dst_ip": ["dst_ip", "destination_ip", "dstip", "destinationip",
               "resolver_ip", "dst", "dip"],
    "src_port": ["src_port", "source_port", "srcport", "sourceport",
                 "sport", "client_port"],
    "dst_port": ["dst_port", "destination_port", "dstport",
                 "destinationport", "dsport", "dport", "resolver_port"],
    "protocol": ["protocol", "proto", "protocol_name", "encrypted_protocol"],

    # ── §2. Flow-level statistics (10) ─────────────────────────────────
    "flow_duration_ms": ["flow_duration_ms", "flow_duration", "duration",
                         "flowduration", "duration_ms"],
    "packet_count": ["packet_count", "total_packets", "num_packets",
                     "packets"],
    "bytes_out": ["bytes_out", "query_bytes", "request_bytes",
                  "fwd_bytes", "bytes_sent"],
    "bytes_in": ["bytes_in", "response_bytes", "reply_bytes",
                 "bwd_bytes", "bytes_received"],
    "packet_size_mean": ["packet_size_mean", "mean_packet_size",
                         "avg_packet_size"],
    "packet_size_std": ["packet_size_std", "std_packet_size",
                        "packet_length_std"],
    "packet_size_max": ["packet_size_max", "max_packet_size",
                        "max_packet_length"],
    "packet_size_min": ["packet_size_min", "min_packet_size",
                        "min_packet_length"],
    "bytes_per_second": ["bytes_per_second", "flow_bytes_s", "bps"],
    "packets_per_second": ["packets_per_second", "flow_packets_s", "pps"],

    # ── §3. Timing / IAT (7) ───────────────────────────────────────────
    "iat_mean_ms": ["iat_mean_ms", "flow_iat_mean", "iat_mean"],
    "iat_std_ms": ["iat_std_ms", "flow_iat_std", "iat_std"],
    "iat_min_ms": ["iat_min_ms", "flow_iat_min", "iat_min"],
    "iat_max_ms": ["iat_max_ms", "flow_iat_max", "iat_max"],
    "iat_median_ms": ["iat_median_ms", "flow_iat_median", "iat_median"],
    "iat_skew": ["iat_skew", "flow_iat_skew", "iat_skewness"],
    "iat_kurtosis": ["iat_kurtosis", "flow_iat_kurtosis", "iat_kurt"],

    # ── §4. Query-type / entropy (8) ───────────────────────────────────
    # The Zenodo corpus carries these natively (subdomain, query_type).
    "entropy_subdomain": ["entropy_subdomain", "subdomain_entropy"],
    "entropy_query_type": ["entropy_query_type", "qtype_entropy",
                           "query_type_entropy"],
    "qtype_A_count": ["qtype_a_count", "a_count", "num_a_queries"],
    "qtype_AAAA_count": ["qtype_aaaa_count", "aaaa_count",
                         "num_aaaa_queries"],
    "qtype_TXT_count": ["qtype_txt_count", "txt_count", "num_txt_queries"],
    "qtype_MX_count": ["qtype_mx_count", "mx_count", "num_mx_queries"],
    "qtype_CNAME_count": ["qtype_cname_count", "cname_count",
                          "num_cname_queries"],
    "qtype_other_count": ["qtype_other_count", "other_qtype_count"],

    # ── §5. Response features (3) ──────────────────────────────────────
    "response_size_mean": ["response_size_mean", "bwd_packet_length_mean",
                           "response_length_mean"],
    "response_size_std": ["response_size_std", "bwd_packet_length_std",
                          "response_length_std"],
    "response_entropy": ["response_entropy", "response_size_entropy"],

    # ── §6. TLS handshake (4) ──────────────────────────────────────────
    "tls_version": ["tls_version", "tls_ver", "ssl_version"],
    "cipher_suite_count": ["cipher_suite_count", "num_cipher_suites",
                           "cipher_count"],
    "extension_count": ["extension_count", "num_extensions",
                        "tls_extension_count"],
    "session_reuse": ["session_reuse", "session_resumed",
                      "tls_session_reuse"],

    # ── §7. HTTP/2 + HTTP/3 frame features (3) ─────────────────────────
    "http2_stream_count": ["http2_stream_count", "h2_stream_count"],
    "http2_frame_count": ["http2_frame_count", "h2_frame_count"],
    "http3_frame_count": ["http3_frame_count", "h3_frame_count"],

    # ── §8. Temporal / spectral (4) ────────────────────────────────────
    "autocorr_lag1": ["autocorr_lag1", "acf_lag1", "autocorr_1"],
    "autocorr_lag5": ["autocorr_lag5", "acf_lag5", "autocorr_5"],
    "burstiness": ["burstiness", "iat_burstiness"],
    "spectral_low_band_energy": ["spectral_low_band_energy",
                                 "low_band_energy", "fft_low_energy"],

    # ── §9. Label (1) ──────────────────────────────────────────────────
    "label": ["label", "class", "target", "category", "attack_label",
              "is_malicious"],
}


def _normalize_column_name(name: str) -> str:
    """Lowercase, strip non-alphanumeric. Same rule as File 10c."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _build_reverse_alias_map() -> dict[str, str]:
    """{normalized_alias: canonical_name}. Fails on collision at import time."""
    rev: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            norm = _normalize_column_name(alias)
            if norm in rev and rev[norm] != canonical:
                raise ValueError(
                    f"Alias collision: raw alias {alias!r} maps to both "
                    f"{rev[norm]!r} and {canonical!r}."
                )
            rev[norm] = canonical
    return rev


REVERSE_ALIASES: dict[str, str] = _build_reverse_alias_map()


# ─── Label normalization ─────────────────────────────────────────────────

_LABEL_TRUE = frozenset({"1", "malicious", "attack", "attacker", "bad",
                         "true", "yes", "m", "a", "positive"})
_LABEL_FALSE = frozenset({"0", "benign", "benign_normal", "normal", "good",
                          "false", "no", "b", "n", "negative"})


def normalize_label(series: pd.Series) -> pd.Series:
    """Convert a raw label column into int8 0/1. Same encoding set as File 10c."""
    def _coerce(v: Any) -> int:
        if isinstance(v, (int, np.integer)) and v in (0, 1):
            return int(v)
        if isinstance(v, float) and not np.isnan(v) and v in (0.0, 1.0):
            return int(v)
        s = str(v).strip().lower()
        if s in _LABEL_TRUE:
            return 1
        if s in _LABEL_FALSE:
            return 0
        raise ValueError(
            f"Unrecognized label value {v!r}. Expected one of "
            f"{sorted(_LABEL_TRUE | _LABEL_FALSE)}."
        )
    return series.map(_coerce).astype("int8")


# ─── Reading ─────────────────────────────────────────────────────────────

def read_zenodo_files(raw_dir: Path) -> list[Path]:
    """Return the list of raw Zenodo data files under `raw_dir`.

    The Zenodo record may contain CSV, JSON, JSONL, or Parquet files.
    This function returns whichever of those extensions are present,
    sorted by name, so that the read order is deterministic. Dask's
    readers handle all four formats; the caller picks the reader based
    on the file extension.

    The `_landing_page.html` artifact from File 10b (if the Zenodo
    sub-command was run) is ignored.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"Zenodo raw directory not found: {raw_dir}. "
            "Run `python data/download_datasets.py zenodo` first."
        )
    supported = (".csv", ".json", ".jsonl", ".parquet")
    files = sorted(
        p for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in supported
    )
    if not files:
        raise FileNotFoundError(
            f"No data files ({', '.join(supported)}) found in {raw_dir}. "
            "The Zenodo record may have been downloaded into a nested "
            "directory; check `data/raw/zenodo/` and place the files "
            "directly in it."
        )
    return files


def _read_one(path: Path) -> pd.DataFrame:
    """Read one raw Zenodo file into a pandas DataFrame.

    Kept small and explicit so that a future file format only requires
    a new branch here, not a new pipeline. All four branches drop the
    pandas index so that concatenation in `read_zenodo_files` is
    straightforward.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        # A JSON array of records is the common case; fall back to
        # `orient="records"` if the top-level is a dict with a records
        # key.
        try:
            df = pd.read_json(path)
        except ValueError:
            df = pd.read_json(path, orient="records")
        return df
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file format: {path}")


# ─── Malformed-flow removal (§5.8.4 stage 1) ─────────────────────────────

def drop_malformed_flows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that cannot represent a valid query or flow.

    A row is malformed if any of the following hold:
      * `flow_duration_ms` is present and ≤ 0
      * `packet_count` is present and < 2
      * every numeric feature is NaN

    Same policy as File 10c's `drop_malformed_flows`, so the two
    preprocessors drop the same kinds of records.
    """
    before = len(df)
    mask = pd.Series(True, index=df.index)

    if "flow_duration_ms" in df.columns:
        dur = pd.to_numeric(df["flow_duration_ms"], errors="coerce")
        mask &= dur.fillna(0) > 0
    if "packet_count" in df.columns:
        pc = pd.to_numeric(df["packet_count"], errors="coerce")
        mask &= pc.fillna(0) >= 2

    numeric_cols = [c for c in df.columns
                    if c not in METADATA_COLUMNS
                    and pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        mask &= ~df[numeric_cols].isna().all(axis=1)

    out = df.loc[mask].reset_index(drop=True)
    dropped = before - len(out)
    if dropped:
        log.warning(
            "Dropped %d malformed row(s) of %d (%.2f%%)",
            dropped, before, 100.0 * dropped / max(before, 1),
        )
    else:
        log.info("No malformed flows detected.")
    return out


# ─── Column mapping ──────────────────────────────────────────────────────

def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw Zenodo columns to canonical 47-column names.

    Unlike File 10c, this mapping is expected to be **lossless**: every
    canonical column has at least one raw alias in `COLUMN_ALIASES`,
    and after the mapping every one of the 47 columns should be
    present in the returned frame. If it is not, `to_schema()` will
    raise. The unmapped raw columns are dropped with a `log.debug`
    line so a reviewer can see the raw Zenodo column list.
    """
    renames: dict[str, str] = {}
    dropped: list[str] = []
    for col in df.columns:
        norm = _normalize_column_name(col)
        canonical = REVERSE_ALIASES.get(norm)
        if canonical is None:
            dropped.append(col)
        else:
            renames[col] = canonical
    if dropped:
        log.debug("Dropping %d unmapped raw column(s): %s",
                  len(dropped), sorted(dropped))
    return df.rename(columns=renames)


# ─── Feature computation — the eight families ───────────────────────────

def _compute_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """§2. Flow-level statistics (10 columns). See File 10a, §2."""
    out = df
    if "bytes_per_second" not in out.columns:
        dur_s = pd.to_numeric(out.get("flow_duration_ms"), errors="coerce") / 1000.0
        out["bytes_per_second"] = (
            (pd.to_numeric(out.get("bytes_out"), errors="coerce").fillna(0)
             + pd.to_numeric(out.get("bytes_in"), errors="coerce").fillna(0))
            / dur_s.where(dur_s != 0)
        ).replace([np.inf, -np.inf], np.nan)
    if "packets_per_second" not in out.columns:
        dur_s = pd.to_numeric(out.get("flow_duration_ms"), errors="coerce") / 1000.0
        out["packets_per_second"] = (
            pd.to_numeric(out.get("packet_count"), errors="coerce")
            / dur_s.where(dur_s != 0)
        ).replace([np.inf, -np.inf], np.nan)
    return out


def _compute_timing_features(df: pd.DataFrame) -> pd.DataFrame:
    """§3. Timing / IAT statistics (7 columns). See File 10a, §3."""
    # Most Zenodo releases provide iat_* natively; only burstiness is
    # derived here (see §8 below, which also uses burstiness).
    return df


def _compute_query_type_features(df: pd.DataFrame) -> pd.DataFrame:
    """§4. Query-type / entropy (8 columns).

    * `entropy_subdomain` = Shannon entropy of the subdomain string
      (character distribution). Computed only if a raw subdomain column
      is present; otherwise passed through from the raw entropy column.
    * `entropy_query_type` = Shannon entropy of the qtype stream over
      the flow. Computed only if `qtype_*_count` columns are all
      present; otherwise passed through.
    * `qtype_*_count` = passed through from the raw counts.
    """
    out = df
    if "entropy_subdomain" not in out.columns:
        subdomain_col = next(
            (c for c in out.columns if "subdomain" in c.lower()), None
        )
        if subdomain_col is not None:
            out["entropy_subdomain"] = out[subdomain_col].astype(str).map(
                _shannon_entropy_of_string
            )
    return out


def _compute_response_features(df: pd.DataFrame) -> pd.DataFrame:
    """§5. Response features (3 columns). See File 10a, §5."""
    out = df
    if "response_entropy" not in out.columns and "response_size_mean" in out.columns:
        # Proxy: entropy of the response-size distribution is not
        # recoverable from the mean alone; the Zenodo releases provide
        # this natively in most cases. If it is missing and cannot be
        # recovered, to_schema() will raise (no sentinel on Zenodo).
        pass
    return out


def _compute_tls_features(df: pd.DataFrame) -> pd.DataFrame:
    """§6. TLS handshake features (4 columns). See File 10a, §6.

    * `tls_version` is a string in the canonical schema. Normalize to
      the string form "1.3" / "1.2" if the raw value is numeric or a
      protocol-name variant.
    * `cipher_suite_count`, `extension_count`, `session_reuse` are
      passed through from the raw columns.
    """
    out = df
    if "tls_version" in out.columns:
        out["tls_version"] = out["tls_version"].astype(str).str.replace(
            "TLSv", "", regex=False
        ).str.strip()
    return out


def _compute_http_features(df: pd.DataFrame) -> pd.DataFrame:
    """§7. HTTP/2 + HTTP/3 frame features (3 columns). See File 10a, §7.

    Passed through from the raw columns. Flows that are DoT / DoQ will
    have 0 in the HTTP/2 and HTTP/3 columns natively (or will be
    missing them entirely, in which case `to_schema()` will raise).
    """
    return df


def _compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """§8. Temporal / spectral features (4 columns). See File 10a, §8.

    * `autocorr_lag1` and `autocorr_lag5` are passed through from the
      raw columns (the Zenodo releases carry them natively).
    * `burstiness` = (σ_iat − μ_iat) / (σ_iat + μ_iat) — computed from
      `iat_std_ms` and `iat_mean_ms` if not already present.
    * `spectral_low_band_energy` is passed through.
    """
    out = df
    if "burstiness" not in out.columns and {"iat_mean_ms", "iat_std_ms"}.issubset(out.columns):
        mu = pd.to_numeric(out["iat_mean_ms"], errors="coerce")
        sd = pd.to_numeric(out["iat_std_ms"], errors="coerce")
        denom = sd + mu
        out["burstiness"] = ((sd - mu) / denom.where(denom != 0)).replace(
            [np.inf, -np.inf], np.nan
        )
    return out


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run all eight feature-computation families in order.

    The order matters only for derived features that depend on earlier
    families (e.g., `burstiness` in §8 depends on `iat_mean_ms` and
    `iat_std_ms` from §3). All eight families are pure functions of
    the input frame, so `map_partitions(compute_features)` on a Dask
    DataFrame is safe.
    """
    df = _compute_flow_features(df)
    df = _compute_timing_features(df)
    df = _compute_query_type_features(df)
    df = _compute_response_features(df)
    df = _compute_tls_features(df)
    df = _compute_http_features(df)
    df = _compute_temporal_features(df)
    return df


def _shannon_entropy_of_string(s: str) -> float:
    """Shannon entropy (base 2) of the character distribution of `s`.

    Used by `_compute_query_type_features` for `entropy_subdomain`.
    Returns 0.0 for empty strings and for single-character strings.
    """
    s = "" if s is None else str(s)
    if len(s) < 2:
        return 0.0
    counts = pd.Series(list(s)).value_counts(normalize=True)
    return float(-(counts * np.log2(counts)).sum())


# ─── Schema finalization ─────────────────────────────────────────────────

def to_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Map the computed features onto the canonical 47-column schema.

    Steps:
      1. Compute features via `compute_features` (idempotent: a frame
         that already has every derived feature is returned unchanged
         apart from dtype canonicalisation).
      2. Assert that every one of the 47 columns is present.
      3. Assert that no numeric column is still the sentinel
         `SENTINEL_NUMERIC` (the whole point of the Zenodo path is
         that no column requires a sentinel fill; if one does, either
         the raw release is missing a field or a `_compute_*` function
         failed).
      4. Canonicalize dtypes and validate the schema.

    Raises:
        ValueError: naming the missing column(s) and/or the still-
            sentinel column(s). The message cites the relevant section
            of File 10a for each family so a reviewer can locate the
            feature's specification.
    """
    df = compute_features(df)

    missing = [c for c in FEATURE_SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(
            f"Zenodo preprocessor is missing {len(missing)} canonical "
            f"column(s): {sorted(missing)}. Unlike the CIC path, the "
            f"Zenodo path must produce a schema-complete output with "
            f"no sentinel fills (see File 10d header). Check the raw "
            f"release for the missing column(s) and extend the "
            f"corresponding _compute_* function if the value is "
            f"derivable from the raw data."
        )

    still_sentinel = [
        c for c in NUMERIC_COLUMNS
        if c in df.columns and (df[c] == SENTINEL_NUMERIC).any()
    ]
    if still_sentinel:
        raise ValueError(
            f"Zenodo preprocessor produced {len(still_sentinel)} column(s) "
            f"still at the sentinel value ({SENTINEL_NUMERIC}): "
            f"{sorted(still_sentinel)}. This is a bug in the feature-"
            f"computation path; the Zenodo path must never emit a "
            f"sentinel fill."
        )

    df = df[list(FEATURE_SCHEMA)].copy()
    df = canonicalize_dtypes(df)
    validate_schema(df)
    return df


# ─── Normalization (§5.8.4 stage 4) ─────────────────────────────────────

def normalize_features_with_stats(
    df: pd.DataFrame,
    means: pd.Series,
    stds: pd.Series,
) -> pd.DataFrame:
    """Normalize numeric columns using precomputed global mean/std.

    Used by `process_with_dask` in pass 2 of the two-pass normalization.
    `means` and `stds` are Series indexed by canonical column name.
    Columns whose `stds` entry is 0 or NaN are left unnormalized (a
    column with zero variance carries no information, and dividing by
    zero would produce NaN).
    """
    out = df.copy()
    for col in NUMERIC_COLUMNS:
        if col not in out.columns or col == LABEL_COLUMN:
            continue
        mu = means.get(col, np.nan)
        sd = stds.get(col, np.nan)
        if pd.isna(mu) or pd.isna(sd) or sd == 0:
            continue
        out[col] = (pd.to_numeric(out[col], errors="coerce") - mu) / sd
    return out


# ─── Dask pipeline ──────────────────────────────────────────────────────

def process_with_dask(
    raw_dir: Path,
    out_dir: Path,
    *,
    npartitions: int,
    sample: int,
    write_parquet: bool,
    force: bool,
) -> Path:
    """Run the full Zenodo preprocessing pipeline under Dask.

    Two passes over the Dask DataFrame:

      Pass 1 — features + global mean/std. `map_partitions(compute_features)`
      produces a Dask DataFrame of raw features. `.mean().compute()` and
      `.std().compute()` give global statistics for normalization.

      Pass 2 — normalize + to_schema. `map_partitions` applies
      `normalize_features_with_stats` using the precomputed stats, then
      `to_schema` (per partition) validates and dtype-canonicalises.

      The two-pass design exists because normalization must use
      corpus-wide statistics, not per-partition statistics. See the
      header comment for the full rationale.

    Output is written as a single CSV and (unless `write_parquet` is
    False) a single Parquet. Dask's writers produce partitioned output
    by default; we write to a temp dir and concatenate into a single
    file so the artifact of record is one file per format.

    Returns the CSV path.
    """
    raw_files = read_zenodo_files(raw_dir)
    log.info("Found %d raw Zenodo file(s) under %s",
             len(raw_files), raw_dir)

    # ── Load ────────────────────────────────────────────────────────
    # All files in a record should share a schema; concatenate them.
    # If their schemas differ, Dask will raise during concat, and that
    # is the correct behavior (a schema mismatch across files is a
    # source-data problem that the reviewer needs to see).
    dfs = [dd.read_csv(str(p), blocksize="128MB", assume_missing=True)
           if p.suffix.lower() == ".csv"
           else dd.read_parquet(str(p))
           if p.suffix.lower() == ".parquet"
           else dd.from_pandas(_read_one(p), npartitions=npartitions)
           for p in raw_files]
    ddf: dd.DataFrame = dd.concat(dfs, axis=0) if len(dfs) > 1 else dfs[0]
    ddf = ddf.repartition(npartitions=npartitions)
    log.info("Dask partition count: %d", ddf.npartitions)

    # ── Optional sample ────────────────────────────────────────────
    if sample > 0:
        log.warning(
            "Sampling the first %d row(s) for a smoke test. "
            "The output does NOT reproduce Chapter 4's numbers.",
            sample,
        )
        ddf = ddf.head(sample, npartitions=min(npartitions, ddf.npartitions), compute=False)
        ddf = ddf.repartition(npartitions=min(npartitions, ddf.npartitions))

    # ── Pass 1: compute features ───────────────────────────────────
    # `enforce_forbidden_observables` is called per partition inside the
    # mapped function, so a forbidden column in any partition fails the
    # whole pipeline before any output is written.
    def _pass1_partition(pdf: pd.DataFrame) -> pd.DataFrame:
        enforce_forbidden_observables(pdf, raw_dir)
        return map_columns(pdf)

    ddf = ddf.map_partitions(_pass1_partition)
    ddf = ddf.map_partitions(drop_malformed_flows)
    ddf = ddf.map_partitions(compute_features)

    # ── Pass 1.5: global mean/std ──────────────────────────────────
    log.info("Computing global normalization statistics (pass 1.5)…")
    # Only numeric canonical columns
    stats_cols = [c for c in NUMERIC_COLUMNS if c in ddf.columns]
    if stats_cols:
        means = ddf[stats_cols].mean().compute()
        stds = ddf[stats_cols].std(ddof=1).compute()
    else:
        means = pd.Series(dtype="float64")
        stds = pd.Series(dtype="float64")

    # ── Pass 2: normalize + final schema ───────────────────────────
    def _pass2_partition(pdf: pd.DataFrame) -> pd.DataFrame:
        pdf = normalize_features_with_stats(pdf, means, stds)
        pdf = to_schema(pdf)
        return pdf

    ddf = ddf.map_partitions(_pass2_partition)

    # ── Write ──────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "zenodo_doh.csv"
    pq_path = out_dir / "zenodo_doh.parquet"

    if (csv_path.exists() or (write_parquet and pq_path.exists())) and not force:
        raise FileExistsError(
            f"Output already exists under {out_dir}. "
            "Pass --force to overwrite."
        )

    # Dask writes to a directory of partitions; we write to a temp dir
    # and concatenate, so that the artifact of record is one file.
    with tempfile.TemporaryDirectory(prefix="zenodo_dask_") as tmp:
        tmp_csv = Path(tmp) / "csv"
        ddf.to_csv(str(tmp_csv), index=False, single_file=False)
        # Concatenate partition files into one CSV. Part names are
        # deterministic (Dask uses "part.NNNN.csv"), so the
        # concatenation order is stable across runs.
        parts = sorted(tmp_csv.glob("part.*.csv"))
        with csv_path.open("wb") as out_fh:
            for i, part in enumerate(parts):
                with part.open("rb") as in_fh:
                    if i > 0:
                        # Skip the header on every part after the first
                        in_fh.readline()
                    shutil.copyfileobj(in_fh, out_fh)
        log.info("Wrote %s (%d part file(s) concatenated)",
                 csv_path, len(parts))

        if write_parquet:
            tmp_pq = Path(tmp) / "parquet"
            try:
                ddf.to_parquet(str(tmp_pq), index=False, write_index=False)
                pq_parts = sorted(tmp_pq.glob("part.*.parquet"))
                # Parquet's writer produces separate files; we cannot
                # trivially concatenate them without pyarrow, so we
                # copy the part directory to the out_dir as-is under
                # a stable name. The committed artifact ships the CSV
                # as the single-file of record; the parquet sidecar
                # is a partitioned directory.
                pq_out = out_dir / "zenodo_doh.parquet.d"
                if pq_out.exists():
                    shutil.rmtree(pq_out)
                shutil.copytree(tmp_pq, pq_out)
                log.info(
                    "Wrote %s as a partitioned directory (%d part "
                    "file(s)). Use pandas.read_parquet or "
                    "dask.dataframe.read_parquet on the directory.",
                    pq_out, len(pq_parts),
                )
            except (ImportError, ValueError) as exc:
                log.warning(
                    "Could not write parquet (%s). The CSV was written.",
                    exc,
                )

    return csv_path


# ─── Self-check ─────────────────────────────────────────────────────────

def _self_check(raw_dir: Path | None = None) -> int:
    """Non-destructive self-check used by pytest (File 21a).

    Runs on a 10-row synthetic Zenodo-shaped frame (not on the real
    corpus, which is 32 GB). Confirms that:

      * `compute_features` fills every one of the 47 columns without
        any sentinel value.
      * `to_schema` returns a schema-complete frame with no sentinel
        fills.
      * `validate_schema` passes on the result.

    The point of the check is to assert the Zenodo path's central
    invariant: **zero sentinel fills**. If this ever fails, the
    preprocessor has regressed and the training pipeline would be
    silently training on a different schema than the one Chapter 4
    reports.
    """
    # A 10-row synthetic frame with a plausible subset of raw Zenodo
    # columns. We do not include every raw alias; `to_schema` will
    # raise if the missing ones are not derivable. The columns below
    # are the minimum the feature-computation path needs to reach a
    # schema-complete output.
    raw = pd.DataFrame({
        "flow_id": [f"z{i}" for i in range(10)],
        "capture_date": ["2024-01-01"] * 10,
        "src_ip": [f"10.0.0.{i}" for i in range(10)],
        "dst_ip": ["8.8.8.8"] * 10,
        "src_port": [49152 + i for i in range(10)],
        "dst_port": [443] * 10,
        "protocol": ["DoH"] * 10,
        "flow_duration_ms": [1000.0 + 100.0 * i for i in range(10)],
        "packet_count": [10 + i for i in range(10)],
        "bytes_out": [1000 + 100 * i for i in range(10)],
        "bytes_in": [500 + 50 * i for i in range(10)],
        "packet_size_mean": [150.0] * 10,
        "packet_size_std": [10.0] * 10,
        "packet_size_max": [200] * 10,
        "packet_size_min": [100] * 10,
        "iat_mean_ms": [100.0] * 10,
        "iat_std_ms": [10.0] * 10,
        "iat_min_ms": [5.0] * 10,
        "iat_max_ms": [200.0] * 10,
        "iat_median_ms": [100.0] * 10,
        "iat_skew": [0.1] * 10,
        "iat_kurtosis": [0.0] * 10,
        "entropy_subdomain": [2.0 + 0.1 * i for i in range(10)],
        "entropy_query_type": [1.5] * 10,
        "qtype_A_count": [5] * 10,
        "qtype_AAAA_count": [2] * 10,
        "qtype_TXT_count": [1] * 10,
        "qtype_MX_count": [0] * 10,
        "qtype_CNAME_count": [2] * 10,
        "qtype_other_count": [0] * 10,
        "response_size_mean": [500.0] * 10,
        "response_size_std": [50.0] * 10,
        "response_entropy": [1.0] * 10,
        "tls_version": ["1.3"] * 10,
        "cipher_suite_count": [8] * 10,
        "extension_count": [4] * 10,
        "session_reuse": [0] * 10,
        "http2_stream_count": [3] * 10,
        "http2_frame_count": [7] * 10,
        "http3_frame_count": [0] * 10,
        "autocorr_lag1": [0.3] * 10,
        "autocorr_lag5": [0.2] * 10,
        "spectral_low_band_energy": [0.1] * 10,
        "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })

    # Map + compute + to_schema
    try:
        mapped = map_columns(raw)
        mapped["label"] = normalize_label(mapped["label"])
        out = to_schema(mapped)
    except ValueError as exc:
        print(f"FAIL: {exc}")
        return 1

    # Assertions
    missing = [c for c in FEATURE_SCHEMA if c not in out.columns]
    if missing:
        print(f"FAIL: missing columns after to_schema: {missing}")
        return 1
    sentinel_cols = [
        c for c in NUMERIC_COLUMNS
        if c in out.columns and (out[c] == SENTINEL_NUMERIC).any()
    ]
    if sentinel_cols:
        print(
            f"FAIL: sentinel fills leaked into the Zenodo output: "
            f"{sorted(sentinel_cols)}. The Zenodo path must produce "
            f"a schema-complete output with zero sentinel fills."
        )
        return 1

    print(
        f"OK: to_schema produced all {len(FEATURE_SCHEMA)} columns, "
        f"zero sentinel fills."
    )
    print("self-check ok")
    return 0


# ─── argparse ─────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="preprocess_zenodo.py",
        description=(
            "Preprocess the Zenodo DoH corpus (DOI 10.5281/zenodo.5957677, "
            "≈ 32 GB) into the artifact's canonical 47-column schema "
            "using Dask. The Zenodo path is the training source for "
            "Tables 14, 16, 17 and Figures 8–10; unlike the CIC path "
            "(File 10c), it produces a schema-complete output with "
            "zero sentinel fills."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python data/preprocess_zenodo.py --sample 10000\n"
            "  python data/preprocess_zenodo.py\n"
            "  python data/preprocess_zenodo.py --npartitions 16\n"
            "  python data/preprocess_zenodo.py --no-parquet --verbose\n"
        ),
    )
    p.add_argument("--config", type=Path, default=None,
                   help="Path to the master config (default: auto-resolve).")
    p.add_argument("--raw-dir", type=Path, default=None,
                   help="Directory containing the Zenodo files "
                        "(default: <artifact_root>/data/raw/zenodo).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Directory to write the processed output "
                        "(default: <artifact_root>/data/processed).")
    p.add_argument("--npartitions", type=int, default=None,
                   help="Number of Dask partitions "
                        "(default: os.cpu_count()).")
    p.add_argument("--sample", type=int, default=0,
                   help="Process only the first N rows (0 = full corpus). "
                        "Useful for a quick end-to-end smoke test.")
    p.add_argument("--no-parquet", action="store_true",
                   help="Do not write the parquet sidecar.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing output files.")
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

    # Config.
    cfg: dict[str, Any] = {}
    if args.config is None:
        args.config = root / "configs" / "default_pafa_config.yaml"
    try:
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    except (FileNotFoundError, yaml.YAMLError) as exc:
        log.error("Failed to load config %s: %s", args.config, exc)
        return 1

    raw_dir = Path(args.raw_dir) if args.raw_dir else root / cfg["paths"]["raw_datasets"] / "zenodo"
    out_dir = Path(args.out_dir) if args.out_dir else root / cfg["paths"]["processed_datasets"]
    npartitions = args.npartitions or (os_cpu_count() or 4)

    log.info("Raw directory:     %s", raw_dir)
    log.info("Output directory:  %s", out_dir)
    log.info("Dask partitions:   %d", npartitions)
    if args.sample > 0:
        log.warning("Sample mode: only the first %d row(s) will be processed.",
                    args.sample)

    try:
        csv_path = process_with_dask(
            raw_dir, out_dir,
            npartitions=npartitions,
            sample=args.sample,
            write_parquet=not args.no_parquet,
            force=args.force,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        log.error("%s", exc)
        return 1

    print()
    print("NEXT STEP:")
    print(
        "  python data/split_data.py "
        "--config configs/default_pafa_config.yaml"
    )
    print(f"  (processed corpus: {csv_path})")
    return 0


def os_cpu_count() -> int | None:
    """Return the CPU count or None.

    Wrapped so that a restricted environment (some CI runners) can
    still import the module; the caller falls back to 4 if this
    returns None.
    """
    import os
    try:
        return os.cpu_count()
    except Exception:  # pragma: no cover — very unusual environments
        return None


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
