#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# preprocess_doh2023.py — PAFA-DoH Reproducibility Artifact
# ═══════════════════════════════════════════════════════════════════════════
#
# Companion to:
#   README.md, REPRODUCIBILITY_CHECKLIST.md, LICENSE, requirements.txt,
#   environment.yml, Dockerfile, setup.py, .gitignore   (Files 1–8)
#   configs/default_pafa_config.yaml                    (File 9, master config)
#   configs/*.yaml                                      (Files 9b–9h)
#   data/data_utils.py                                  (File 10a, schema +
#                                                        checksum + split +
#                                                        root resolution)
#   data/download_datasets.py                           (File 10b)
#
# Thesis chapter: Chapter 4 (PAFA-DoH)
# Citation DOI:   10.1016/j.neunet.2025.108343   (Neural Networks, April 2026)
# License:        MIT (this file, as part of the artifact's source code)
# Artifact version: 1.0.0
#
# ───────────────────────────────────────────────────────────────────────────
# Purpose
# ───────────────────────────────────────────────────────────────────────────
# Preprocess the CIC-DoH2023 bundle (≈ 2.5 GB of CSV flow features) into
# the artifact's canonical 47-column schema (defined in
# data/data_utils.py, File 10a). The output is what data/split_data.py
# (File 10e) consumes; nothing downstream reads the raw CIC CSVs
# directly.
#
# CIC-DoH2023 is used for the 47-column **schema-parity check** on the
# preprocessor (see File 9's header comment on `datasets.cic_doh2023`).
# It is not a training source. Its role is to confirm that the
# preprocessor produces the same canonical schema on a second,
# independent corpus as it does on the Zenodo DoH corpus
# (File 10d). If the two outputs match column-for-column and dtype-for-
# dtype, the schema is stable and downstream components can trust it.
#
# ───────────────────────────────────────────────────────────────────────────
# Where the pieces are specified in Chapter 4
# ───────────────────────────────────────────────────────────────────────────
#   Dataset identity / size      → §4.24.1, File 9 header
#   Composition / preprocessing  → §4.25.1
#   47-column feature schema     → data/data_utils.py::FEATURE_SCHEMA
#                                  (File 10a). The schema is the same
#                                  one the chapter's Table 14–17
#                                  expected outputs were produced with.
#   Forbidden observables        → §1.7.2 / §4.13, and File 9's
#                                  `threat_model.forbidden_observables`.
#                                  This preprocessor MUST NOT read any
#                                  column that carries query or response
#                                  payload content, application-layer
#                                  semantics, or database information.
#                                  The `enforce_forbidden_observables()`
#                                  function below fails closed if such a
#                                  column appears in a raw CIC CSV.
#   Preprocessing stages         → §5.8.4 (a five-stage pipeline:
#                                  malformed-flow removal, timestamp
#                                  and label unification, statistical
#                                  descriptor extraction, feature
#                                  normalization, and split
#                                  preparation). This script performs
#                                  the first four; split preparation
#                                  lives in data/split_data.py (File
#                                  10e).
#
# ───────────────────────────────────────────────────────────────────────────
# Design notes
# ───────────────────────────────────────────────────────────────────────────
# • The 47-column schema is IMPORTED from data.data_utils. It is never
#   redeclared here. If the schema in File 10a changes, this file's
#   output changes with it, with no possibility of drift.
#
# • CIC-DoH2023's CSV column names vary across releases. The
#   `COLUMN_ALIASES` dict below lists, for each of the 47 target
#   columns, the raw CIC column names that carry the same quantity.
#   Alias matching is case-insensitive and strips underscores /
#   spaces / hyphens, so `Flow Duration`, `Flow_Duration` and
#   `FlowDuration` all match the same alias. A column with no alias
#   (value `None`) must be derived from other columns (see
#   `derive_missing_features()`).
#
# • CIC-DoH2023 is a flow-feature dataset; it does not contain
#   payload bytes. The six forbidden observables never appear in its
#   column names in any release we have seen, but the check is run
#   anyway, because (a) a future CIC release could add them, and
#   (b) an automatic check is cheaper than trusting the source.
#
# • Feature derivation. Features that CIC does not directly expose but
#   that are derivable from what CIC does expose (e.g., ACF at lag 1
#   from a flow's packet-size series) are computed in
#   `derive_missing_features()`. Features that cannot be derived from
#   flow-level data (e.g., `entropy_subdomain` — CIC flow records do
#   not carry the subdomain string) are filled with the documented
#   sentinel below and the column name is logged. The `--strict-
#   schema` flag turns this into a hard error, so a reviewer can see
#   exactly which columns are real vs. sentinel on a given CIC release.
#
# • Output format. Both `cic_doh2023.csv` and (optionally, on by
#   default) `cic_doh2023.parquet` are written. The CSV is the human-
#   readable artifact; the parquet is the canonical serialized form
#   for Dask (File 10d uses the same pattern for Zenodo). `--no-
#   parquet` disables the parquet output for environments where
#   `pyarrow` is not installed.
#
# • Fail-closed on schema violations. If the derived DataFrame does
#   not pass `validate_schema()` (File 10a), the script raises and
#   writes nothing. It does not emit a "best effort" CSV that a
#   downstream component might silently consume.
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


# ─── Local imports — support both `python data/preprocess_doh2023.py` and
#     `python -m data.preprocess_doh2023` from the repo root ─────────────
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
except ImportError:  # script-qualified
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

# Sentinel for numeric columns that cannot be derived from CIC flow-level
# data. The value is deliberately outside the empirical range of every
# real feature and is documented here so a reviewer can grep for `-1.0`
# in a processed CSV and know it means "not present in this source".
SENTINEL_NUMERIC: float = -1.0

log = logging.getLogger("data.preprocess_doh2023")


# ─── Forbidden-observable guard ───────────────────────────────────────────

# The four forbidden observables in File 9 (threat_model.forbidden_observables)
# are: query_payload_content, response_payload_content,
# application_layer_semantics, database_information. CIC-DoH2023 is a
# flow-feature dataset, so these columns should never appear. If they do
# (a future CIC release, or a derivative CSV), this preprocessor refuses
# to read them.
FORBIDDEN_RAW_COLUMNS: frozenset[str] = frozenset({
    # Query / response payload (the first two forbidden observables)
    "query_payload", "query_content", "query_bytes", "query_raw",
    "response_payload", "response_content", "response_bytes", "response_raw",
    # Application-layer semantics (the third forbidden observable)
    "application_semantics", "app_layer_semantics", "app_payload",
    # Database information (the fourth forbidden observable)
    "database_info", "db_info", "db_payload",
})


def enforce_forbidden_observables(df: pd.DataFrame, source: Path) -> None:
    """Fail closed if any forbidden raw column is present.

    The check normalizes column names (lowercase, strip non-alnum) and
    compares against `FORBIDDEN_RAW_COLUMNS`. The normalization handles
    the common CIC name variants (`Query_Payload`, `query-payload`,
    `QUERYPAYLOAD`) without enumerating every casing.

    Raises:
        ValueError: naming the offending columns and the source file, and
            citing the thesis sections that define the four forbidden
            observables.
    """
    normalized = {
        col: "".join(ch for ch in str(col).lower() if ch.isalnum())
        for col in df.columns
    }
    forbidden_normalized = {
        "".join(ch for ch in f.lower() if ch.isalnum()): f
        for f in FORBIDDEN_RAW_COLUMNS
    }
    hits = {
        raw_name: canonical
        for raw_name, norm in normalized.items()
        if norm in forbidden_normalized
    }
    if hits:
        raise ValueError(
            f"Refusing to read forbidden-observable columns from {source}: "
            f"{sorted(hits)}. The four forbidden observables are defined "
            f"in §1.7.2 / §4.13 of the thesis and in "
            f"configs/default_pafa_config.yaml::threat_model.forbidden_observables. "
            f"A column that carries query or response payload content, "
            f"application-layer semantics, or database information may not "
            f"be read by any component of this artifact."
        )


# ─── Alias table ──────────────────────────────────────────────────────────
#
# For each of the 47 target columns, either:
#   * a list of raw CIC column-name aliases (matched case-insensitively,
#     with non-alnum stripped), or
#   * None — meaning the column cannot be passed through from a CIC CSV
#     and must be derived in `derive_missing_features()` (or, if not
#     derivable, filled with the sentinel).
#
# The alias lists below are the union of column names observed in the
# public CIC-DoH2023 releases we have checked. They are deliberately
# permissive: matching is normalized, so an unusual release with e.g.
# `flow.duration` or `FlowDuration` will still match the alias
# `flow_duration`.

COLUMN_ALIASES: dict[str, list[str] | None] = {
    # ── §1. Flow metadata (7) ──────────────────────────────────────────
    "flow_id": ["flow_id", "flowid", "flow", "id"],
    "capture_date": ["capture_date", "date", "timestamp", "flow_date"],
    "src_ip": ["src_ip", "source_ip", "srcip", "sourceip", "src", "sip"],
    "dst_ip": ["dst_ip", "destination_ip", "dstip", "destinationip", "dst", "dip"],
    "src_port": ["src_port", "source_port", "srcport", "sourceport", "sport"],
    "dst_port": ["dst_port", "destination_port", "dstport", "destinationport",
                 "dsport", "dport"],
    "protocol": ["protocol", "proto", "protocol_name"],

    # ── §2. Flow-level statistics (10) ─────────────────────────────────
    "flow_duration_ms": ["flow_duration", "flow_duration_ms", "duration",
                         "flowduration"],
    "packet_count": ["packet_count", "total_packets", "total_fwd_packets",
                     "total_backward_packets"],
    "bytes_out": ["bytes_out", "total_length_of_fwd_packets",
                  "total_fwd_bytes", "fwd_bytes"],
    "bytes_in": ["bytes_in", "total_length_of_backward_packets",
                 "total_backward_bytes", "bwd_bytes"],
    "packet_size_mean": ["packet_size_mean", "flow_packets_s",
                         "fwd_packet_length_mean"],
    "packet_size_std": ["packet_size_std", "fwd_packet_length_std",
                        "packet_length_std"],
    "packet_size_max": ["packet_size_max", "max_packet_length",
                        "fwd_packet_length_max"],
    "packet_size_min": ["packet_size_min", "min_packet_length",
                        "fwd_packet_length_min"],
    "bytes_per_second": ["bytes_per_second", "flow_bytes_s"],
    "packets_per_second": ["packets_per_second", "flow_packets_s"],

    # ── §3. Timing / IAT (7) ───────────────────────────────────────────
    "iat_mean_ms": ["iat_mean_ms", "flow_iat_mean", "iat_mean"],
    "iat_std_ms": ["iat_std_ms", "flow_iat_std", "iat_std"],
    "iat_min_ms": ["iat_min_ms", "flow_iat_min", "iat_min"],
    "iat_max_ms": ["iat_max_ms", "flow_iat_max", "iat_max"],
    "iat_median_ms": ["iat_median_ms", "flow_iat_median", "iat_median"],
    "iat_skew": ["iat_skew", "flow_iat_skew", "iat_skewness"],
    "iat_kurtosis": ["iat_kurtosis", "flow_iat_kurtosis", "iat_kurt"],

    # ── §4. Query-type / entropy (8) ───────────────────────────────────
    # CIC flow records do NOT carry the subdomain string, so entropy_subdomain
    # and the qtype_*_count columns cannot be passed through from a CIC
    # CSV. They are derivable only from the Zenodo DoH corpus (File 10d),
    # which carries per-query records. The aliases below are best-effort
    # pass-throughs for future CIC releases that may add these fields.
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
    "response_entropy": None,  # derivable only from per-response records

    # ── §6. TLS handshake (4) ──────────────────────────────────────────
    # CIC-DoH2023 records are post-handshake flows; the handshake
    # parameters are not present as per-flow columns in the releases
    # we've seen. These are derivable only from packet captures, not
    # from CIC's pre-computed flow CSVs.
    "tls_version": None,
    "cipher_suite_count": None,
    "extension_count": None,
    "session_reuse": None,

    # ── §7. HTTP/2 + HTTP/3 frame features (3) ─────────────────────────
    "http2_stream_count": ["http2_stream_count", "h2_stream_count"],
    "http2_frame_count": ["http2_frame_count", "h2_frame_count"],
    "http3_frame_count": ["http3_frame_count", "h3_frame_count"],

    # ── §8. Temporal / spectral (4) ────────────────────────────────────
    # ACF and spectral features are derivable from the IAT series, but
    # CIC's flow-level CSV does not expose the per-packet IAT series
    # needed to compute them. We therefore mark them as `None` here and
    # document the derivation path (from raw PCAPs) in the
    # `derive_missing_features` docstring below.
    "autocorr_lag1": None,
    "autocorr_lag5": None,
    "burstiness": ["burstiness", "iat_burstiness"],
    "spectral_low_band_energy": None,

    # ── §9. Label (1) ──────────────────────────────────────────────────
    "label": ["label", "class", "target", "category"],
}


def _normalize_column_name(name: str) -> str:
    """Normalize a raw column name for alias matching.

    Lowercase, strip non-alphanumeric characters. `Flow Duration`,
    `flow-duration` and `FLOW_DURATION` all normalize to
    `flowduration`.
    """
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _build_reverse_alias_map() -> dict[str, str]:
    """Return {normalized_alias: canonical_name}.

    Built once at import time from `COLUMN_ALIASES`. A canonical name
    that has `None` in the alias table is skipped: those columns must be
    derived, not passed through.
    """
    rev: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        if aliases is None:
            continue
        for alias in aliases:
            norm = _normalize_column_name(alias)
            if norm in rev and rev[norm] != canonical:
                # Alias collision: two canonical columns both claim the
                # same raw name. This is a bug in the alias table;
                # fail at import time so it can never silently mis-map
                # a real CIC column.
                raise ValueError(
                    f"Alias collision: raw alias {alias!r} maps to both "
                    f"{rev[norm]!r} and {canonical!r}."
                )
            rev[norm] = canonical
    return rev


REVERSE_ALIASES: dict[str, str] = _build_reverse_alias_map()


# ─── Label normalization ─────────────────────────────────────────────────

_LABEL_TRUE_VALUES = frozenset({"1", "malicious", "attack", "attacker", "bad",
                                "true", "yes", "m", "a"})
_LABEL_FALSE_VALUES = frozenset({"0", "benign", "benign_normal", "normal", "good",
                                 "false", "no", "b", "n"})


def normalize_label(series: pd.Series) -> pd.Series:
    """Convert a raw label column into int8 0/1.

    Handles the two label encodings seen in CIC releases: numeric
    (0/1) and string ("Benign" / "Malicious", "BENIGN" / "MALICIOUS",
    "normal" / "attack"). Anything else raises.
    """
    def _coerce(v: Any) -> int:
        if isinstance(v, (int, np.integer)) and v in (0, 1):
            return int(v)
        if isinstance(v, float) and not np.isnan(v) and v in (0.0, 1.0):
            return int(v)
        s = str(v).strip().lower()
        if s in _LABEL_TRUE_VALUES:
            return 1
        if s in _LABEL_FALSE_VALUES:
            return 0
        raise ValueError(
            f"Unrecognized label value {v!r}. Expected one of "
            f"{sorted(_LABEL_TRUE_VALUES | _LABEL_FALSE_VALUES)}."
        )

    return series.map(_coerce).astype("int8")


# ─── Reading ──────────────────────────────────────────────────────────────

def read_cic_csvs(raw_dir: Path) -> pd.DataFrame:
    """Read every `.csv` in `raw_dir`, concatenate, and return.

    The CIC-DoH2023 bundle ships as multiple CSVs (one per day / capture
    session). This function reads them all, drops the landing page HTML
    if it was downloaded by mistake (File 10b saves it as
    `_landing_page.html`, which the glob does not match), and returns
    the concatenated DataFrame with a fresh RangeIndex.

    The forbidden-observable guard runs on the *concatenated* frame, so
    a forbidden column in any single CSV fails the whole batch.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"CIC-DoH2023 raw directory not found: {raw_dir}. "
            "Run `python data/download_datasets.py cic_doh2023` first, "
            "and place the CIC CSVs directly inside that directory."
        )
    csvs = sorted(p for p in raw_dir.glob("*.csv") if p.is_file())
    if not csvs:
        raise FileNotFoundError(
            f"No .csv files found in {raw_dir}. "
            "CIC-DoH2023 requires manual extraction after downloading "
            "the landing page (File 10b, --dry-run to preview). "
            "Place the CSVs directly in this directory and re-run."
        )

    parts: list[pd.DataFrame] = []
    for csv in csvs:
        log.info("Reading %s", csv)
        df = pd.read_csv(csv, low_memory=False)
        df["__source_csv"] = csv.name  # provenance, stripped before write
        parts.append(df)

    combined = pd.concat(parts, ignore_index=True)
    log.info(
        "Concatenated %d CSV(s) into %d rows × %d columns",
        len(parts), len(combined), len(combined.columns),
    )
    enforce_forbidden_observables(combined, raw_dir)
    return combined


# ─── Malformed-flow removal (§5.8.4 stage 1) ─────────────────────────────

def drop_malformed_flows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that cannot represent a valid flow.

    A row is malformed if any of the following hold:
      * `flow_duration_ms` is present and ≤ 0
      * `packet_count` is present and < 2 (a single-packet flow is not a
        flow)
      * every numeric feature is NaN (an entirely empty row)

    The checks are applied only to columns that exist in `df`, so this
    function is safe to run on the raw CIC frame before column mapping.
    Rows are dropped, not imputed; a malformed flow is a measurement
    error, not a missing feature.
    """
    before = len(df)
    mask = pd.Series(True, index=df.index)

    if "flow_duration" in df.columns:
        dur = pd.to_numeric(df["flow_duration"], errors="coerce")
        mask &= dur.fillna(0) > 0
    if "flow_duration_ms" in df.columns:
        dur = pd.to_numeric(df["flow_duration_ms"], errors="coerce")
        mask &= dur.fillna(0) > 0
    if "packet_count" in df.columns:
        pc = pd.to_numeric(df["packet_count"], errors="coerce")
        mask &= pc.fillna(0) >= 2

    # Drop rows where every numeric column is NaN.
    numeric_cols = [c for c in df.columns if c not in METADATA_COLUMNS
                    and c != "__source_csv"]
    if numeric_cols:
        all_nan = df[numeric_cols].isna().all(axis=1)
        mask &= ~all_nan

    out = df.loc[mask].reset_index(drop=True)
    dropped = before - len(out)
    if dropped:
        log.warning("Dropped %d malformed flow(s) of %d (%.2f%%)",
                    dropped, before, 100.0 * dropped / max(before, 1))
    else:
        log.info("No malformed flows detected.")
    return out


# ─── Column mapping ──────────────────────────────────────────────────────

def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw CIC columns to canonical 47-column names.

    For each raw column that matches an alias in `REVERSE_ALIASES`, the
    column is renamed to its canonical name. Columns that match no
    alias are dropped with a `log.debug` line so a reviewer can see the
    full column list of a given CIC release in the run log.

    The result is indexed by canonical name only; a canonical column
    with no matching raw column is left absent (it will be filled in
    `derive_missing_features`).
    """
    renames: dict[str, str] = {}
    dropped: list[str] = []
    for col in df.columns:
        if col == "__source_csv":
            continue
        norm = _normalize_column_name(col)
        canonical = REVERSE_ALIASES.get(norm)
        if canonical is None:
            dropped.append(col)
        else:
            renames[col] = canonical

    if dropped:
        log.debug("Dropping %d unmapped raw column(s): %s",
                  len(dropped), sorted(dropped))
    out = df.rename(columns=renames)
    return out


# ─── Feature derivation ──────────────────────────────────────────────────

def derive_missing_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Fill columns that could not be mapped from the CIC CSV.

    Two kinds of derivation are performed here:

    1. **Direct derivations from flow-level columns** (always possible):
       - `bytes_per_second` = (`bytes_out` + `bytes_in`) / (`flow_duration_ms` / 1000)
       - `packets_per_second` = `packet_count` / (`flow_duration_ms` / 1000)
       - `burstiness` = (σ_iat − μ_iat) / (σ_iat + μ_iat)
       - `packet_size_mean` = `bytes_out` / `packet_count` (fallback)
       - `packet_size_std` = 0.0 (fallback; CIC flow CSVs do not expose
         the per-packet size series, so the true std cannot be recovered
         from a flow-level record)
       These are computed only when the destination column is absent.

    2. **Sentinel fill** for columns that are not derivable from
       CIC's flow-level data:
       - `entropy_subdomain`, `entropy_query_type`
       - `qtype_A_count`, `qtype_AAAA_count`, `qtype_TXT_count`,
         `qtype_MX_count`, `qtype_CNAME_count`, `qtype_other_count`
       - `response_entropy`
       - `tls_version`, `cipher_suite_count`, `extension_count`,
         `session_reuse`
       - `autocorr_lag1`, `autocorr_lag5`, `spectral_low_band_energy`
       These columns are filled with `SENTINEL_NUMERIC` (for numeric
       columns) or the string `"unknown"` (for `tls_version`) and
       returned in the second tuple element so the caller can log
       them or fail if `--strict-schema` is set.

    Returns:
        (df_with_all_columns, list_of_sentinel_filled_column_names)
    """
    out = df.copy()
    filled_sentinels: list[str] = []

    # ── Derivations from flow-level columns ──────────────────────────
    dur_s = (pd.to_numeric(out["flow_duration_ms"], errors="coerce") / 1000.0
             if "flow_duration_ms" in out.columns else None)
    bytes_total = None
    if "bytes_out" in out.columns and "bytes_in" in out.columns:
        bytes_total = (pd.to_numeric(out["bytes_out"], errors="coerce").fillna(0)
                       + pd.to_numeric(out["bytes_in"], errors="coerce").fillna(0))

    if "bytes_per_second" not in out.columns and dur_s is not None and bytes_total is not None:
        out["bytes_per_second"] = (bytes_total / dur_s).replace([np.inf, -np.inf], np.nan)
        log.debug("Derived bytes_per_second from bytes_{out,in} / flow_duration_ms")

    if "packets_per_second" not in out.columns and dur_s is not None and "packet_count" in out.columns:
        out["packets_per_second"] = (
            pd.to_numeric(out["packet_count"], errors="coerce") / dur_s
        ).replace([np.inf, -np.inf], np.nan)
        log.debug("Derived packets_per_second from packet_count / flow_duration_ms")

    if "burstiness" not in out.columns and {"iat_mean_ms", "iat_std_ms"}.issubset(out.columns):
        mu = pd.to_numeric(out["iat_mean_ms"], errors="coerce")
        sd = pd.to_numeric(out["iat_std_ms"], errors="coerce")
        denom = sd + mu
        out["burstiness"] = ((sd - mu) / denom.where(denom != 0)).replace(
            [np.inf, -np.inf], np.nan
        )
        log.debug("Derived burstiness from iat_{mean,std}_ms")

    if "packet_size_mean" not in out.columns and bytes_total is not None and "packet_count" in out.columns:
        pc = pd.to_numeric(out["packet_count"], errors="coerce")
        out["packet_size_mean"] = (bytes_total / pc.where(pc != 0)).replace(
            [np.inf, -np.inf], np.nan
        )
        log.debug("Derived packet_size_mean from bytes_total / packet_count")

    if "packet_size_std" not in out.columns:
        # See docstring: the true std is not recoverable from a
        # flow-level record; fill with 0.0 and log.
        out["packet_size_std"] = 0.0
        log.info(
            "packet_size_std filled with 0.0 (true std not recoverable "
            "from CIC flow-level records)"
        )

    # ── Sentinel fill for non-derivable columns ──────────────────────
    for col in FEATURE_SCHEMA:
        if col in out.columns:
            continue
        if col == LABEL_COLUMN:
            raise ValueError(
                f"Label column {LABEL_COLUMN!r} missing from CIC CSV. "
                f"Found columns: {sorted(out.columns)}."
            )
        if col in ("protocol",):
            out[col] = "DoH"
        elif col == "capture_date":
            out[col] = pd.Timestamp.utcnow().normalize()
        elif col == "flow_id":
            out[col] = pd.Series(
                [f"cic_{i:08d}" for i in range(len(out))], index=out.index
            )
        elif col in NUMERIC_COLUMNS:
            out[col] = SENTINEL_NUMERIC
            filled_sentinels.append(col)
        else:
            out[col] = "unknown"
            filled_sentinels.append(col)

    return out, filled_sentinels


# ─── Normalization (§5.8.4 stage 4) ─────────────────────────────────────

def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score every numeric feature column (mean 0, std 1) across the batch.

    The normalization is applied to the concatenated CIC corpus, not per
    CSV, so that a day with unusually high traffic does not get its own
    mean/std. The label column is excluded (normalizing a 0/1 label
    would destroy its meaning); the metadata columns are excluded
    (their values are identifiers, not measurements); the sentinel
    value `SENTINEL_NUMERIC` is preserved as a distinct "missing"
    marker (it is not folded into the distribution).

    This stage corresponds to §5.8.4's "statistical descriptor
    extraction" + normalization stages. The splitting step
    (`split_data.py`, File 10e) reads the normalized CSV; the model
    never sees the raw scale.
    """
    out = df.copy()
    for col in NUMERIC_COLUMNS:
        if col not in out.columns or col == LABEL_COLUMN:
            continue
        series = pd.to_numeric(out[col], errors="coerce")
        # Do not renormalize sentinel values into the distribution.
        real = series[series != SENTINEL_NUMERIC]
        if len(real) < 2:
            log.warning(
                "Skipping normalization of %s: fewer than 2 non-sentinel "
                "values.", col,
            )
            continue
        mu = float(real.mean())
        sd = float(real.std(ddof=1))
        if sd == 0 or not np.isfinite(sd):
            log.warning(
                "Skipping normalization of %s: std = %s.", col, sd,
            )
            continue
        out[col] = (series - mu) / sd
        # Sentinel columns are all sentinel; the mask leaves them at
        # the sentinel value (-1.0 maps to (sentinel - mu) / sd, but
        # we overwrite sentinel rows back to the sentinel so the
        # downstream can still distinguish them.
        out.loc[series == SENTINEL_NUMERIC, col] = SENTINEL_NUMERIC
    return out


# ─── Writer ──────────────────────────────────────────────────────────────

def write_processed(df: pd.DataFrame, out_dir: Path, *, write_parquet: bool) -> Path:
    """Write the processed frame to `out_dir`.

    Writes:
      * `cic_doh2023.csv`      — human-readable; the artifact of record
      * `cic_doh2023.parquet`  — if `write_parquet`, for Dask consumption

    The CSV is written with the canonical 47-column order
    (`FEATURE_SCHEMA`) and `index=False`. The parquet file preserves
    dtypes for fast reload; the CSV is the diff-friendly artifact.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Strip the provenance column before writing.
    if "__source_csv" in df.columns:
        df = df.drop(columns="__source_csv")

    # Enforce the canonical column order.
    df = df[list(FEATURE_SCHEMA)]

    csv_path = out_dir / "cic_doh2023.csv"
    df.to_csv(csv_path, index=False)
    log.info("Wrote %s (%d rows × %d cols)", csv_path, len(df), df.shape[1])

    if write_parquet:
        try:
            pq_path = out_dir / "cic_doh2023.parquet"
            df.to_parquet(pq_path, index=False)
            log.info("Wrote %s", pq_path)
        except (ImportError, ValueError) as exc:
            log.warning(
                "Could not write parquet (%s). The CSV was written; "
                "downstream scripts read the CSV by default.", exc,
            )

    return csv_path


# ─── argparse ─────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="preprocess_doh2023.py",
        description=(
            "Preprocess the CIC-DoH2023 bundle into the artifact's "
            "canonical 47-column schema (data/data_utils.py::FEATURE_SCHEMA). "
            "Output is consumed by data/split_data.py (File 10e). "
            "CIC-DoH2023 is used for the 47-column schema-parity check, "
            "not as a training source (see File 9 header)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python data/preprocess_doh2023.py \\\n"
            "      --raw-dir data/raw/cic_doh2023 \\\n"
            "      --out-dir data/processed \\\n"
            "      --config configs/default_pafa_config.yaml\n"
            "  python data/preprocess_doh2023.py --strict-schema\n"
        ),
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the master config (default: auto-resolve to "
             "configs/default_pafa_config.yaml under the artifact root).",
    )
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Directory containing the CIC-DoH2023 CSVs "
             "(default: <artifact_root>/data/raw/cic_doh2023).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write the processed CSV / parquet "
             "(default: <artifact_root>/data/processed).",
    )
    p.add_argument(
        "--no-parquet",
        action="store_true",
        help="Do not write the parquet sidecar (CSV only). Useful when "
             "pyarrow is not installed.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files. Without this flag, an "
             "existing cic_doh2023.csv is a hard error.",
    )
    p.add_argument(
        "--strict-schema",
        action="store_true",
        help="Treat sentinel-filled columns as an error (a reviewer who "
             "wants to see exactly which columns are real vs. sentinel "
             "on a given CIC release). Without this flag, sentinel fills "
             "are a warning.",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return p


# ─── main ─────────────────────────────────────────────────────────────────

def _self_check(raw_dir: Path | None = None) -> int:
    """Non-destructive self-check used by pytest (File 21a).

    The check runs on a small synthetic CIC-shaped DataFrame (not on
    the real CIC corpus, which is 2.5 GB and not committed). It
    confirms:

      * `enforce_forbidden_observables` raises on a forbidden column.
      * `normalize_label` handles both numeric and string labels.
      * `map_columns` renames aliases correctly.
      * `derive_missing_features` fills every one of the 47 columns.
      * `validate_schema` passes on the derived frame.
      * `normalize_features` preserves the sentinel.

    `raw_dir` is accepted but unused; it exists so that a future
    version can optionally run against a small sample of the real
    corpus without changing the call site.
    """
    del raw_dir  # reserved for future use

    # 1. Forbidden-observable guard.
    bad = pd.DataFrame({"flow_id": ["f1"], "query_payload": ["secret"]})
    try:
        enforce_forbidden_observables(bad, Path("<synthetic>"))
    except ValueError as exc:
        if "query_payload" not in str(exc):
            print(f"FAIL: forbidden-guard message did not name the column: {exc}")
            return 1
        print("OK: forbidden-observable guard fires on query_payload.")
    else:
        print("FAIL: forbidden-observable guard did not fire.")
        return 1

    # 2. Label normalization, both encodings.
    for raw, expected in (
        (pd.Series([0, 1, 0, 1]), [0, 1, 0, 1]),
        (pd.Series(["Benign", "MALICIOUS", "normal", "attack"]), [0, 1, 0, 1]),
    ):
        got = normalize_label(raw).tolist()
        if got != expected:
            print(f"FAIL: normalize_label({raw.tolist()}) = {got}, expected {expected}")
            return 1
    print("OK: normalize_label handles numeric and string labels.")

    # 3. Column mapping + feature derivation on a synthetic CIC-shaped
    # frame. We deliberately include only a subset of columns so that the
    # missing ones exercise the derivation and sentinel-fill paths.
    raw = pd.DataFrame({
        "Flow ID": [f"f{i}" for i in range(4)],
        "Flow Duration": [1000.0, 2000.0, 1500.0, 800.0],
        "Total Packets": [10, 20, 15, 8],
        "Total Length of Fwd Packets": [1000, 2000, 1500, 800],
        "Total Length of Backward Packets": [500, 1000, 750, 400],
        "Flow IAT Mean": [100.0, 100.0, 100.0, 100.0],
        "Flow IAT Std": [10.0, 20.0, 15.0, 8.0],
        "Protocol": ["DoH", "DoH", "DoH", "DoH"],
        "Label": ["Benign", "Malicious", "Benign", "Malicious"],
    })
    mapped = map_columns(raw)
    if "flow_id" not in mapped.columns:
        print(f"FAIL: flow_id alias not mapped. Columns: {list(mapped.columns)}")
        return 1
    if "label" in mapped.columns:
        mapped["label"] = normalize_label(mapped["label"])
    derived, filled = derive_missing_features(mapped)
    validate_schema(derived)
    missing_after = [c for c in FEATURE_SCHEMA if c not in derived.columns]
    if missing_after:
        print(f"FAIL: missing columns after derivation: {missing_after}")
        return 1
    print(
        f"OK: mapped {len(mapped.columns)} raw column(s); "
        f"derived {len(FEATURE_SCHEMA) - len(mapped.columns)} column(s) "
        f"({len(filled)} via sentinel)."
    )

    # 4. Normalization preserves the sentinel.
    before_sentinel = derived.loc[0, "entropy_subdomain"]
    normalized = normalize_features(derived)
    after_sentinel = normalized.loc[0, "entropy_subdomain"]
    if before_sentinel != SENTINEL_NUMERIC or after_sentinel != SENTINEL_NUMERIC:
        print(
            f"FAIL: sentinel not preserved through normalization "
            f"(before={before_sentinel}, after={after_sentinel})."
        )
        return 1
    print("OK: sentinel preserved through normalization.")

    print("self-check ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Program entry point. Returns 0 on success, 1 on any failure."""
    args = build_parser().parse_args(argv)

    # Logging
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

    # Resolve paths (CLI overrides > config defaults > root-relative).
    cfg: dict[str, Any] = {}
    if args.config is None:
        args.config = root / "configs" / "default_pafa_config.yaml"
    try:
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    except (FileNotFoundError, yaml.YAMLError) as exc:
        log.error("Failed to load config %s: %s", args.config, exc)
        return 1
    raw_dir = Path(args.raw_dir) if args.raw_dir else root / cfg["paths"]["raw_datasets"] / "cic_doh2023"
    out_dir = Path(args.out_dir) if args.out_dir else root / cfg["paths"]["processed_datasets"]

    log.info("Raw directory:     %s", raw_dir)
    log.info("Output directory:  %s", out_dir)

    # Refuse to overwrite without --force.
    out_csv = out_dir / "cic_doh2023.csv"
    out_pq = out_dir / "cic_doh2023.parquet"
    if out_csv.exists() and not args.force:
        log.error(
            "%s already exists. Pass --force to overwrite, or remove the "
            "file manually.", out_csv,
        )
        return 1
    if out_pq.exists() and not args.no_parquet and not args.force:
        log.error(
            "%s already exists. Pass --force to overwrite, or pass "
            "--no-parquet to skip the parquet sidecar.", out_pq,
        )
        return 1

    # Stage 1: read + forbidden-observable guard
    try:
        raw = read_cic_csvs(raw_dir)
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 1

    # Stage 2: drop malformed flows
    cleaned = drop_malformed_flows(raw)
    if len(cleaned) == 0:
        log.error("No valid flows after malformed-flow removal. Aborting.")
        return 1

    # Stage 3: map columns + normalize labels
    mapped = map_columns(cleaned)
    if "label" in mapped.columns:
        try:
            mapped["label"] = normalize_label(mapped["label"])
        except ValueError as exc:
            log.error("Label normalization failed: %s", exc)
            return 1

    # Stage 4: derive missing features
    derived, filled = derive_missing_features(mapped)
    if filled and args.strict_schema:
        log.error(
            "Strict-schema mode: %d column(s) would be filled with a "
            "sentinel because CIC does not expose them: %s. "
            "Re-run without --strict-schema to accept the fill, or use "
            "the Zenodo DoH corpus (File 10d) which carries these "
            "fields natively.", len(filled), sorted(filled),
        )
        return 1
    if filled:
        log.warning(
            "%d column(s) filled with sentinel (not derivable from CIC "
            "flow-level data): %s. This is expected for the CIC-DoH2023 "
            "schema-parity check; the Zenodo DoH corpus (File 10d) "
            "carries these fields natively.", len(filled), sorted(filled),
        )

    # Stage 5: normalize
    normalized = normalize_features(derived)

    # Validate before writing.
    try:
        validate_schema(normalized)
    except ValueError as exc:
        log.error("Final schema validation failed: %s", exc)
        return 1

    # Write.
    try:
        write_processed(normalized, out_dir, write_parquet=not args.no_parquet)
    except (OSError, ValueError) as exc:
        log.error("Failed to write output: %s", exc)
        return 1

    # NEXT STEP banner.
    print()
    print("NEXT STEP:")
    print(
        "  python data/split_data.py "
        "--config configs/default_pafa_config.yaml"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
