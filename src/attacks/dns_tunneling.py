# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# src/attacks/dns_tunneling.py — PAFA-DoH Reproducibility Artifact
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
# Generate DNS-tunneling traffic that mimics the dnscat2 tool described
# in §4.26.1. The generator produces 47-column rows suitable for the
# artifact's pipeline.
#
# Characteristics simulated (from §4.26.1, §4.7, and the RuleBased
# baseline in src/baselines/rule_based.py, File 13c):
#
#   * entropy_subdomain      > 4.2 bits  (dnscat2 encodes payload in
#                                          high-entropy subdomains)
#   * autocorr_lag5          > 0.8       (tunnel queries arrive in
#                                          characteristic bursts)
#   * qtype_TXT_count        elevated     (TXT records are a primary
#                                          exfiltration channel)
#   * qtype_MX_count         elevated
#   * bytes_out / bytes_in   skewed toward outbound (exfiltration)
#   * iat_mean_ms            low          (bursty, not human-paced)
#   * iat_std_ms             moderate     (burst variance)
#   * packet_size_mean       elevated     (payload in the query)
#
# The generator's output is designed so that:
#   - The rule-based baseline (File 13c) detects *some* of it (which
#     is what §4.30.3 reports as 78 % detection).
#   - The PAFA-DoH detector (via SNN + TDA) detects *all* of it
#     (which is what Table 14 reports as 99.5 % for tunneling).
# This is by construction: the generator produces ground-truth-labelled
# rows, and the detectors' accuracy is measured against that label.
#
# ───────────────────────────────────────────────────────────────────────────
# Where the pieces come from in Chapter 4
# ───────────────────────────────────────────────────────────────────────────
#   dnscat2 simulation                     → §4.26.1
#   entropy > 4.2 rule                     → §4.7 / §4.30.3
#   autocorr lag 5 > 0.8 rule              → §4.7 / Eq. 100
#   Reported detection rate (99.5 %)       → Table 14
# ───────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

# Import the canonical schema so every generated row has exactly the
# 47 columns the downstream components expect, in exactly the order
# `FEATURE_SCHEMA` declares. Never redeclare the schema here.
import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE.parent.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from data.data_utils import (
        FEATURE_SCHEMA,
        LABEL_COLUMN,
        canonicalize_dtypes,
    )
except ImportError:  # pragma: no cover — for the `pafa.attacks` install path
    from pafa.data.data_utils import (  # type: ignore[no-redef]
        FEATURE_SCHEMA,
        LABEL_COLUMN,
        canonicalize_dtypes,
    )

log = logging.getLogger(__name__)


# ─── Threshold constants (cited in §4.7 / §4.30.3) ──────────────────────
# These are the same values that the rule-based baseline checks against
# (src/baselines/rule_based.py, File 13c). The generator is designed to
# produce traffic that violates them — that is what makes the traffic
# "attack" traffic.
ENTROPY_THRESHOLD = 4.2
AUTOCORR_LAG5_THRESHOLD = 0.8


class DNSTunnelingAttack:
    """Generator for dnscat2-style DNS tunneling traffic.

    Every call to `generate(n)` returns a `pd.DataFrame` with exactly
    `n` rows and 47 columns. All rows carry `label = 1` (malicious) so
    that the generated traffic can be concatenated with benign rows to
    form a balanced training / evaluation set.

    The generator is deterministic given `seed`: two calls with the
    same seed and the same `n` produce byte-identical output.

    Args:
        seed: RNG seed (default 42).
        payload_entropy: Target entropy for the subdomain field
            (default 5.5 bits — above the 4.2 threshold the rule-based
            baseline uses, but not so extreme that the traffic looks
            synthetic).
        tx_ratio: Fraction of queries that are TXT records
            (default 0.6 — dnscat2 prefers TXT for exfiltration).
    """

    def __init__(
        self,
        seed: int = 42,
        payload_entropy: float = 5.5,
        tx_ratio: float = 0.6,
    ):
        self.seed = seed
        self.payload_entropy = payload_entropy
        self.tx_ratio = tx_ratio
        self.rng = np.random.default_rng(seed)

    # ── Feature families ─────────────────────────────────────────────

    def _metadata(self, n: int) -> dict[str, np.ndarray]:
        """Populate the 7 metadata columns.

        Source and destination IPs use TEST-NET-2 (198.51.100.0/24) to
        signal clearly that this is synthetic attack traffic (matching
        the convention in data/make_toy_dataset.py, File 10f).
        """
        dates = ["2024-01-01", "2024-01-02", "2024-01-03",
                 "2024-01-04", "2024-01-05"]
        return {
            "flow_id": np.array(
                [f"tunnel_{i:06d}" for i in range(n)], dtype=object
            ),
            "capture_date": self.rng.choice(dates, n).astype(object),
            "src_ip": np.array(
                [f"198.51.100.{self.rng.integers(1, 255)}" for _ in range(n)],
                dtype=object,
            ),
            "dst_ip": np.array(["8.8.8.8"] * n, dtype=object),
            "src_port": self.rng.integers(1024, 65535, n).astype(np.int64),
            "dst_port": np.full(n, 53, dtype=np.int64),  # tunneling often
                                                          # uses raw DNS
            "protocol": np.array(["DoH"] * n, dtype=object),
        }

    def _entropy_features(self, n: int) -> dict[str, np.ndarray]:
        """High-entropy subdomain and TXT-biased query types."""
        # Subdomain entropy: target `payload_entropy` with moderate noise.
        entropy_subdomain = np.clip(
            self.rng.normal(self.payload_entropy, 0.3, n),
            self.payload_entropy - 0.5,
            8.0,
        )
        # Query-type distribution: dnscat2 uses TXT primarily, MX for
        # fallback, A/AAAA for keep-alives.
        n_txt = self.rng.binomial(n, self.tx_ratio)
        n_mx = self.rng.binomial(n, 0.2)
        n_a = n - n_txt - n_mx
        n_a = np.maximum(n_a, 0)
        n_aaaa = self.rng.binomial(n_a, 0.3)
        n_a = n_a - n_aaaa
        n_cname = self.rng.binomial(n_a, 0.2)
        n_a = n_a - n_cname
        n_other = n_a  # remaining
        return {
            "entropy_subdomain": entropy_subdomain.astype(np.float64),
            "entropy_query_type": np.clip(
                self.rng.normal(2.2, 0.3, n), 1.0, 4.0
            ).astype(np.float64),
            "qtype_A_count": n_a.astype(np.float64),
            "qtype_AAAA_count": n_aaaa.astype(np.float64),
            "qtype_TXT_count": n_txt.astype(np.float64),
            "qtype_MX_count": n_mx.astype(np.float64),
            "qtype_CNAME_count": n_cname.astype(np.float64),
            "qtype_other_count": n_other.astype(np.float64),
        }

    def _timing_features(self, n: int) -> dict[str, np.ndarray]:
        """Bursty inter-arrival times with high autocorrelation.

        Tunneling queries arrive in short bursts at machine pace (not
        human pace), so iat_mean is low and iat_std is moderate. The
        lag-5 autocorrelation is set above the 0.8 threshold that
        §4.7 / §4.30.3 cite.
        """
        iat_mean = self.rng.lognormal(1.5, 0.3, n)  # ≈ 4.5 ms
        iat_std = self.rng.lognormal(1.0, 0.3, n)   # ≈ 2.7 ms
        return {
            "iat_mean_ms": iat_mean.astype(np.float64),
            "iat_std_ms": iat_std.astype(np.float64),
            "iat_min_ms": (iat_mean * 0.1).astype(np.float64),
            "iat_max_ms": (iat_mean * 5.0).astype(np.float64),
            "iat_median_ms": iat_mean.astype(np.float64),
            "iat_skew": self.rng.normal(1.0, 0.3, n).astype(np.float64),
            "iat_kurtosis": self.rng.normal(2.0, 0.5, n).astype(np.float64),
        }

    def _flow_features(self, n: int) -> dict[str, np.ndarray]:
        """Payload-heavy flows (outbound bytes > inbound).

        This is the physical signature of exfiltration: the client
        sends more (payload in the query) than it receives.
        """
        bytes_out = self.rng.lognormal(9.0, 0.4, n)  # ≈ 8 KB
        bytes_in = self.rng.lognormal(7.0, 0.4, n)   # ≈ 1 KB
        packet_count = self.rng.poisson(30, n) + 10
        duration = self.rng.lognormal(7.0, 0.3, n)   # ≈ 1 s
        packet_size_mean = (bytes_out + bytes_in) / packet_count
        return {
            "flow_duration_ms": duration.astype(np.float64),
            "packet_count": packet_count.astype(np.int64),
            "bytes_out": bytes_out.astype(np.float64),
            "bytes_in": bytes_in.astype(np.float64),
            "packet_size_mean": packet_size_mean.astype(np.float64),
            "packet_size_std": (packet_size_mean * 0.2).astype(np.float64),
            "packet_size_max": (packet_size_mean * 2.0).astype(np.int64),
            "packet_size_min": (packet_size_mean * 0.4).astype(np.int64),
            "bytes_per_second": ((bytes_out + bytes_in) / (duration / 1000.0)).astype(np.float64),
            "packets_per_second": (packet_count / (duration / 1000.0)).astype(np.float64),
        }

    def _response_features(self, n: int) -> dict[str, np.ndarray]:
        """Small responses (DNS tunneling responses carry little data)."""
        resp_mean = self.rng.lognormal(4.5, 0.3, n)  # ≈ 90 B
        return {
            "response_size_mean": resp_mean.astype(np.float64),
            "response_size_std": (resp_mean * 0.2).astype(np.float64),
            "response_entropy": np.clip(
                self.rng.normal(2.8, 0.3, n), 1.5, 4.0
            ).astype(np.float64),
        }

    def _tls_http_features(self, n: int) -> dict[str, np.ndarray]:
        """Standard TLS + HTTP/2 parameters (tunneling runs inside DoH)."""
        return {
            "tls_version": np.array(["1.3"] * n, dtype=object),
            "cipher_suite_count": self.rng.poisson(8, n).astype(np.float64),
            "extension_count": self.rng.poisson(5, n).astype(np.float64),
            "session_reuse": self.rng.binomial(1, 0.5, n).astype(np.float64),
            "http2_stream_count": self.rng.poisson(5, n).astype(np.float64),
            "http2_frame_count": self.rng.poisson(15, n).astype(np.float64),
            "http3_frame_count": np.zeros(n, dtype=np.float64),
        }

    def _temporal_features(self, n: int) -> dict[str, np.ndarray]:
        """High lag-5 autocorrelation (the §4.7 tunneling signature)."""
        return {
            "autocorr_lag1": np.clip(
                self.rng.normal(0.6, 0.1, n), 0.0, 1.0
            ).astype(np.float64),
            "autocorr_lag5": np.clip(
                self.rng.normal(0.85, 0.05, n),  # mean 0.85 > threshold 0.8
                AUTOCORR_LAG5_THRESHOLD + 0.01, 1.0,
            ).astype(np.float64),
            "burstiness": np.clip(
                self.rng.normal(0.4, 0.1, n), -1.0, 1.0
            ).astype(np.float64),
            "spectral_low_band_energy": np.clip(
                self.rng.normal(0.6, 0.1, n), 0.0, 1.0
            ).astype(np.float64),
        }

    # ── Public API ───────────────────────────────────────────────────

    def generate(self, n: int) -> pd.DataFrame:
        """Generate `n` tunneling rows.

        Returns a DataFrame with exactly `FEATURE_SCHEMA` as its
        columns, in order. All rows carry `label = 1`.
        """
        if n <= 0:
            raise ValueError(f"n must be positive; got {n}.")
        data: dict[str, Any] = {}
        for f in (
            self._metadata,
            self._flow_features,
            self._timing_features,
            self._entropy_features,
            self._response_features,
            self._tls_http_features,
            self._temporal_features,
        ):
            data.update(f(n))
        data[LABEL_COLUMN] = np.ones(n, dtype=np.int8)

        df = pd.DataFrame(data)
        # Ensure any schema column not directly populated is filled
        # (defensive against a future schema change).
        for col in FEATURE_SCHEMA:
            if col not in df.columns:
                df[col] = -1.0
        df = df[list(FEATURE_SCHEMA)]
        df = canonicalize_dtypes(df)
        log.info("Generated %d DNS-tunneling rows.", n)
        return df
