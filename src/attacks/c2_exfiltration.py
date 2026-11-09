# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# src/attacks/c2_exfiltration.py — PAFA-DoH Reproducibility Artifact
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
# Generate command-and-control (C2) beaconing traffic with data
# exfiltration, as described in §4.26.4. The defining characteristic
# is *periodicity*: the beacon sends a query at a very regular
# interval (e.g., every 60 s), with payload data hidden in the query.
#
# Characteristics simulated:
#
#   * iat_std_ms     very low (near-periodic, unlike human traffic)
#   * autocorr_lag5  high (periodic signal)
#   * qtype_TXT_count elevated (C2 often uses TXT for payload)
#   * bytes_out      moderately high (small exfiltration payloads
#                     per beacon)
#   * regularity     > 0.9 (the §4.26.4 rule's threshold)
#
# A periodic C2 beacon is also a topological signature: it produces a
# persistent 1-dimensional homology class (Betti-1) in the traffic
# graph, which is what PAFA-DoH's TDA component detects (§4.28.2,
# Table 17). The generator's high autocorr_lag5 and low iat_std create
# exactly this signature.
#
# ───────────────────────────────────────────────────────────────────────────
# Where the pieces come from in Chapter 4
# ───────────────────────────────────────────────────────────────────────────
#   C2 beaconing simulation              → §4.26.4
#   Periodicity rule (> 0.9)             → §4.26.4, File 13c
#   Betti-1 topological signature        → §4.28.2, Table 17
#   Reported detection rate (98.7 %)     → Table 14
# ───────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE.parent.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from data.data_utils import FEATURE_SCHEMA, LABEL_COLUMN, canonicalize_dtypes
except ImportError:  # pragma: no cover
    from pafa.data.data_utils import (  # type: ignore[no-redef]
        FEATURE_SCHEMA, LABEL_COLUMN, canonicalize_dtypes,
    )

log = logging.getLogger(__name__)


class C2ExfiltrationAttack:
    """Generator for C2 beaconing + exfiltration traffic.

    Args:
        seed: RNG seed (default 42).
        beacon_interval_ms: Mean inter-beacon delay in ms
            (default 60000 = 60 s, a typical C2 sleep time).
        beacon_jitter_ms: Standard deviation of the inter-beacon delay
            (default 500 ms — 0.83 % of the mean, which produces the
            regularity > 0.9 that §4.26.4 requires).
    """

    def __init__(
        self,
        seed: int = 42,
        beacon_interval_ms: float = 60_000.0,
        beacon_jitter_ms: float = 500.0,
    ):
        self.seed = seed
        self.beacon_interval_ms = beacon_interval_ms
        self.beacon_jitter_ms = beacon_jitter_ms
        self.rng = np.random.default_rng(seed)

    def _metadata(self, n: int) -> dict[str, np.ndarray]:
        dates = ["2024-01-01", "2024-01-02", "2024-01-03",
                 "2024-01-04", "2024-01-05"]
        # C2 beacons rotate through resolvers to evade IP blocking
        # (§4.26.4: "multi-resolver beaconing").
        resolvers = ["8.8.8.8", "1.1.1.1", "9.9.9.9", "208.67.222.222"]
        return {
            "flow_id": np.array([f"c2_{i:06d}" for i in range(n)], dtype=object),
            "capture_date": self.rng.choice(dates, n).astype(object),
            "src_ip": np.array(
                [f"198.51.100.{self.rng.integers(1, 255)}" for _ in range(n)],
                dtype=object,
            ),
            "dst_ip": np.array(
                [resolvers[self.rng.integers(0, len(resolvers))] for _ in range(n)],
                dtype=object,
            ),
            "src_port": self.rng.integers(1024, 65535, n).astype(np.int64),
            "dst_port": np.full(n, 443, dtype=np.int64),
            "protocol": np.array(["DoH"] * n, dtype=object),
        }

    def _entropy_features(self, n: int) -> dict[str, np.ndarray]:
        """C2 payloads are high-entropy; query types favor TXT."""
        return {
            "entropy_subdomain": np.clip(
                self.rng.normal(4.5, 0.3, n), 3.5, 5.5,
            ).astype(np.float64),
            "entropy_query_type": np.clip(
                self.rng.normal(1.8, 0.3, n), 1.0, 3.0,
            ).astype(np.float64),
            "qtype_A_count": self.rng.poisson(1, n).astype(np.float64),
            "qtype_AAAA_count": self.rng.poisson(0.5, n).astype(np.float64),
            "qtype_TXT_count": self.rng.poisson(2, n).astype(np.float64),
            "qtype_MX_count": self.rng.poisson(0.2, n).astype(np.float64),
            "qtype_CNAME_count": self.rng.poisson(0.5, n).astype(np.float64),
            "qtype_other_count": self.rng.poisson(0.1, n).astype(np.float64),
        }

    def _timing_features(self, n: int) -> dict[str, np.ndarray]:
        """Near-periodic timing (the beaconing signature).

        The mean is `beacon_interval_ms` and the std is `beacon_jitter_ms`.
        This produces `regularity = 1 - std/mean = 1 - 500/60000 ≈ 0.992`,
        above the 0.9 threshold that §4.26.4 cites.
        """
        # Per-flow inter-query timing. Because the attack is one beacon,
        # the "flow" is a long-lived connection with periodic queries
        # at exactly the beacon interval.
        return {
            "iat_mean_ms": np.full(
                n, self.beacon_interval_ms, dtype=np.float64,
            ),
            "iat_std_ms": np.clip(
                self.rng.normal(self.beacon_jitter_ms, 50.0, n),
                10.0, 1000.0,
            ).astype(np.float64),
            "iat_min_ms": (self.beacon_interval_ms * 0.95).astype(np.float64),
            "iat_max_ms": (self.beacon_interval_ms * 1.05).astype(np.float64),
            "iat_median_ms": np.full(
                n, self.beacon_interval_ms, dtype=np.float64,
            ),
            "iat_skew": self.rng.normal(0.0, 0.1, n).astype(np.float64),
            "iat_kurtosis": self.rng.normal(0.0, 0.2, n).astype(np.float64),
        }

    def _flow_features(self, n: int) -> dict[str, np.ndarray]:
        """Small periodic flows with outbound-heavy payload."""
        bytes_out = self.rng.lognormal(6.5, 0.3, n)  # ≈ 665 B
        bytes_in = self.rng.lognormal(5.5, 0.3, n)   # ≈ 250 B
        packet_count = np.full(n, 4, dtype=np.int64)  # 1 query + 1 response
                                                       # + 2 keep-alives
        duration = self.beacon_interval_ms  # one beacon period
        packet_size_mean = (bytes_out + bytes_in) / packet_count
        return {
            "flow_duration_ms": np.full(n, duration, dtype=np.float64),
            "packet_count": packet_count,
            "bytes_out": bytes_out.astype(np.float64),
            "bytes_in": bytes_in.astype(np.float64),
            "packet_size_mean": packet_size_mean.astype(np.float64),
            "packet_size_std": (packet_size_mean * 0.15).astype(np.float64),
            "packet_size_max": (packet_size_mean * 1.5).astype(np.int64),
            "packet_size_min": (packet_size_mean * 0.5).astype(np.int64),
            "bytes_per_second": ((bytes_out + bytes_in) / (duration / 1000.0)).astype(np.float64),
            "packets_per_second": (packet_count / (duration / 1000.0)).astype(np.float64),
        }

    def _response_features(self, n: int) -> dict[str, np.ndarray]:
        resp_mean = self.rng.lognormal(5.0, 0.3, n)  # ≈ 150 B
        return {
            "response_size_mean": resp_mean.astype(np.float64),
            "response_size_std": (resp_mean * 0.2).astype(np.float64),
            "response_entropy": np.clip(
                self.rng.normal(2.5, 0.3, n), 1.5, 3.5,
            ).astype(np.float64),
        }

    def _tls_http_features(self, n: int) -> dict[str, np.ndarray]:
        return {
            "tls_version": np.array(["1.3"] * n, dtype=object),
            "cipher_suite_count": self.rng.poisson(8, n).astype(np.float64),
            "extension_count": self.rng.poisson(5, n).astype(np.float64),
            "session_reuse": self.rng.binomial(1, 0.9, n).astype(np.float64),
            "http2_stream_count": np.full(n, 1, dtype=np.float64),
            "http2_frame_count": np.full(n, 2, dtype=np.float64),
            "http3_frame_count": np.zeros(n, dtype=np.float64),
        }

    def _temporal_features(self, n: int) -> dict[str, np.ndarray]:
        """High autocorrelation (periodic signal)."""
        return {
            "autocorr_lag1": np.clip(
                self.rng.normal(0.95, 0.02, n), 0.85, 1.0,
            ).astype(np.float64),
            "autocorr_lag5": np.clip(
                self.rng.normal(0.9, 0.03, n), 0.8, 1.0,
            ).astype(np.float64),
            "burstiness": np.clip(
                self.rng.normal(-0.8, 0.05, n), -1.0, 1.0,
            ).astype(np.float64),
            "spectral_low_band_energy": np.clip(
                self.rng.normal(0.9, 0.02, n), 0.8, 1.0,
            ).astype(np.float64),
        }

    def generate(self, n: int) -> pd.DataFrame:
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
        for col in FEATURE_SCHEMA:
            if col not in df.columns:
                df[col] = -1.0
        df = df[list(FEATURE_SCHEMA)]
        df = canonicalize_dtypes(df)
        log.info("Generated %d C2 exfiltration rows.", n)
        return df
