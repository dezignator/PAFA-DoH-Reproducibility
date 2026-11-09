# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# src/attacks/zero_day_phishing.py — PAFA-DoH Reproducibility Artifact
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
# Generate zero-day phishing traffic: domains that are newly registered
# and not on any blocklist, so rule-based detection cannot catch them.
# The generator mimics the GAN-generated domains described in §4.26.2.
#
# Characteristics simulated (from §4.26.2 and the topological
# properties that PAFA-DoH uses to detect them — §4.7.1, §4.28.2):
#
#   * entropy_subdomain    moderately high but not maximal (GAN
#                          generates plausible-looking names, not random
#                          strings — the phishing domain must look
#                          legitimate to a human)
#   * autocorr_lag5        low (a phishing flow is usually short and
#                          non-repetitive)
#   * qtype distribution   uniform (phishing spreads its queries across
#                          A, AAAA, and CNAME)
#   * response_entropy     moderate-high (phishing responds with
#                          credentials pages, not tiny DNS responses)
#   * packet_count         moderate (phishing resolves a handful of
#                          subresources)
#
# The §4.7.1 TDA signature that distinguishes this attack from benign
# traffic is its *topological* properties (Betti-1 in the H_1
# persistence diagram), which the SNN and TDA detectors capture.
#
# ───────────────────────────────────────────────────────────────────────────
# Where the pieces come from in Chapter 4
# ───────────────────────────────────────────────────────────────────────────
#   GAN-generated phishing domains        → §4.26.2
#   Topological detection rationale      → §4.7.1, §4.28.2
#   Betti-1 / persistence entropy        → Table 17
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


class ZeroDayPhishingAttack:
    """Generator for GAN-style zero-day phishing traffic.

    Every call to `generate(n)` returns a `pd.DataFrame` with exactly
    `n` rows and 47 columns. All rows carry `label = 1`.

    Args:
        seed: RNG seed (default 42).
        domain_entropy: Target subdomain entropy (default 3.8 — higher
            than benign (~2.0) but below tunnel (~5.5), matching the
            "plausible but novel" characteristic of GAN-generated
            phishing domains).
    """

    def __init__(self, seed: int = 42, domain_entropy: float = 3.8):
        self.seed = seed
        self.domain_entropy = domain_entropy
        self.rng = np.random.default_rng(seed)

    def _metadata(self, n: int) -> dict[str, np.ndarray]:
        dates = ["2024-01-01", "2024-01-02", "2024-01-03",
                 "2024-01-04", "2024-01-05"]
        return {
            "flow_id": np.array([f"phish_{i:06d}" for i in range(n)], dtype=object),
            "capture_date": self.rng.choice(dates, n).astype(object),
            "src_ip": np.array(
                [f"198.51.100.{self.rng.integers(1, 255)}" for _ in range(n)],
                dtype=object,
            ),
            "dst_ip": np.array(["8.8.8.8"] * n, dtype=object),
            "src_port": self.rng.integers(1024, 65535, n).astype(np.int64),
            "dst_port": np.full(n, 443, dtype=np.int64),
            "protocol": np.array(["DoH"] * n, dtype=object),
        }

    def _entropy_features(self, n: int) -> dict[str, np.ndarray]:
        """Moderately high subdomain entropy, uniform query types."""
        entropy_subdomain = np.clip(
            self.rng.normal(self.domain_entropy, 0.4, n), 2.5, 4.5,
        )
        # Uniform query type distribution is the GAN signature.
        counts = self.rng.multinomial(n, [0.25, 0.25, 0.15, 0.10, 0.15, 0.10])
        a, aaaa, txt, mx, cname, other = counts.tolist()
        return {
            "entropy_subdomain": entropy_subdomain.astype(np.float64),
            "entropy_query_type": np.clip(
                self.rng.normal(2.0, 0.3, n), 1.0, 3.5,
            ).astype(np.float64),
            "qtype_A_count": np.full(n, a, dtype=np.float64),
            "qtype_AAAA_count": np.full(n, aaaa, dtype=np.float64),
            "qtype_TXT_count": np.full(n, txt, dtype=np.float64),
            "qtype_MX_count": np.full(n, mx, dtype=np.float64),
            "qtype_CNAME_count": np.full(n, cname, dtype=np.float64),
            "qtype_other_count": np.full(n, other, dtype=np.float64),
        }

    def _timing_features(self, n: int) -> dict[str, np.ndarray]:
        """Human-paced timing (phishing is user-driven, unlike tunnels)."""
        iat_mean = self.rng.lognormal(4.5, 0.5, n)  # ≈ 90 ms
        iat_std = self.rng.lognormal(3.5, 0.4, n)   # ≈ 33 ms
        return {
            "iat_mean_ms": iat_mean.astype(np.float64),
            "iat_std_ms": iat_std.astype(np.float64),
            "iat_min_ms": (iat_mean * 0.2).astype(np.float64),
            "iat_max_ms": (iat_mean * 4.0).astype(np.float64),
            "iat_median_ms": iat_mean.astype(np.float64),
            "iat_skew": self.rng.normal(0.5, 0.3, n).astype(np.float64),
            "iat_kurtosis": self.rng.normal(1.0, 0.5, n).astype(np.float64),
        }

    def _flow_features(self, n: int) -> dict[str, np.ndarray]:
        """Moderate flow size (phishing resolves a handful of subresources)."""
        bytes_out = self.rng.lognormal(7.5, 0.5, n)  # ≈ 1.8 KB
        bytes_in = self.rng.lognormal(8.0, 0.5, n)   # ≈ 3 KB
        packet_count = self.rng.poisson(15, n) + 5
        duration = self.rng.lognormal(7.5, 0.4, n)   # ≈ 1.8 s
        packet_size_mean = (bytes_out + bytes_in) / packet_count
        return {
            "flow_duration_ms": duration.astype(np.float64),
            "packet_count": packet_count.astype(np.int64),
            "bytes_out": bytes_out.astype(np.float64),
            "bytes_in": bytes_in.astype(np.float64),
            "packet_size_mean": packet_size_mean.astype(np.float64),
            "packet_size_std": (packet_size_mean * 0.25).astype(np.float64),
            "packet_size_max": (packet_size_mean * 2.0).astype(np.int64),
            "packet_size_min": (packet_size_mean * 0.3).astype(np.int64),
            "bytes_per_second": ((bytes_out + bytes_in) / (duration / 1000.0)).astype(np.float64),
            "packets_per_second": (packet_count / (duration / 1000.0)).astype(np.float64),
        }

    def _response_features(self, n: int) -> dict[str, np.ndarray]:
        """Larger responses (phishing serves credential pages)."""
        resp_mean = self.rng.lognormal(8.0, 0.5, n)  # ≈ 3 KB
        return {
            "response_size_mean": resp_mean.astype(np.float64),
            "response_size_std": (resp_mean * 0.3).astype(np.float64),
            "response_entropy": np.clip(
                self.rng.normal(3.2, 0.3, n), 2.0, 4.5,
            ).astype(np.float64),
        }

    def _tls_http_features(self, n: int) -> dict[str, np.ndarray]:
        return {
            "tls_version": np.array(["1.3"] * n, dtype=object),
            "cipher_suite_count": self.rng.poisson(6, n).astype(np.float64),
            "extension_count": self.rng.poisson(4, n).astype(np.float64),
            "session_reuse": self.rng.binomial(1, 0.2, n).astype(np.float64),
            "http2_stream_count": self.rng.poisson(3, n).astype(np.float64),
            "http2_frame_count": self.rng.poisson(8, n).astype(np.float64),
            "http3_frame_count": np.zeros(n, dtype=np.float64),
        }

    def _temporal_features(self, n: int) -> dict[str, np.ndarray]:
        """Low lag-5 autocorrelation (phishing is non-repetitive)."""
        return {
            "autocorr_lag1": np.clip(
                self.rng.normal(0.2, 0.15, n), 0.0, 1.0,
            ).astype(np.float64),
            "autocorr_lag5": np.clip(
                self.rng.normal(0.15, 0.1, n), 0.0, 0.5,
            ).astype(np.float64),
            "burstiness": np.clip(
                self.rng.normal(0.0, 0.15, n), -1.0, 1.0,
            ).astype(np.float64),
            "spectral_low_band_energy": np.clip(
                self.rng.normal(0.3, 0.1, n), 0.0, 1.0,
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
        log.info("Generated %d zero-day phishing rows.", n)
        return df
