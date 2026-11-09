# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# src/attacks/side_channel.py — PAFA-DoH Reproducibility Artifact
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
# Generate side-channel attack traffic. The attack simulates the
# ChipWhisperer-style timing / power analysis described in §4.26.3:
# an adversary measures timing variance of DNS resolution to infer
# cryptographic key material.
#
# Characteristics simulated (from §4.26.1, §4.26.3, and the timing
# rule in src/baselines/rule_based.py, File 13c):
#
#   * iat_std_ms     large (the attack probes with varying delays)
#   * iat_mean_ms    moderate (not as fast as tunnel, not as slow as
#                     human browsing)
#   * timing variance > 15 ms² (the side-channel rule threshold)
#   * response_entropy moderate (responses are normal-length DNS,
#                                not tiny keep-alives)
#
# The §4.13.1 SNN-guided noise injection (File 12h) is the defense
# that reduces the SNR below the adversary's detection threshold —
# but the attack generator itself does not know about the defense;
# it just produces the attack traffic.
#
# ───────────────────────────────────────────────────────────────────────────
# Where the pieces come from in Chapter 4
# ───────────────────────────────────────────────────────────────────────────
#   ChipWhisperer side-channel simulation → §4.26.3
#   Timing variance threshold (> 15)     → §4.26.1, §4.13.1
#   Reported detection rate (100 %)      → Table 14
#   SNN-guided noise injection defense   → §4.13.1, Eq. 47
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


class SideChannelAttack:
    """Generator for ChipWhisperer-style side-channel attack traffic.

    The defining feature is high `iat_std_ms` (large timing variance),
    because the adversary is deliberately varying the inter-query
    delay to probe the cryptographic implementation's timing
    characteristics.

    Args:
        seed: RNG seed (default 42).
        target_iat_mean_ms: Mean inter-arrival time (default 150 ms —
            the value used in §4.26.3's `tc netem` command).
        target_iat_std_ms: Standard deviation (default 25 ms — the
            value from §4.26.3's `tc netem` command).
    """

    def __init__(
        self,
        seed: int = 42,
        target_iat_mean_ms: float = 150.0,
        target_iat_std_ms: float = 25.0,
    ):
        self.seed = seed
        self.target_iat_mean_ms = target_iat_mean_ms
        self.target_iat_std_ms = target_iat_std_ms
        self.rng = np.random.default_rng(seed)

    def _metadata(self, n: int) -> dict[str, np.ndarray]:
        dates = ["2024-01-01", "2024-01-02", "2024-01-03",
                 "2024-01-04", "2024-01-05"]
        return {
            "flow_id": np.array([f"sc_{i:06d}" for i in range(n)], dtype=object),
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
        return {
            "entropy_subdomain": np.clip(
                self.rng.normal(2.5, 0.3, n), 1.5, 4.0,
            ).astype(np.float64),
            "entropy_query_type": np.clip(
                self.rng.normal(1.5, 0.3, n), 0.5, 3.0,
            ).astype(np.float64),
            "qtype_A_count": self.rng.poisson(5, n).astype(np.float64),
            "qtype_AAAA_count": self.rng.poisson(2, n).astype(np.float64),
            "qtype_TXT_count": self.rng.poisson(1, n).astype(np.float64),
            "qtype_MX_count": self.rng.poisson(0, n).astype(np.float64),
            "qtype_CNAME_count": self.rng.poisson(1, n).astype(np.float64),
            "qtype_other_count": self.rng.poisson(0, n).astype(np.float64),
        }

    def _timing_features(self, n: int) -> dict[str, np.ndarray]:
        """High timing variance is the defining feature of this attack."""
        # Draw from a lognormal centered on the target mean, with the
        # target standard deviation. The lognormal's mode is shifted,
        # so we tune sigma to get std ≈ target_iat_std_ms.
        mu_log = np.log(self.target_iat_mean_ms ** 2 / np.sqrt(
            self.target_iat_mean_ms ** 2 + self.target_iat_std_ms ** 2
        ))
        sigma_log = np.sqrt(np.log(
            1 + (self.target_iat_std_ms / self.target_iat_mean_ms) ** 2
        ))
        iat_mean = self.rng.lognormal(mu_log, sigma_log, n)
        # Force the variance to be high (above the > 15 ms² rule).
        iat_std = np.clip(
            self.rng.normal(self.target_iat_std_ms, 2.0, n),
            np.sqrt(15.0) + 1.0,  # above the §4.26.1 threshold
            100.0,
        )
        return {
            "iat_mean_ms": iat_mean.astype(np.float64),
            "iat_std_ms": iat_std.astype(np.float64),
            "iat_min_ms": (iat_mean * 0.5).astype(np.float64),
            "iat_max_ms": (iat_mean + 3 * iat_std).astype(np.float64),
            "iat_median_ms": iat_mean.astype(np.float64),
            "iat_skew": self.rng.normal(0.5, 0.3, n).astype(np.float64),
            "iat_kurtosis": self.rng.normal(1.0, 0.5, n).astype(np.float64),
        }

    def _flow_features(self, n: int) -> dict[str, np.ndarray]:
        bytes_out = self.rng.lognormal(6.5, 0.4, n)  # ≈ 665 B
        bytes_in = self.rng.lognormal(6.8, 0.4, n)   # ≈ 900 B
        packet_count = self.rng.poisson(8, n) + 4
        duration = (self.target_iat_mean_ms * packet_count)  # total ≈ iat × count
        packet_size_mean = (bytes_out + bytes_in) / packet_count
        return {
            "flow_duration_ms": duration.astype(np.float64),
            "packet_count": packet_count.astype(np.int64),
            "bytes_out": bytes_out.astype(np.float64),
            "bytes_in": bytes_in.astype(np.float64),
            "packet_size_mean": packet_size_mean.astype(np.float64),
            "packet_size_std": (packet_size_mean * 0.3).astype(np.float64),
            "packet_size_max": (packet_size_mean * 2.5).astype(np.int64),
            "packet_size_min": (packet_size_mean * 0.3).astype(np.int64),
            "bytes_per_second": ((bytes_out + bytes_in) / (duration / 1000.0)).astype(np.float64),
            "packets_per_second": (packet_count / (duration / 1000.0)).astype(np.float64),
        }

    def _response_features(self, n: int) -> dict[str, np.ndarray]:
        resp_mean = self.rng.lognormal(5.5, 0.4, n)  # ≈ 250 B
        return {
            "response_size_mean": resp_mean.astype(np.float64),
            "response_size_std": (resp_mean * 0.3).astype(np.float64),
            "response_entropy": np.clip(
                self.rng.normal(2.0, 0.3, n), 1.0, 3.5,
            ).astype(np.float64),
        }

    def _tls_http_features(self, n: int) -> dict[str, np.ndarray]:
        return {
            "tls_version": np.array(["1.3"] * n, dtype=object),
            "cipher_suite_count": self.rng.poisson(8, n).astype(np.float64),
            "extension_count": self.rng.poisson(5, n).astype(np.float64),
            "session_reuse": self.rng.binomial(1, 0.5, n).astype(np.float64),
            "http2_stream_count": self.rng.poisson(2, n).astype(np.float64),
            "http2_frame_count": self.rng.poisson(5, n).astype(np.float64),
            "http3_frame_count": np.zeros(n, dtype=np.float64),
        }

    def _temporal_features(self, n: int) -> dict[str, np.ndarray]:
        return {
            "autocorr_lag1": np.clip(
                self.rng.normal(0.1, 0.1, n), 0.0, 0.5,
            ).astype(np.float64),
            "autocorr_lag5": np.clip(
                self.rng.normal(0.05, 0.05, n), 0.0, 0.2,
            ).astype(np.float64),
            "burstiness": np.clip(
                self.rng.normal(-0.5, 0.15, n), -1.0, 1.0,
            ).astype(np.float64),
            "spectral_low_band_energy": np.clip(
                self.rng.normal(0.15, 0.1, n), 0.0, 0.5,
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
        log.info("Generated %d side-channel attack rows.", n)
        return df
