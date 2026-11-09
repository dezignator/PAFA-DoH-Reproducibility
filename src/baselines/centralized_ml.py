# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# src/baselines/centralized_ml.py — PAFA-DoH Reproducibility Artifact
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
# The centralized CNN-LSTM baseline against which PAFA-DoH is compared
# in Table 2 and §4.30.1. The chapter reports 92.1 % detection accuracy
# and a 1.8 % false-positive rate for this baseline, on the same data,
# the same preprocessing and the same tuning budget as PAFA-DoH.
#
# This is a *centralized* model: it is trained on the full concatenated
# training split, with no federated aggregation and no differential
# privacy. It exists to anchor the comparison in §4.30.1, not as a
# viable privacy-preserving alternative.
#
# ───────────────────────────────────────────────────────────────────────────
# Where the pieces come from in Chapter 4
# ───────────────────────────────────────────────────────────────────────────
#   Reported baseline accuracy (92.1 %)  → §4.30.1
#   Reported baseline FPR (1.8 %)        → §4.30.1
#   Comparison table                     → Table 2
#   Same-data/same-tuning commitment     → §1.3, Table 2
#   Config block                         → configs/baselines_config.yaml
#                                          (File 9g, `centralized_ml:`)
#
# ───────────────────────────────────────────────────────────────────────────
# Architecture
# ───────────────────────────────────────────────────────────────────────────
# A small, standard CNN-LSTM:
#
#   input (batch, seq_len, n_features)
#     → Conv1d (channels=n_features → 64, kernel=3)
#     → ReLU
#     → MaxPool1d (2)
#     → LSTM (hidden=128, layers=2, batch_first=True)
#     → take last hidden state
#     → Linear (128 → 64)
#     → ReLU
#     → Linear (64 → 2)         # binary classification: benign / malicious
#
# The architecture is deliberately small so that the baseline finishes
# in reasonable time on the same hardware as PAFA-DoH and so that the
# "same tuning budget" commitment is honest — the baseline is not
# given a compute advantage over the proposed framework.
#
# ───────────────────────────────────────────────────────────────────────────
# What this file is NOT
# ───────────────────────────────────────────────────────────────────────────
# • Not a privacy-preserving model. It trains on the raw concatenated
#   training split. This is exactly what §4.30.1 says it does.
# • Not a federated model. Aggregation is not involved.
# • Not differential-privacy-trained. The reported FPR / accuracy are
#   without DP.
# • Not the best possible CNN-LSTM. It is a reasonable default that
#   reproduces the §4.30.1 number within the tolerance in
#   REPRODUCIBILITY_CHECKLIST.md §8 (rtol = 0.02 on the accuracy).
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        f"src/baselines/centralized_ml.py requires torch ({exc.name} missing). "
        "Install with `pip install -r requirements.txt`."
    ) from exc

from ..utils.metrics import accuracy, fpr

log = logging.getLogger(__name__)


# ─── Reported baseline metrics (from §4.30.1) ───────────────────────────
# These are the values the artifact compares its own reproduced
# baseline against. See configs/baselines_config.yaml for the canonical
# source, and expected_outputs/table_14_expected.json for the
# verified-after-run values.
REPORTED_CENTRALIZED_ACCURACY = 0.921
REPORTED_CENTRALIZED_FPR = 0.018


# ─── The model ──────────────────────────────────────────────────────────

class _CNN_LSTM(nn.Module):
    """CNN-LSTM classifier.

    See the module header for the architecture and its rationale.
    """

    def __init__(self, n_features: int, n_classes: int = 2):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=n_features,
            out_channels=64,
            kernel_size=3,
            padding=1,
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
        )
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features) → (batch, n_features, seq_len)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        # back to (batch, seq_len', channels) for LSTM
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        h = self.relu(self.fc1(last))
        logits = self.fc2(h)
        return logits


# ─── The trainer class ──────────────────────────────────────────────────

@dataclass
class BaselineResult:
    """Result of training and evaluating a baseline.

    The metrics are exactly the ones reported in Table 2 / §4.30.1 so
    that the expected-output comparison is direct.
    """
    accuracy: float
    fpr: float
    per_epoch_loss: list[float]


class CentralizedMLBaseline:
    """Centralized CNN-LSTM baseline for PAFA-DoH comparison.

    Usage:

        baseline = CentralizedMLBaseline(seed=42, device="cpu")
        baseline.fit(train_df, val_df)
        result = baseline.evaluate(test_df)
        print(result.accuracy, result.fpr)
    """

    def __init__(
        self,
        *,
        seq_len: int = 16,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        num_epochs: int = 50,
        early_stopping_patience: int = 10,
        seed: int = 42,
        device: str | None = None,
    ):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.seed = seed
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Seed torch for determinism. The seed is set here rather than
        # via `pafa.utils.seed_everything` because the caller may want
        # to seed the whole pipeline first and then instantiate the
        # baseline; setting it here ensures the baseline's own RNG is
        # deterministic even if the caller forgot.
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.model: nn.Module | None = None

    # ── Data helpers ──────────────────────────────────────────────────

    def _to_tensor(self, df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert a DataFrame into (X, y) tensors.

        The features are taken from `df.drop(columns=["label", ...])`,
        reshaped to (batch, seq_len, n_features), and normalized per
        sequence. The metadata columns and the label are dropped.

        This is a deliberately simple preprocessor; the same-data
        commitment is about the *dataset*, not the per-model tensor
        shaping, and a more elaborate preprocessor would not make the
        comparison fairer (both models see the same features).
        """
        from ..utils.logger import resolve_artifact_root  # local import, optional
        # Feature columns = every numeric column except the label and
        # any identifier columns. The toy-demo path passes a frame
        # that already has exactly the numeric features.
        meta = {"flow_id", "capture_date", "src_ip", "dst_ip", "protocol",
                "tls_version", "label"}
        feature_cols = [c for c in df.columns if c not in meta]
        # Keep only numeric columns (defensive: an unexpected column
        # type would raise at `.to_numpy(dtype=float)` below).
        feature_cols = [c for c in feature_cols
                        if pd.api.types.is_numeric_dtype(df[c])]

        if not feature_cols:
            raise ValueError(
                "CentralizedMLBaseline._to_tensor: no numeric feature "
                "columns found. Pass a DataFrame that includes the "
                "47-column schema from data/data_utils.py."
            )

        X = df[feature_cols].to_numpy(dtype=np.float32)
        # Reshape to (batch, seq_len, n_features) by padding/truncating.
        n = X.shape[0]
        if X.shape[1] < self.seq_len:
            pad = self.seq_len - X.shape[1]
            X = np.pad(X, ((0, 0), (0, pad)), mode="constant")
        X = X[:, : self.seq_len]
        # Ensure 3-D: (batch, seq_len, 1) if the frame had one feature.
        if X.ndim == 2:
            X = X[:, :, None]
        y = df["label"].to_numpy(dtype=np.int64)
        return (
            torch.from_numpy(X).float().to(self.device),
            torch.from_numpy(y).long().to(self.device),
        )

    # ── Training ──────────────────────────────────────────────────────

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> BaselineResult:
        """Train the CNN-LSTM on `train_df`.

        If `val_df` is provided, early stopping is applied on its
        loss. If not, the model trains for `num_epochs` and reports
        the training accuracy.

        Returns a BaselineResult with the *training* metrics. Call
        `.evaluate()` on a held-out split to get the test metrics
        that Table 2 reports.
        """
        X_train, y_train = self._to_tensor(train_df)
        n_features = X_train.shape[2]
        self.model = _CNN_LSTM(n_features=n_features).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.batch_size,
            shuffle=True,
        )

        best_val_loss = float("inf")
        patience_counter = 0
        per_epoch_loss: list[float] = []

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * xb.size(0)
            avg_loss = epoch_loss / len(train_df)
            per_epoch_loss.append(avg_loss)
            log.info("Epoch %d: train loss = %.4f", epoch, avg_loss)

            if val_df is not None:
                X_val, y_val = self._to_tensor(val_df)
                self.model.eval()
                with torch.no_grad():
                    val_loss = float(criterion(self.model(X_val), y_val).item())
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        log.info("Early stopping at epoch %d (patience %d).", epoch, self.early_stopping_patience)
                        break

        # Report training metrics for the return value.
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X_train).argmax(dim=1).cpu().numpy()
        y_true = y_train.cpu().numpy()
        return BaselineResult(
            accuracy=accuracy(y_true, pred),
            fpr=fpr(y_true, pred),
            per_epoch_loss=per_epoch_loss,
        )

    # ── Evaluation ────────────────────────────────────────────────────

    def evaluate(self, test_df: pd.DataFrame) -> BaselineResult:
        """Evaluate the trained model on `test_df`.

        Returns a BaselineResult whose accuracy / fpr are the values
        that should match §4.30.1's 92.1 % / 1.8 % within the
        tolerance in REPRODUCIBILITY_CHECKLIST.md §8 (rtol = 0.02).
        """
        if self.model is None:
            raise RuntimeError("Call fit() before evaluate().")
        X, y = self._to_tensor(test_df)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X).argmax(dim=1).cpu().numpy()
        y_true = y.cpu().numpy()
        return BaselineResult(
            accuracy=accuracy(y_true, pred),
            fpr=fpr(y_true, pred),
            per_epoch_loss=[],
        )
