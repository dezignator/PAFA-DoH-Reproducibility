#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# scripts/train_pafa_v2.py — PAFA-DoH Reproducibility Artifact
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
# This script actually trains PAFA-DoH. It uses PyTorch, partitions the
# dataset, instantiates FederatedClientV2, and runs the FL loop.
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pafa.utils.seed import seed_everything
from pafa.utils.logger import get_logger, resolve_artifact_root
from pafa.models.federated_learning_v2 import PAFAFederatedLearningV2, FederatedClientV2

log = logging.getLogger(__name__)

def prepare_data(df: pd.DataFrame):
    meta_cols = {"flow_id", "capture_date", "src_ip", "dst_ip", "protocol", "tls_version", "label"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int64)
    
    # Normalize features
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return torch.tensor(X), torch.tensor(y)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["toy", "full"], default="toy")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    root = resolve_artifact_root()
    
    data_path = root / "data" / "processed" / ("toy_dataset/combined.csv" if args.mode == "toy" else "zenodo_doh.csv")
    if not data_path.exists():
        log.error(f"Data not found: {data_path}")
        return 1
        
    df = pd.read_csv(data_path)
    X, y = prepare_data(df)
    log.info(f"Loaded {len(X)} samples with {X.shape[1]} features.")

    # Partition data into N clients
    num_clients = 10 if args.mode == "toy" else 100
    chunk_size = len(X) // num_clients
    clients = []
    
    for i in range(num_clients):
        start, end = i * chunk_size, (i + 1) * chunk_size if i < num_clients - 1 else len(X)
        clients.append(FederatedClientV2(i, X[start:end], y[start:end]))
        
    log.info(f"Initialized {num_clients} federated clients.")

    # Train
    fl = PAFAFederatedLearningV2(num_clients=num_clients)
    rounds = 5 if args.mode == "toy" else 100
    fl.fit(clients, rounds=rounds)

    # Save checkpoint
    ckpt_dir = root / "outputs" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(fl.global_model.state_dict(), ckpt_dir / "pafa_model_v2.pth")
    log.info(f"Saved real model checkpoint to {ckpt_dir / 'pafa_model_v2.pth'}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
