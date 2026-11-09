# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# src/utils/seed.py — PAFA-DoH Reproducibility Artifact
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
# Pin every source of randomness the artifact can control. The commit
# in Table 16 (seed 42) is only meaningful if all five RNG streams are
# seeded deterministically:
#
#   random.seed(42)                     — Python stdlib
#   numpy.random.seed(42)               — legacy NumPy global
#   numpy.random.default_rng(42)        — explicit PCG64 (used by the
#                                          toy generator, File 10f)
#   torch.manual_seed(42)               — torch CPU + default device
#   torch.cuda.manual_seed_all(42)      — all CUDA devices (if any)
#   torch.backends.cudnn.deterministic  — deterministic cuDNN kernels
#   torch.backends.cudnn.benchmark      — disabled for determinism
#   os.environ["PYTHONHASHSEED"] = "42" — set/dict iteration order
#
# It also calls torch.use_deterministic_algorithms(True) when the
# accelerator supports it, so that a reviewer who runs on a GPU gets
# the same numerical path the expected outputs were produced with.
# This is a best-effort call: on some ops, some backends, torch will
# raise rather than silently give a nondeterministic result. The
# try/except below turns that into a logged warning, not a crash,
# because the artifact's default path is CPU and the CPU path is
# already deterministic.
#
# ───────────────────────────────────────────────────────────────────────────
# Where this is called
# ───────────────────────────────────────────────────────────────────────────
#   scripts/train_pafa.py            (File 17a)  — at startup
#   scripts/train_baselines.py        (File 17b)  — at startup
#   scripts/evaluate_models.py         (File 17c)  — at startup
#   scripts/reproduce_tables/*         (Files 18a–18f)
#   scripts/reproduce_figures/*        (Files 19a–19d)
#   data/make_toy_dataset.py            (File 10f) — at the top of main()
#   data/split_data.py                  (File 10e) — at the top of main()
#   tests/*                             (Files 21a–21d) — via pytest fixture
#
# Every entry point calls `seed_everything(config_seed)` before doing any
# work. The `seed` value is always read from
# `configs/default_pafa_config.yaml::seed`, never hard-coded at the
# call site (the one exception is `make_toy_dataset.py`, which accepts
# `--seed` and cross-checks it against the config).
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import logging
import os
import random

import numpy as np

log = logging.getLogger(__name__)


def seed_everything(seed: int = 42) -> int:
    """Seed every RNG stream the artifact uses.

    Args:
        seed: The seed to pin. Defaults to 42, which is the value
            committed in `configs/default_pafa_config.yaml::seed` and
            used to produce the expected outputs. Passing any other
            value is a deliberate override that will break the
            `verification_script.py` strict-tolerance check (File 20i).

    Returns:
        The seed, unchanged. Callers sometimes chain
        `seed_everything(seed_everything(42))`; returning the value
        makes that a no-op instead of a double-seed.

    Notes:
        The function is idempotent: calling it twice with the same seed
        resets every stream to the same state as the first call. This
        matters for scripts that re-seed between epochs.
    """
    # 1. Python stdlib
    random.seed(seed)

    # 2. NumPy legacy global
    np.random.seed(seed)

    # 3. Environment hash seed. This must be set *before* any dict is
    #    constructed with string keys that will be iterated in a
    #    hash-order-dependent way. In practice the artifact sets it
    #    at the very top of `main()`, before config parsing.
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 4. Torch (optional import — the toy demo runs without torch if
    #    the reviewer only wants the schema checks).
    try:
        import torch
    except ImportError:
        log.warning(
            "torch is not installed; skipped torch RNG seeding. "
            "Install with `pip install -r requirements.txt` for the "
            "full artifact. The schema checks and the data-layer "
            "self-checks run without torch."
        )
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 5. cuDNN determinism (CPU path is unaffected; this matters if
    #    the reviewer runs on a GPU).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 6. Strict deterministic algorithms (best effort). Some ops will
    #    raise on this; that is the correct behaviour (a
    #    nondeterministic result is worse than a loud failure). We
    #    only warn on unsupported backends rather than failing the
    #    whole pipeline, because the CPU path is already deterministic
    #    and the GPU path is opt-in.
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
    except (RuntimeError, AttributeError) as exc:
        log.warning(
            "torch.use_deterministic_algorithms not fully supported on "
            "this backend (%s). Numerical results may drift slightly "
            "from the committed expected outputs; the tolerance in "
            "REPRODUCIBILITY_CHECKLIST.md §8 was chosen to absorb this "
            "drift on the CPU path.", exc,
        )

    return seed


def make_generator(seed: int = 42, *, cuda: bool = False):
    """Return a fresh `torch.Generator` seeded with `seed`.

    Used by DataLoader `worker_init_fn` and by the DP-SGD noise sampler
    in `src/models/federated_learning.py` (File 12b), so that the noise
    stream is independent of the model-init stream.

    Args:
        seed: The seed for this generator.
        cuda: If True, return a CUDA generator; else a CPU one.

    Raises:
        ImportError: if `torch` is not installed. The function is
            intentionally not lazy — a caller that wants a generator
            is a caller that is about to use torch.
    """
    import torch
    g = torch.Generator(device="cuda" if cuda else "cpu")
    g.manual_seed(seed)
    return g
