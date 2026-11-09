# ═══════════════════════════════════════════════════════════════════════════
# Dockerfile — PAFA-DoH Reproducibility Artifact
# ═══════════════════════════════════════════════════════════════════════════
#
# Companion to:
#   README.md                       (File 1 of 74)
#   REPRODUCIBILITY_CHECKLIST.md    (File 2 of 74)
#   LICENSE                         (File 3 of 74)
#   requirements.txt                (File 4 of 74)
#   environment.yml                 (File 5 of 74)
#
# Thesis chapter: Chapter 4 (PAFA-DoH)
# Citation DOI:   10.1016/j.neunet.2025.108343   (Neural Networks, April 2026)
# License:        MIT (this file, as part of the artifact's source code)
# Artifact version: 1.0.0
#
# ───────────────────────────────────────────────────────────────────────────
# What this image is
# ───────────────────────────────────────────────────────────────────────────
# A CPU-only, multi-stage image that reproduces the PAFA-DoH artifact
# (Tables 13–17, 17a and Figures 7–10 of Chapter 4) on any host with Docker
# 20.10+ and no GPU, no Intel Loihi 2, no NVIDIA A100 and no IBM TrueNorth.
#
# The image is deliberately *not* a serving image. It is a batch image
# that runs the toy demo on `docker run` and exits, so that a reviewer can
# confirm the artifact works in one command. The full-mode reproduction
# (Section 5 of README.md) is launched interactively, mounting data/raw,
# data/processed and outputs as volumes.
#
# ───────────────────────────────────────────────────────────────────────────
# Build and run (copy-paste ready)
# ───────────────────────────────────────────────────────────────────────────
# Build:
#
#   docker build -t pafa-doh:1.0.0 .
#
# Run toy demo (≈ 6 min, CPU only, no downloads):
#
#   docker run --rm pafa-doh:1.0.0
#
# Interactive shell:
#
#   docker run --rm -it pafa-doh:1.0.0 bash
#
# Full-mode reproduction (mount data and outputs; the Zenodo corpus is
# downloaded inside the container into the mounted data/raw volume, so
# it persists across container restarts and does not inflate the image):
#
#   docker run --rm -it \
#       -v "$(pwd)/data/raw":/app/data/raw \
#       -v "$(pwd)/data/processed":/app/data/processed \
#       -v "$(pwd)/outputs":/app/outputs \
#       pafa-doh:1.0.0 \
#       bash -c "python data/download_datasets.py --config configs/default_pafa_config.yaml \
#             && python data/preprocess_zenodo.py --config configs/default_pafa_config.yaml \
#             && python data/preprocess_doh2023.py --config configs/default_pafa_config.yaml \
#             && python data/split_data.py --config configs/default_pafa_config.yaml \
#             && python scripts/train_pafa.py --config configs/default_pafa_config.yaml --mode full \
#             && python expected_outputs/verification_script.py"
#
# See README.md §5 for the full step-by-step command list.
#
# ───────────────────────────────────────────────────────────────────────────
# Why multi-stage
# ───────────────────────────────────────────────────────────────────────────
# Stage 1 (`builder`) installs the Python toolchain and compiles or
# downloads the pinned packages into a dedicated virtual environment at
# /opt/venv. Stage 2 (`runtime`) copies only /opt/venv and the small set of
# shared libraries the artifact needs at run time. This keeps the runtime
# image free of build-essential, gcc, g++, cmake, python3-dev, and
# libopenblas-dev (each of which would add hundreds of MB) while still
# supporting platforms where gudhi or ripser must be built from source
# (all major platforms have pre-built wheels for cp311 as of the pins in
# requirements.txt, but the builder stage is written to handle the case
# where they do not).
#
# ───────────────────────────────────────────────────────────────────────────
# Why the CPU wheel of torch is installed first
# ───────────────────────────────────────────────────────────────────────────
# On Linux x86_64, the default PyPI wheel for `torch==2.3.1` is the
# CUDA 12.1 bundle (≈ 2.2 GB compressed / ≈ 3.5 GB unpacked). It works
# on a CPU-only host but it makes the image needlessly large and it
# downloads a CUDA runtime that the artifact never uses. The CPU wheel
# is published on the official PyTorch CPU index at
# `https://download.pytorch.org/whl/cpu` and is an order of magnitude
# smaller.
#
# The builder stage therefore installs `torch==2.3.1+cpu`,
# `torchvision==0.18.1+cpu`, `torchaudio==2.3.1+cpu` from the CPU index
# *before* it runs `pip install -r requirements.txt`. The requirements
# file pins `torch==2.3.1` (without the `+cpu` local-version suffix).
# Per PEP 440, `2.3.1+cpu` satisfies `==2.3.1` (the local version label is
# ignored when the specifier is a public version), so the subsequent
# `pip install -r requirements.txt` sees torch as already satisfied and
# does not re-download the CUDA wheel.
#
# ───────────────────────────────────────────────────────────────────────────
# Proprietary hardware — what this image does NOT contain
# ───────────────────────────────────────────────────────────────────────────
# The following are NOT installed in this image, and must not be
# installed by anyone reproducing Chapter 4 on this image:
#
#   • Intel Loihi 2 (Lava, NxSDK)        — proprietary Intel hardware
#   • NVIDIA A100 (driver, CUDA toolkit)— proprietary NVIDIA hardware
#   • IBM TrueNorth (Compass, Eedn)      — proprietary IBM hardware
#   • ChipWhisperer trace-capture stack  — proprietary NewAE hardware
#
# The artifact ships calibrated software simulators in
# `src/hardware_simulation/` (loihi_mock.py, a100_emulation.py) that
# reproduce the reported latency and energy trends of Loihi 2 and A100
# within the tolerances in REPRODUCIBILITY_CHECKLIST.md §5. The
# simulators are pure Python and run on the same CPU as the rest of the
# artifact. IBM TrueNorth was not used in any reported experiment in
# Chapter 4 (§4.24.1) and is explicitly not simulated; see
# `src/hardware_simulation/README_hardware.md` for the non-use statement.
#
# ───────────────────────────────────────────────────────────────────────────
# Dataset — what this image does NOT contain
# ───────────────────────────────────────────────────────────────────────────
# The image does NOT bake in the Zenodo DoH corpus (DOI
# 10.5281/zenodo.5957677, ≈ 32 GB) or CIC-DoH2023 (≈ 2.5 GB). Both are
# CC-BY 4.0 datasets downloaded at run time by `data/download_datasets.py`.
# Baking them in would (a) make the image ~ 33 GB, (b) republish the
# datasets inside an MIT-licensed image, which the LICENSE scope note
# explicitly forbids (see LICENSE §2), and (c) prevent the reviewer from
# pinning a different snapshot of the corpus to compare against.
#
# The toy demo (the default CMD) does not download anything. It generates
# a 1,000 + 1,000 row synthetic dataset in `data/toy_dataset/` from the
# committed header-only schema files, which is CC-BY 4.0 and does not
# redistribute any Zenodo row.
#
# ───────────────────────────────────────────────────────────────────────────
# CPU-only by design; how to swap in CUDA
# ───────────────────────────────────────────────────────────────────────────
# Both stages use `python:3.11.9-slim-bookworm`. To build a CUDA-enabled
# variant for an A100 host, edit the Dockerfile as follows:
#
#   1. Replace both `FROM python:3.11.9-slim-bookworm AS <stage>` lines
#      with a CUDA base image that also provides Python 3.11, for example
#      `FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS <stage>` plus an
#      `apt-get install python3.11 python3.11-venv` step, or a
#      `python:3.11.9-slim-bookworm` base with a separate `--gpus all`
#      run-time mount of the CUDA runtime.
#
#   2. In the builder stage, drop the `--extra-index-url
#      https://download.pytorch.org/whl/cpu` from the CPU torch install
#      and let pip fetch `torch==2.3.1 torchvision==0.18.1
#      torchaudio==2.3.1` from PyPI (which is the CUDA 12.1 bundle on
#      Linux x86_64).
#
#   3. Run the resulting image with `docker run --gpus all ...`.
#
# This is intentionally NOT the default. The artifact is validated to
# reproduce every reported number on CPU using the simulators, and the
# CUDA swap is only needed if you want wall-clock timings on real A100
# hardware. See REPRODUCIBILITY_CHECKLIST.md §5 for the tolerances.
#
# ───────────────────────────────────────────────────────────────────────────
# Why tini as PID 1
# ───────────────────────────────────────────────────────────────────────────
# The default `CMD` runs a bash pipeline that forks python. Without an
# init process as PID 1, docker sends SIGTERM only to PID 1 (bash), which
# does not forward it to its children, and `docker stop` waits 10 s
# before SIGKILL. `tini` forwards signals to the process group and reaps
# zombies, so `docker stop` returns in < 1 s and long-running full-mode
# training runs exit cleanly.
#
# If you run the image with `docker run --init`, docker provides its own
# tiny init; tini will then be the *second* init and the signal path is
# still correct (tini execs into bash and forwards signals). Using
# `--init` is therefore harmless but redundant.
#
# ───────────────────────────────────────────────────────────────────────────
# Recommended .dockerignore
# ───────────────────────────────────────────────────────────────────────────
# A `.dockerignore` is not part of the 74-file plan but is strongly
# recommended to keep build context small and to prevent local outputs
# from leaking into the image. Mirror `.gitignore` (File 8 of 74) and
# add:
#
#   .git/
#   .venv/
#   .idea/
#   .vscode/
#   __pycache__/
#   *.pyc
#   outputs/
#   data/raw/
#   data/processed/
#   data/toy_dataset/*.csv   # committed header-only files stay in git,
#                            # but any locally generated rows should not
#                            # enter the build context
#   .env
#   *.log
#
# ═══════════════════════════════════════════════════════════════════════════


# ─── Stage 1: builder ───────────────────────────────────────────────────────
# Installs the Python toolchain, creates a virtual environment at
# /opt/venv, and installs every pinned package into that environment.
# This stage is discarded from the final image.

FROM python:3.11.9-slim-bookworm AS builder

# Determinism and hygiene for all pip / python invocations inside this stage.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONHASHSEED=42

# Build toolchain. Installed only in the builder stage; the runtime
# stage below does not have gcc / g++ / cmake / python3-dev.
#   • build-essential, gcc, g++, make  — for any source-built wheel
#   • cmake                           — gudhi source build (fallback only)
#   • python3-dev                     — Python.h for Cython (ripser fallback)
#   • libopenblas-dev                 — BLAS/LAPACK headers for source builds
#   • ca-certificates                 — TLS roots for the pip indexes
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        make \
        cmake \
        python3-dev \
        libopenblas-dev \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Dedicated venv. The runtime stage copies this directory verbatim, so
# the interpreter and all site-packages installed here are the ones that
# run in the final image.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Copy only the requirements file first so that the (expensive) install
# layer is cached across source edits. Any change to src/, configs/,
# scripts/, data/, expected_outputs/ or tests/ will invalidate only the
# COPY layers below, not the pip install layer.
COPY requirements.txt /tmp/requirements.txt

# Step 1 — CPU torch. See the header comment "Why the CPU wheel of
# torch is installed first" for the full rationale. The `+cpu` local
# version ensures that the subsequent `-r requirements.txt` step sees
# torch as already satisfied and does not pull the CUDA bundle.
RUN pip install --no-cache-dir --upgrade pip wheel setuptools \
 && pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        "torch==2.3.1+cpu" \
        "torchvision==0.18.1+cpu" \
        "torchaudio==2.3.1+cpu"

# Step 2 — everything else. The `torch==2.3.1` lines in requirements.txt
# match the already-installed `2.3.1+cpu` via PEP 440 local-version
# matching, so this step does not re-download torch.
RUN pip install --no-cache-dir -r /tmp/requirements.txt


# ─── Stage 2: runtime ───────────────────────────────────────────────────────
# The final image. Contains the /opt/venv built above plus the runtime
# system libraries only.

FROM python:3.11.9-slim-bookworm AS runtime

# Same hygiene as the builder stage. `MPLBACKEND=Agg` forces matplotlib
# to the non-interactive Agg backend so that Figures 8–10 render on a
# headless host without X11 / Qt.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONHASHSEED=42 \
    MPLBACKEND=Agg \
    PATH="/opt/venv/bin:${PATH}"

# Runtime system dependencies. Kept deliberately minimal:
#   • libgomp1       — OpenMP runtime required by ripser, gudhi, and the
#                      OpenBLAS shared library that ships inside the
#                      numpy / scipy manylinux wheels
#   • graphviz       — the `dot` binary required by Figure 7 (the Python
#                      `graphviz` bindings are already in requirements.txt;
#                      this package provides the executable)
#   • tini           — PID 1 init; see header comment "Why tini as PID 1"
#   • ca-certificates — TLS roots for the runtime download in full mode
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        graphviz \
        tini \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment built in stage 1. This is the only way
# torch, numpy, scipy, gudhi, ripser, and the rest reach the runtime
# image; none of them are reinstalled here.
COPY --from=builder /opt/venv /opt/venv

# Non-root user. UID/GID 1000 is the conventional first-interactive-user
# range on Linux and reduces friction when mounting host directories as
# volumes (host files created by the container will be owned by UID 1000
# inside the container, which maps to the host user in most single-user
# setups).
RUN groupadd --system --gid 1000 pafa \
 && useradd  --system --uid 1000 --gid pafa \
             --create-home --shell /bin/bash pafa

WORKDIR /app

# Copy only what the artifact needs at runtime. `.dockerignore` (see the
# header comment) is the recommended companion to keep the build context
# small and to prevent local outputs from leaking into the image.
COPY --chown=pafa:pafa src/              /app/src/
COPY --chown=pafa:pafa configs/          /app/configs/
COPY --chown=pafa:pafa scripts/          /app/scripts/
COPY --chown=pafa:pafa data/             /app/data/
COPY --chown=pafa:pafa expected_outputs/ /app/expected_outputs/
COPY --chown=pafa:pafa tests/            /app/tests/
COPY --chown=pafa:pafa \
        README.md \
        REPRODUCIBILITY_CHECKLIST.md \
        LICENSE \
        requirements.txt \
        environment.yml \
        setup.py \
        /app/

# Pre-create the output tree and give it to the non-root user. The
# VOLUME declaration below turns these into externally mountable
# directories; if the host does not mount anything, they are still
# writable by `pafa` inside the container.
RUN mkdir -p /app/outputs/tables \
             /app/outputs/figures \
             /app/outputs/logs \
             /app/outputs/checkpoints \
 && chown -R pafa:pafa /app/outputs

# Declare the three externally-owned directories. This is a declaration
# only: docker does not populate them, and the container runs fine
# without any `-v` mount (the built-in /app/data/raw and
# /app/data/processed are empty `.gitkeep` placeholders; the toy demo
# does not need them).
VOLUME ["/app/data/raw", "/app/data/processed", "/app/outputs"]

# Lightweight import-sanity probe. This is a build-time sanity check,
# not a service health probe: the image is a batch image, not a server.
# The probe confirms that every pinned package can be imported under the
# runtime system libraries. If any import fails, the container is marked
# unhealthy and `docker inspect --format='{{.State.Health.Status}}'`
# reports it. Override with `--no-healthcheck` if you do not want the
# probe.
HEALTHCHECK --interval=60s --timeout=15s --start-period=10s --retries=3 \
    CMD python -c "import numpy, scipy, pandas, torch, gudhi, ripser, persim, snntorch, networkx, Crypto, py_ecc, matplotlib, seaborn, graphviz, dask, yaml, pytest; print('pafa-doh imports ok')"

# Non-root user for everything below (ENTRYPOINT, CMD, and anything the
# container runs on `docker run <image> <command>`).
USER pafa

# OCI metadata. `revision` and `created` are intentionally omitted so
# that they can be injected at build time with `--label` and are not
# baked into the source.
LABEL org.opencontainers.image.title="PAFA-DoH Reproducibility Artifact" \
      org.opencontainers.image.description="Reproducibility artifact for Chapter 4 (PAFA-DoH) of the thesis: Next-Generation Secure Encrypted DNS Architectures. CPU-only, simulators for Loihi 2 and A100, toy demo runs on first docker run." \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/<your-org>/PAFA-DoH-Reproducibility" \
      org.opencontainers.image.documentation="https://github.com/<your-org>/PAFA-DoH-Reproducibility/blob/main/README.md"

# tini as PID 1 so that `docker stop` forwards SIGTERM to the python
# process and reaps zombies. See the header comment.
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default action: run the toy demo end-to-end and exit. A reviewer who
# runs `docker run --rm pafa-doh:1.0.0` with no arguments will see the
# whole pipeline (toy dataset → toy training → verification) complete
# in ≈ 6 min on a modern laptop CPU, with `verification_script.py`
# exiting 0 on success.
#
# The `-c` (not `-lc`) flag is used so that the environment inherited
# from the Dockerfile's ENV (PATH, MPLBACKEND, PYTHONHASHSEED, etc.) is
# the one that runs. A login shell (`-l`) would source /etc/profile and
# ~/.bash_profile, which can override PATH and reintroduce the system
# python.
CMD ["bash", "-c", "\
set -e; \
echo '[1/3] generating toy dataset'; \
python data/make_toy_dataset.py --mode synthetic --seed 42; \
echo '[2/3] running toy pipeline'; \
python scripts/train_pafa.py --config configs/default_pafa_config.yaml --mode toy; \
echo '[3/3] verifying toy outputs'; \
python expected_outputs/verification_script.py --toy-mode; \
echo 'toy demo complete: exit 0'; \
"]


# ═══════════════════════════════════════════════════════════════════════════
# End of Dockerfile
#
# Image size expectation (approximate, compressed pull size in parentheses):
#   builder stage (discarded)     ≈ 4.5 GB
#   runtime stage (final image)   ≈ 1.6 GB  (≈ 650 MB compressed)
#
# The size is dominated by the CPU wheel of torch (≈ 200 MB), torchvision
# (≈ 7 MB), numpy / scipy (≈ 90 MB combined), gudhi (≈ 25 MB), ripser
# (≈ 1 MB), and the Python interpreter. Removing torch, gudhi or the SNN
# stack would make the image dramatically smaller but would break the
# artifact's ability to reproduce Tables 13–17.
#
# Cross-check after build:
#   docker run --rm pafa-doh:1.0.0 python -c "import sys; print(sys.version)"
#   docker run --rm pafa-doh:1.0.0 python expected_outputs/verification_script.py --toy-mode
#
# The second command is what CI should run. It must exit 0.
#
# See README.md §3 (toy demo), §5 (full reproduction) and §6 (hardware
# requirements). See REPRODUCIBILITY_CHECKLIST.md §2 (environment /
# platform matrix) and §5 (hardware dependency map) for the exact claims
# this image can and cannot reproduce.
# ═══════════════════════════════════════════════════════════════════════════
