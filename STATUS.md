<!--
STATUS.md — PAFA-DoH Reproducibility Artifact
═══════════════════════════════════════════════════════════════════════════

Companion to:
  README.md                       (File 1 of 74)
  REPRODUCIBILITY_CHECKLIST.md    (File 2 of 74)
  LICENSE                         (File 3 of 74)

Thesis chapter: Chapter 4 (PAFA-DoH)
Citation DOI:   10.1016/j.neunet.2025.108343   (Neural Networks, April 2026)
License:        MIT (this file, as part of the artifact's documentation)
Artifact version: 1.0.0
═══════════════════════════════════════════════════════════════════════════
-->

# Artifact Status — PAFA-DoH Reproducibility Package

**Version:** 1.0.0
**Date:** 2026-11-09
**Status:** Reference Implementation / Structural Scaffolding

---

## 1. Executive Summary

This repository is the reproducibility artifact for Chapter 4 (PAFA-DoH) of the thesis. It contains the complete code, configuration, data-layer logic, and mathematical foundations of the framework described in the chapter.

However, **this artifact does not, by itself, reproduce the headline numeric results reported in Chapter 4** (e.g., 99.4 % federated accuracy, 432 ms Groth16 proof generation). It is a **reference implementation** and a **structural scaffold**. Several key components — specifically the zero-knowledge proving system and the full federated training loop — are implemented as reference stubs or simulators rather than production systems. The toy pipeline (`--mode toy`) is designed to run in under two minutes, verify that the code is wired correctly, and pass the structural integrity tests, but it does **not** perform real training on the 32 GB corpus.

The goal of this document is to be transparent with reviewers about exactly what the artifact does and does not demonstrate, so that its contribution can be evaluated honestly.

---

## 2. Component-by-Component Status

The table below classifies every component of the artifact as either **Fully Implemented** (real code that does what it says), **Partial / Reference** (correct structure but simplified or simulated behavior), or **Stub / Placeholder** (not a real implementation of the underlying cryptographic or ML primitive).

| Component | File(s) | Status | What this actually means |
|---|---|---|---|
| Data utilities & schema | `data/data_utils.py` (10a) | ✅ Fully Implemented | 47-column schema, SHA-256, artifact-root resolution, deterministic splitting. |
| Dataset downloaders | `data/download_datasets.py` (10b) | ✅ Fully Implemented | Downloads Zenodo and CIC-DoH2023 by DOI/URL. |
| Preprocessors | `data/preprocess_*.py` (10c–10d) | ✅ Fully Implemented | Parses raw CSVs, applies the 47-column schema, filters malformed flows. |
| Toy dataset generator | `data/make_toy_dataset.py` (10f) | ✅ Fully Implemented | Generates 2,000 synthetic rows (1,000/1,000) matching the schema. |
| Mathematics (all equations) | `src/utils/math_utils.py` (11f) | ✅ Fully Implemented | Every equation from §4.2–4.31 is implemented and verified by unit tests. |
| SNN Detector | `src/models/snn_detector.py` (12c) | ✅ Fully Implemented | Real LIF neurons, STDP, and log-burst encoding. |
| TDA Detector | `src/models/tda_detector.py` (12d) | ✅ Fully Implemented | Real Vietoris-Rips filtration, Betti numbers, Wasserstein distance. |
| Ring-LWE Homomorphic Inference | `src/models/homomorphic_inference.py` (12f) | ✅ Fully Implemented | Real negacyclic polynomial multiplication, encryption, and decryption. |
| AI-Obfuscated Crypto | `src/models/ai_obfuscated_crypto.py` (12h) | ✅ Fully Implemented | Real SNR-guided noise injection. |
| Self-Healing Trust Fabric | `src/models/self_healing.py` (12g) | ✅ Fully Implemented | Real entropy-based scoring and blacklist logic. |
| Experimental Baselines | `src/baselines/*` (13b–13c) | ✅ Fully Implemented | Real CNN-LSTM (via torch) and rule-based detectors. |
| Attack Generators | `src/attacks/*` (14b–14e) | ✅ Fully Implemented | Real generators for DNS tunnelling, phishing, C2, side-channel. |
| Hardware Simulators | `src/hardware_simulation/*` (16b–16c) | ✅ Fully Implemented | Real calibrated simulators for Loihi 2 and A100 (within tolerance). |
| **Federated Learning Trainer** | `src/models/federated_learning.py` (12b) | ⚠️ **Partial / Reference** | The DP-SGD update and geometric median logic are real. However, `scripts/train_pafa.py` **simulates** the convergence curve rather than running 100 rounds of real gradient computation on the 32 GB corpus. |
| **Zero-Knowledge Prover (Groth16)** | `src/models/zk_attestation.py` (12e) | ⚠️ **Stub / Placeholder** | The `Groth16Attestation` class uses SHA-256 hashing of inputs to mimic a proof, rather than implementing the actual BN128 elliptic-curve pairing arithmetic required for a real zk-SNARK. |
| **Table Reproduction Scripts** | `scripts/reproduce_tables/*` (18a–18f) | ⚠️ **Partial / Reference** | The scripts correctly write their outputs to `outputs/tables/`. However, in toy mode, they write the *expected* thesis values directly (or compute them on the 2,000-row toy data), rather than deriving them from a trained model. |
| **Verification Script** | `expected_outputs/verification_script.py` (20i) | ✅ Fully Implemented | Genuinely compares fresh outputs against committed expected values. |

---

## 3. What the Toy Demo Actually Proves

When a reviewer runs the commands in `README.md §3` (the 2-minute toy demo), the artifact demonstrates that:

1. **The code is syntactically and structurally valid.** Every module imports, every class instantiates, every function call resolves.
2. **The schema is consistent.** The 47-column schema flows correctly from the data generator through the preprocessors, the models, and the expected outputs.
3. **The math is correct.** Every equation (Eqs. 36, 38, 40–46, 58, 60, etc.) produces the expected result on synthetic input, as verified by `pytest`.
4. **The verification logic is sound.** If the pipeline produces a value outside the tolerance, `verification_script.py` will fail.

**The toy demo does NOT demonstrate that:**
- The model achieves 99.4 % accuracy on real DoH traffic.
- The Groth16 implementation generates a valid cryptographic proof.
- The federated training loop converges in 8 epochs on the real dataset.
- The TDA detector's Betti-1 comparison holds on the real 32 GB corpus.

---

## 4. What a Reviewer Should Do

### If you are reviewing the **structural integrity** of the artifact:
Run the toy demo and `pytest tests/`. If both pass (exit 0), the artifact is correctly structured and the code is sound where it exists.

### If you are reviewing the **numeric claims** of Chapter 4:
This artifact alone is not sufficient. You would need:
- The 32 GB Zenodo corpus (`10.5281/zenodo.5957677`).
- A production Groth16 prover (which this artifact does not contain).
- A real federated training loop (which this artifact simulates).
- Access to Intel Loihi 2 and NVIDIA A100 hardware (which this artifact simulates).

The chapter itself states that the results were produced on proprietary hardware and a full 32 GB corpus. This artifact provides the scaffolding to *rebuild* that pipeline, but the rebuild is a multi-week project, not a `pip install` and run.

---

## 5. Path Forward (How to make this a full reproduction)

To transition this artifact from a "reference implementation" to a "full numeric reproduction," the following steps are required:

1. **Replace the Groth16 stub** with a real `py_ecc`-based proving system. This requires ~1,000 lines of pairing, QAP, and Fiat-Shamir implementation.
2. **Rewrite `scripts/train_pafa.py`** to run a real federated training loop: load the 32 GB corpus, instantiate a trainable model (e.g., the CNN-LSTM from `src/baselines/`), run 100 rounds of DP-SGD, and compute actual convergence.
3. **Fit the TDA detector** on real point clouds extracted from the Zenodo corpus, rather than on the 100-row toy sample.
4. **Run the full pipeline** (README §5) and compare the fresh `outputs/` against the committed `expected_outputs/` with strict tolerances.

---

## 6. Contact

If you are a reviewer and have questions about what is and is not implemented, please open a GitHub issue with the label `reviewer-question`. The maintainer will respond within 48 hours.

---

*End of `STATUS.md`.*
