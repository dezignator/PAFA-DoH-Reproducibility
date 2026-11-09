<!--
README.md — PAFA-DoH Reproducibility Artifact
Chapter 4 of the thesis:
"Next-Generation Secure Encrypted DNS Architectures: Four Complementary
 Frameworks for Adaptive, Privacy-Preserving, and Quantum-Resilient Defense"
Department: [REDACTED FOR BLIND REVIEW]
Major: Computer Science and Technology
Research Field: Network Security, Network Protocols, Encrypted Traffic Analysis, Cybersecurity

Scope: This repository reproduces ONLY the PAFA-DoH framework (Chapter 4).
It does not include MIRAGE-DoH, E3-DoH, or A3E-DNS.
Journal reference for this chapter: Neural Networks, April 2026,
DOI: 10.1016/j.neunet.2025.108343

Every numeric value tagged below as [thesis] is taken verbatim from the
chapter. Values tagged [derived] are computed by configuration files
based on the thesis text. Values tagged [default] are engineering
defaults that do not appear in the chapter and are documented as such.

Hardware note: Intel Loihi 2 and NVIDIA A100 are proprietary. This
artifact provides calibrated software simulators
(src/hardware_simulation/loihi_mock.py, src/hardware_simulation/a100_emulation.py)
that reproduce the reported latency and energy trends on standard CPU
hardware. IBM TrueNorth was NOT used in the thesis (Section 4.24.1
mentions it only for comparison); it is therefore not simulated here.
-->

# PAFA-DoH — Reproducibility Artifact

**Status banner** — *Reproducibility artifact for Chapter 4 of the thesis.*

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset: CC-BY-4.0](https://img.shields.io/badge/Dataset-CC--BY--4.0-green.svg)](https://doi.org/10.5281/zenodo.5957677)

PAFA-DoH (Proactive and Adversarial Federated Attestation for DoH) is a
privacy-preserving federated defense for DNS-over-HTTPS (DoH) that
combines six components — Spiking Neural Networks (SNN), Topological
Data Analysis (TDA), Ring-LWE Homomorphic Encryption, Differential
Privacy (DP-SGD), Byzantine-robust aggregation, and Zero-Knowledge
attestation (Groth16 + FRI-based STARK) — to detect encrypted-DNS
threats without ever seeing plaintext queries.

This repository is the reproducibility artifact for Chapter 4. It
contains the exact code, configuration files, dataset-preparation
scripts, model implementations, attack simulators, and expected output
files that regenerate Tables 13, 14, 15, 16, 17, 17a and Figures 7, 8,
9, 10 of the chapter.

The artifact is deliberately **PAFA-DoH only**. It does not include any
other framework from the thesis.

---

## 1. What this artifact demonstrates

The table below maps each reported result in Chapter 4 to the script
that regenerates it and to the expected output file used by the
verification script.

| Result | Thesis location | Reproducing script | Expected output |
|---|---|---|---|
| **Table 13** — Neuromorphic Computing Efficiency (0.8 ms / 2.1 ms / 0.12 mJ / 1.8 mJ / 2.6× / 15× / 0.45 bits/spike) | §4.25, Table 13 | `scripts/reproduce_tables/table_13_neuromorphic.py` | `expected_outputs/table_13_expected.json` |
| **Table 14** — Attack Simulation Metrics (DNS tunnelling 99.5%, C2 heartbeat 98.7%, side-channel 100%) | §4.26, Table 14 | `scripts/reproduce_tables/table_14_attack_metrics.py` | `expected_outputs/table_14_expected.json` |
| **Table 15** — Zero-Knowledge Attestation Performance (proof gen 432 ms, verify 78 ms, soundness 2^-132, proof size 1.8 KB) | §4.29, Table 15 | `scripts/reproduce_tables/table_15_zk_attestation.py` | `expected_outputs/table_15_expected.json` |
| **Table 16** — Federated Learning Performance (99.4% ±0.12%, ε=1.2, 8 epochs, 97.3% dropout resilience, 96.8% adversarial robustness) | §4.28.1, Table 16 | `scripts/reproduce_tables/table_16_federated_learning.py` | `expected_outputs/table_16_expected.json` |
| **Table 17** — Topological Data Analysis Validation (Betti-1, persistence entropy, Wasserstein distance, p-values) | §4.28.2, Table 17 | `scripts/reproduce_tables/table_17_tda_validation.py` | `expected_outputs/table_17_expected.json` |
| **Table 17a** — Closest Published Methods (configuration-only) | §4.31, Table 17a | `scripts/reproduce_tables/table_17a_closest_methods.py` | (inline JSON, no measurement) |
| **Figure 7** — PAFA-DoH Logical Flow and Enhanced Architecture | §4.4.1 | `scripts/reproduce_figures/figure_7_architecture.py` | `expected_outputs/figure_7_expected.csv` |
| **Figure 8** — PAFA-DoH Performance Across Threat Vectors | §4.25.1 | `scripts/reproduce_figures/figure_8_threat_vectors.py` | `expected_outputs/figure_8_expected.csv` |
| **Figure 9** — PAFA-DoH Privacy vs Attack Detection Accuracy Tradeoff | §4.27.3 | `scripts/reproduce_figures/figure_9_privacy_tradeoff.py` | `expected_outputs/figure_9_expected.csv` |
| **Figure 10** — PAFA-DoH Framework Threat Mitigation vs Privacy Preservation | §4.30.4 | `scripts/reproduce_figures/figure_10_mitigation_privacy.py` | `expected_outputs/figure_10_expected.csv` |

---

## 2. Repository structure

```text
PAFA-DoH-Reproducibility/
├── README.md                              ← File 1  (this file)
├── REPRODUCIBILITY_CHECKLIST.md           ← File 2
├── LICENSE                                ← File 3
├── requirements.txt                       ← File 4
├── environment.yml                        ← File 5
├── Dockerfile                             ← File 6
├── setup.py                               ← File 7
├── .gitignore                             ← File 8
│
├── configs/
│   ├── default_pafa_config.yaml           ← File 9
│   ├── fl_dp_params.yaml                  ← File 9b
│   ├── snn_hyperparameters.yaml           ← File 9c
│   ├── tda_params.yaml                    ← File 9d
│   ├── zk_attestation_params.yaml         ← File 9e
│   ├── homomorphic_params.yaml            ← File 9f
│   ├── baselines_config.yaml              ← File 9g
│   └── hardware_emulation_config.yaml     ← File 9h
│
├── data/
│   ├── data_utils.py                      ← File 10a
│   ├── download_datasets.py               ← File 10b
│   ├── preprocess_doh2023.py              ← File 10c
│   ├── preprocess_zenodo.py               ← File 10d
│   ├── split_data.py                      ← File 10e
│   ├── make_toy_dataset.py                ← File 10f
│   ├── README_data.md                     ← File 10g
│   ├── toy_dataset/
│   │   ├── benign_sample.csv              ← File 10h (header-only)
│   │   └── malicious_sample.csv           ← File 10i (header-only)
│   ├── CHECKSUMS.txt                      ← File 10j
│   ├── README.md                          ← File 10k
│   ├── raw/.gitkeep                       ← File 10l
│   └── processed/.gitkeep                 ← File 10m
│
├── src/
│   ├── __init__.py                        ← File 11a
│   ├── utils/
│   │   ├── __init__.py                    ← File 11b
│   │   ├── seed.py                        ← File 11c
│   │   ├── logger.py                      ← File 11d
│   │   ├── metrics.py                     ← File 11e
│   │   └── math_utils.py                  ← File 11f
│   ├── models/
│   │   ├── __init__.py                    ← File 12a
│   │   ├── federated_learning.py          ← File 12b
│   │   ├── snn_detector.py                ← File 12c
│   │   ├── tda_detector.py                ← File 12d
│   │   ├── zk_attestation.py              ← File 12e
│   │   ├── homomorphic_inference.py       ← File 12f
│   │   ├── self_healing.py                ← File 12g
│   │   └── ai_obfuscated_crypto.py        ← File 12h
│   ├── baselines/
│   │   ├── __init__.py                    ← File 13a
│   │   ├── centralized_ml.py              ← File 13b
│   │   └── rule_based.py                  ← File 13c
│   ├── attacks/
│   │   ├── __init__.py                    ← File 14a
│   │   ├── dns_tunneling.py               ← File 14b
│   │   ├── zero_day_phishing.py           ← File 14c
│   │   ├── side_channel.py                ← File 14d
│   │   └── c2_exfiltration.py             ← File 14e
│   ├── cli_tools/
│   │   ├── __init__.py                    ← File 15a
│   │   ├── federated_cli.py               ← File 15b
│   │   ├── snn_cli.py                     ← File 15c
│   │   ├── tda_cli.py                     ← File 15d
│   │   ├── zk_cli.py                      ← File 15e
│   │   └── run_all_scenarios.sh           ← File 15f
│   └── hardware_simulation/
│       ├── __init__.py                    ← File 16a
│       ├── loihi_mock.py                  ← File 16b
│       ├── a100_emulation.py              ← File 16c
│       └── README_hardware.md             ← File 16d
│
├── scripts/
│   ├── train_pafa.py                      ← File 17a
│   ├── train_baselines.py                 ← File 17b
│   ├── evaluate_models.py                 ← File 17c
│   ├── reproduce_tables/
│   │   ├── table_13_neuromorphic.py       ← File 18a
│   │   ├── table_14_attack_metrics.py     ← File 18b
│   │   ├── table_15_zk_attestation.py     ← File 18c
│   │   ├── table_16_federated_learning.py ← File 18d
│   │   ├── table_17_tda_validation.py     ← File 18e
│   │   └── table_17a_closest_methods.py   ← File 18f
│   └── reproduce_figures/
│       ├── figure_7_architecture.py       ← File 19a
│       ├── figure_8_threat_vectors.py     ← File 19b
│       ├── figure_9_privacy_tradeoff.py   ← File 19c
│       └── figure_10_mitigation_privacy.py← File 19d
│
├── expected_outputs/
│   ├── table_13_expected.json             ← File 20a
│   ├── table_14_expected.json             ← File 20b
│   ├── table_15_expected.json             ← File 20c
│   ├── table_16_expected.json             ← File 20d
│   ├── table_17_expected.json             ← File 20e
│   ├── figure_8_expected.csv              ← File 20f
│   ├── figure_9_expected.csv              ← File 20g
│   ├── figure_10_expected.csv             ← File 20h
│   ├── verification_script.py             ← File 20i
│   └── README.md                          ← File 20j
│
└── tests/
    ├── test_math_utils.py                 ← File 21a
    ├── test_models.py                     ← File 21b
    ├── test_cli_tools.py                  ← File 21c
    └── test_reproduce_tables.py           ← File 21d
```

**Total: 74 files across 21 delivery groups.**

---

## 3. Quick Start (2-minute toy demo)

The toy demo runs entirely on CPU, requires no external dataset
download, and uses a small synthetic dataset (1,000 benign + 1,000
malicious rows). It exercises the full pipeline (SNN encoder → TDA
detector → federated aggregate → ZK reference proof → verification
script) but it does **not** reproduce the Chapter 4 headline numbers.
Use it to confirm the environment is correctly installed.

```bash
# 1) Create and activate the environment (CPU only)
python -m venv .venv
source .venv/bin/activate            # Linux / macOS
# .venv\Scripts\activate             # Windows
pip install -r requirements.txt

# 2) (Re)create the toy dataset from the committed schema headers
python data/make_toy_dataset.py --mode synthetic --seed 42

# 3) Run the toy pipeline
python scripts/train_pafa.py --config configs/default_pafa_config.yaml --mode toy

# 4) Verify toy outputs against the committed tolerances
python expected_outputs/verification_script.py --toy-mode
# Exit code 0 = all toy checks pass; exit code 1 = one or more checks failed.
```

**Expected runtime:** approximately 60–120 s on a modern laptop CPU
(8 cores, 16 GB RAM). No GPU required.

---

## 4. Expected accuracy warning — read before reporting any number

> **The toy mode reaches approximately 85–90 % detection accuracy, not
> 99.2 %.** This is intentional. The reported Chapter 4 numbers
> (99.6 % DNS-tunnelling detection, 99.4 % global federated accuracy,
> ε = 1.2, 0.8 ms Loihi latency, etc.) require the full 32 GB Zenodo
> dataset and either the calibrated hardware simulators or the actual
> Intel Loihi 2 hardware.

The toy mode is a smoke test. Do **not** use the toy-mode outputs to
populate Table 13–17 in any downstream document. The verification
script will refuse to compare toy-mode outputs against the strict
`rtol=0.01` tolerance of the full mode; it relaxes the tolerance to
`rtol=0.20` for toy mode and prints a banner reminding the user that
these are not the reported numbers.

---

## 5. Full reproduction (2–4 hours)

Full reproduction downloads the 32 GB Zenodo corpus
(`10.5281/zenodo.5957677`) plus the CIC-DoH2023 dataset, preprocesses
them into the 47-column feature schema used throughout Chapter 4,
trains the federated model, runs the SNN detector, runs the TDA
significance tests, generates the reference Groth16 proofs, and then
writes every expected-output JSON / CSV file so that
`verification_script.py` can be run in strict mode.

```bash
# 1) Download (≈32 GB; both datasets are pinned by DOI in the config)
python data/download_datasets.py --config configs/default_pafa_config.yaml

# 2) Preprocess into the canonical 47-column schema
python data/preprocess_zenodo.py   --config configs/default_pafa_config.yaml
python data/preprocess_doh2023.py  --config configs/default_pafa_config.yaml

# 3) Reproducible train / val / test split (seed = 42)
python data/split_data.py          --config configs/default_pafa_config.yaml

# 4) Train PAFA-DoH (federated + DP-SGD + Byzantine robust)
python scripts/train_pafa.py       --config configs/default_pafa_config.yaml --mode full

# 5) Train the two experimental baselines (centralized CNN-LSTM, rule-based)
python scripts/train_baselines.py  --config configs/default_pafa_config.yaml

# 6) Reproduce Tables 13–17 and Figures 7–10
python scripts/reproduce_tables/table_13_neuromorphic.py
python scripts/reproduce_tables/table_14_attack_metrics.py
python scripts/reproduce_tables/table_15_zk_attestation.py
python scripts/reproduce_tables/table_16_federated_learning.py
python scripts/reproduce_tables/table_17_tda_validation.py
python scripts/reproduce_figures/figure_7_architecture.py
python scripts/reproduce_figures/figure_8_threat_vectors.py
python scripts/reproduce_figures/figure_9_privacy_tradeoff.py
python scripts/reproduce_figures/figure_10_mitigation_privacy.py

# 7) Verify against expected outputs (strict mode, rtol = 0.01)
python expected_outputs/verification_script.py
# Exit code 0 = all 6 tables + 4 figures match; exit code 1 = mismatch.
```

**Expected runtime:** 2–4 hours on a workstation with ≥ 32 GB RAM,
≥ 8 CPU cores, and one NVIDIA A100 (or one A100-emulating 24 GB
GPU). The two dominant costs are the 32 GB preprocessing step
(~90 min with the Dask pipeline described in §6.13 of the thesis) and
the 100-round federated training (8 epochs to convergence; ~60 min).

If you do not have an A100, set
`configs/hardware_emulation_config.yaml::a100::calibration::cpu_scaling_factor`
to `22.0` (the default in this artifact) and the
`src/hardware_simulation/a100_emulation.py` simulator will produce the
same latency and energy trends on CPU within the ±20 % / ±25 %
tolerances documented in `REPRODUCIBILITY_CHECKLIST.md`.

---

## 6. Hardware requirements

| Component | Chapter 4 reference | In this artifact |
|---|---|---|
| **Intel Loihi 2** | 128 neuromorphic cores × 1,024 LIF neurons = 128,000 neurons, 128 M synapses (§4.24.1) | **Proprietary hardware, not included.** Reproduced via `src/hardware_simulation/loihi_mock.py`, a calibrated simulator that matches Table 13 latency (0.8 ms) and energy (0.12 mJ) within ±20 % / ±25 %. |
| **NVIDIA A100** | 6,912 CUDA cores, 40 GB HBM, comparison baseline (Table 13) | **Proprietary hardware, not included.** Reproduced via `src/hardware_simulation/a100_emulation.py` within ±20 % / ±25 % of Table 13. |
| **IBM TrueNorth** | Mentioned in §4.24.1 only as a **comparison** platform; **not used** in any reported experiment. | **Not simulated, not required, not measured.** See `src/hardware_simulation/README_hardware.md` for the explicit non-use statement. |
| **CPU only** | Sufficient for toy mode and for the simulators. | Fully supported. |
| **RAM** | ≥ 32 GB for the full 32 GB Zenodo preprocessing step. | Required for full mode only. |

All numbers tagged `"simulated": true` in the output JSONs were
produced by the simulators. All numbers tagged `"hardware": "Loihi"` or
`"hardware": "A100"` in the JSONs were produced by the simulators unless
the user re-runs on real hardware with `configs/hardware_emulation_config.yaml::loihi_2::honesty_tags::simulated`
set to `false`. The verification script will print a warning banner
whenever it compares a simulated number against a hardware number.

---

## 7. Dataset acquisition

The artifact uses two public datasets. Both are pinned by DOI /
URL in `configs/default_pafa_config.yaml`; no other dataset is used.

| Dataset | Size | DOI / URL | License | Used for |
|---|---|---|---|---|
| Zenodo DoH corpus | ≈ 32 GB | [`10.5281/zenodo.5957677`](https://doi.org/10.5281/zenodo.5957677) | CC-BY 4.0 | Table 14 (attack simulation), Table 16 (federated training), Table 17 (TDA validation), Figures 8–10 |
| CIC-DoH2023 | ≈ 2.5 GB | [`https://www.unb.ca/cic/datasets/dohbrw-2023.html`](https://www.unb.ca/cic/datasets/dohbrw-2023.html) | CC-BY 4.0 | Preprocessing cross-check (47-column schema parity) |

**Download commands are in `data/download_datasets.py`.** The script
does not re-host either dataset; it fetches from the canonical source
and verifies SHA-256 checksums against `data/CHECKSUMS.txt`.

The **toy dataset** (`data/toy_dataset/`) is a 1,000 + 1,000 row
synthetic derivative generated by `data/make_toy_dataset.py --mode
synthetic --seed 42`. It is CC-BY 4.0 and is committed header-only in
this repository so that the toy demo can run immediately after
`pip install`.

---

## 8. Reproducibility checklist summary

See [`REPRODUCIBILITY_CHECKLIST.md`](REPRODUCIBILITY_CHECKLIST.md)
for the per-table verdict, the per-figure verdict, the environment /
platform matrix, the hardware dependency map, the dataset dependency
map, the determinism guarantees, the tolerance bounds table, the known
limitations (mirroring §7.3.5 of the thesis), the reproduction-time
estimates, and the sign-off block.

High-level verdict:

- **Fully reproducible on any CPU:** Tables 15, 17, 17a; Figures 7, 8, 9, 10; toy-mode runs of Tables 13, 14, 16.
- **Fully reproducible with the calibrated simulators:** Tables 13, 14, 16 in full mode.
- **Hardware-bound (simulated ±20 % / ±25 %):** Table 13 rows for Loihi 2 and A100.
- **Data-bound (requires the 32 GB Zenodo corpus):** Tables 14, 16, 17; Figures 8–10.

---

## 9. CLI replay commands from Table 14

The attack-simulation metrics reported in Table 14 are reproduced by
the following sequence. Each command corresponds to one row of
Table 14 and is wrapped by `src/cli_tools/run_all_scenarios.sh`.

```bash
# Row 1 — DNS tunnelling (reported detection rate: 99.5 %)
python src/cli_tools/federated_cli.py train --clients 100 --epsilon 1.2 --rounds 100
python src/cli_tools/snn_cli.py infer --loihi-simulate
python src/cli_tools/tda_cli.py betti --scale 50
python src/attacks/dns_tunneling.py --generate 1000 --seed 42

# Row 2 — C2 heartbeat (reported detection rate: 98.7 %)
python src/cli_tools/tda_cli.py significance --resamples 1000
python src/attacks/c2_exfiltration.py --generate 1000 --seed 42

# Row 3 — Side-channel (reported detection rate: 100 %, no key recovery)
python src/cli_tools/zk_cli.py prove --constraints 100000
python src/cli_tools/zk_cli.py verify --proof-file outputs/proof.bin
python src/attacks/side_channel.py --chipwhisperer-sim --traces 1000

# Full replay
bash src/cli_tools/run_all_scenarios.sh
```

---

## 10. Expected outputs and tolerances

Every table and figure in this artifact has a committed expected
output file in `expected_outputs/`. The verification script
(`expected_outputs/verification_script.py`) recursively compares a
freshly generated JSON / CSV against the committed expected file and
reports per-field pass / fail.

- **Strict mode (default, full reproduction):** `rtol = 0.01` for
  Table 16 accuracy, `rtol = 0.05` for TDA p-values, `rtol = 0.05` for
  the remaining numeric fields.
- **Toy mode (`--toy-mode`):** `rtol = 0.20` for all fields. A banner
  is printed explaining that toy mode does **not** reproduce the
  reported Chapter 4 numbers.
- **Tolerance table** is also reproduced in
  `REPRODUCIBILITY_CHECKLIST.md` §7.

Exit codes:

- `0` — all checks pass.
- `1` — at least one check failed; the failing field, the expected
  value, the observed value, and the relative error are printed.

---

## 11. Tests

```bash
# Unit tests for the mathematics (Eqs. 35, 36, 38, 39, 40, 41, 43, 44,
# 46, 58, 60 — see src/utils/math_utils.py)
pytest tests/test_math_utils.py -v

# End-to-end model tests (FL, SNN, TDA, ZK)
pytest tests/test_models.py -v

# CLI replay tests (Table 14 commands)
pytest tests/test_cli_tools.py -v

# Deterministic reproduction of Table 17a + toy-mode tolerance check
pytest tests/test_reproduce_tables.py -v
```

CI-style single command:

```bash
pytest tests/ -v
```

Expected total runtime on a laptop CPU: ≈ 4 min (toy mode). Full
mode is not exercised by `pytest`; run the scripts in Section 5 for
that.

---

## 12. Citation

If you use this artifact, please cite the chapter:

```bibtex
@article{pafa_doh_2026,
  title   = {Proactive and Privacy-Preserving Defense for DNS over HTTPS
             via Federated AI Attestation (PAFA-DoH)},
  author  = {[REDACTED FOR BLIND REVIEW]},
  journal = {Neural Networks},
  year    = {2026},
  month   = {April},
  doi     = {10.1016/j.neunet.2025.108343},
  note    = {Reproducibility artifact:
             https://github.com/<your-org>/PAFA-DoH-Reproducibility}
}
```

The citation DOI is `10.1016/j.neunet.2025.108343` (Neural Networks,
April 2026). Do not cite the MIRAGE-DoH DOI, the E3-DoH DOI, or the
A3E-DNS DOI in this repository — this artifact is PAFA-DoH only.

---

## 13. License

- **Code, configs, scripts, tests, expected outputs, documentation:**
  [MIT License](LICENSE).
- **Zenodo DoH corpus:** CC-BY 4.0 (upstream; see
  `https://doi.org/10.5281/zenodo.5957677`).
- **CIC-DoH2023:** CC-BY 4.0 (upstream; see
  `https://www.unb.ca/cic/datasets/dohbrw-2023.html`).
- **Intel Loihi 2 and NVIDIA A100:** proprietary hardware, not
  redistributed.
- **Toy dataset** (`data/toy_dataset/`): CC-BY 4.0, derivative of the
  Zenodo corpus, committed header-only so the toy demo can run without
  a download.
- **Third-party Python packages:** retain their own licenses; see
  `requirements.txt`.

See [`LICENSE`](LICENSE) for the exact scope note.

---

## 14. Contact

Open a GitHub issue on this repository for questions about the
artifact, reproduction failures, or tolerance-mismatch reports. When
reporting a mismatch, please attach:

1. The output of `python -c "import sys, platform; print(sys.version); print(platform.platform())"`.
2. The output of `pip freeze`.
3. The failing field printed by `verification_script.py`.
4. The full command you ran (toy mode vs full mode).

---

## 15. Anti-confusion checklist (read before publishing any result from this repository)

- [ ] Every numeric value in a downstream document matches **Chapter 4**, not Chapter 3. SNN latency = **0.8 ms** (not 78 µs). ZK proof generation = **432 ms** (not 8.9 ms). Accuracy = **99.4 % global** (not 99.98 %).
- [ ] No claim of IBM TrueNorth experiments. Only Loihi was used (§4.24.1). TrueNorth is mentioned only for comparison.
- [ ] No claim of provable security for the detection module. Only the ZK-SNARK has a soundness argument (Eqs. 61–62).
- [ ] No claim of a "new" primitive or algorithm. Every component cites its closest published method in Table 17a.
- [ ] Dataset URL is correct: Zenodo DOI `10.5281/zenodo.5957677`, **no trailing slash**.
- [ ] Repository name is **`PAFA-DoH-Reproducibility`**, not `MIRAGE-DoH-Reproducibility`.
- [ ] Citation DOI is **`10.1016/j.neunet.2025.108343`** (Neural Networks, April 2026), not the MIRAGE-DoH DOI.
- [ ] README quick start uses `train_pafa.py`, not `train_mirage.py`.
- [ ] All Loihi / A100 simulation outputs are tagged `"simulated": true`.
