# PAFA-DoH Chapter 4: extended artifact (v2)

Additional material for Chapter 4 (*Privacy-Preserving Federated Defense for
Encrypted DNS*). Everything lives under `extended_v2/` and every file name ends
in `_v2`, so nothing here replaces an existing file.

## Layout

| Path | Content |
|---|---|
| `pafa_v2/` | Reference implementations of the numbered equations (Python, numpy/scipy only) |
| `tests/` | 75 unit tests, analytic cases and cross-checks |
| `c/` | C99 LIF population kernel + self-test + CPU benchmark (Sec 4.24.1 layout) |
| `js/` | RFC 1035/8484 query codec, subdomain-entropy feature, seeded tunnel corpus (Node >= 18) |
| `tables/` | `reported_values_v2.yaml` (thesis values, with evidence class) and generated CSV / Markdown / LaTeX |
| `latex/` | Standalone build of Tables 13-17a |
| `configs/` | `params_v2.yaml`, `run_manifest_v2.yaml` (every value the thesis says is "recorded in the artifact") |
| `scripts/` | table generator, consistency checks, manifest validator, TDA pipeline demo, env report, run-all |
| `experiments/` | three large suites: federated training (E1-E6), attack scenarios + detectors (A1-A7), report builder + component benchmarks (B1-B7) |
| `results_v2/` | outputs of those suites as run for this release (CSV, PNG, Markdown, HTML report) |
| `docs/` | equation-to-code map, hardware dependency matrix, threat model, artifact checklist, parameter note |
| `docker/` | CPU container recipe |

## Quick start

```bash
cd extended_v2
pip install -r requirements_v2.txt
make test              # python + C + node
make consistency       # arithmetic checks on Tables 13-16
make tables            # CSV / MD / LaTeX
bash scripts/run_all_v2.sh
make experiments       # ~6 min, 1 core; writes results_v2/
make report            # HTML + Markdown report
```

## What this does and does not reproduce

- Implements the formulations of Eqs. 36-67 as software references and tests them
  against analytic cases (e.g. LIF inter-spike period, circle H1 bar, R1CS/QAP
  divisibility, Merkle openings, FRI folding, Ring-LWE round trip).
- Does **not** regenerate hardware figures (Loihi/A100 latency and energy,
  ChipWhisperer SNR). See `docs/HARDWARE_DEPENDENCY_MATRIX_v2.md`.
- `scripts/tda_pipeline_demo_v2.py` runs on synthetic flows. Its output checks the
  pipeline; it is not Table 17.
- `zk_r1cs_v2.py` and `merkle_fri_v2.py` are didactic: toy circuit, no pairing,
  no trusted setup. Table 15 times come from the Groth16 prover used in the thesis.

## Filling the manifest

`configs/run_manifest_v2.yaml` lists the parameters that Chapter 4 defers to the
artifact (seeds, split sizes, resamples, SNN encoding constants, FRI blow-up, etc.).
`python3 scripts/validate_manifest_v2.py configs/run_manifest_v2.yaml --strict`
exits non-zero while any entry is `null`.
