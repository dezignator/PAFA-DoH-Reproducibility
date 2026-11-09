<!--
README.md — PAFA-DoH Reproducibility Artifact
Documentation for the expected_outputs/ directory.
═══════════════════════════════════════════════════════════════════════════

Companion to:
  README.md (File 1), REPRODUCIBILITY_CHECKLIST.md (File 2),
  LICENSE (File 3).

Thesis chapter: Chapter 4 (PAFA-DoH)
Citation DOI:   10.1016/j.neunet.2025.108343
License:        MIT
Artifact version: 1.0.0
═══════════════════════════════════════════════════════════════════════════
-->

# Expected Outputs — PAFA-DoH Reproducibility Artifact

This directory holds the **committed reference values** that
`verification_script.py` compares against a fresh pipeline run.
Every reported number in Tables 13–17 of Chapter 4 has a
corresponding entry here.

Full documentation of the tolerances, the toy-mode skip rules, and
the anti-confusion discipline is in
[`../REPRODUCIBILITY_CHECKLIST.md`](../REPRODUCIBILITY_CHECKLIST.md) §8
and [`../README.md`](../README.md) §4.

---

## Contents

| File | Sourced from | Compares against |
|---|---|---|
| `table_13_expected.json` | `scripts/reproduce_tables/table_13_neuromorphic.py` | `outputs/tables/table_13_neuromorphic.json` |
| `table_14_expected.json` | `scripts/reproduce_tables/table_14_attack_metrics.py` | `outputs/tables/table_14_attack_metrics.json` |
| `table_15_expected.json` | `scripts/reproduce_tables/table_15_zk_attestation.py` | `outputs/tables/table_15_zk_attestation.json` |
| `table_16_expected.json` | `scripts/reproduce_tables/table_16_federated_learning.py` | `outputs/tables/table_16_federated_learning.json` |
| `table_17_expected.json` | `scripts/reproduce_tables/table_17_tda_validation.py` | `outputs/tables/table_17_tda_validation.json` |
| `figure_8_expected.csv` | `scripts/reproduce_figures/figure_8_threat_vectors.py` | `expected_outputs/figure_8_expected.csv` (self-referential; see note) |
| `figure_9_expected.csv` | `scripts/reproduce_figures/figure_9_privacy_tradeoff.py` | `expected_outputs/figure_9_expected.csv` (self-referential; see note) |
| `figure_10_expected.csv` | `scripts/reproduce_figures/figure_10_mitigation_privacy.py` | `expected_outputs/figure_10_expected.csv` (self-referential; see note) |
| `verification_script.py` | (the verifier itself) | — |

**Note on figures.** The figure scripts (`figure_8_threat_vectors.py`,
`figure_9_privacy_tradeoff.py`, `figure_10_mitigation_privacy.py`)
write their output CSVs *directly into this directory*. This means
the committed CSV is the reference **before** the first run; after
running the scripts, the committed file is overwritten by the fresh
one, and verification compares the fresh file against itself. For a
meaningful figure-level verification, restore the canonical version
of this directory before running verification:

```bash
git checkout HEAD -- expected_outputs/
python expected_outputs/verification_script.py --toy-mode
```

This design is honest: the figures are reproducible, but their
generation overwrites their own reference. Tables do not have this
problem (they write to `outputs/tables/`, never to this directory).

---

## Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Clone the repository                                         │
│    (expected_outputs/ contains canonical reference values)      │
└──────────────────────────────────┬──────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Run the pipeline (README §3 for toy, §5 for full)            │
│    Populates outputs/ and (for figures) overwrites expected/    │
└──────────────────────────────────┬──────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. (Optional, for figures only) restore the canonical expected  │
│    files:  git checkout HEAD -- expected_outputs/               │
└──────────────────────────────────┬──────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Run verification                                             │
│    python expected_outputs/verification_script.py [--toy-mode]  │
│    Exit 0 = pass; exit 1 = fix the FAIL lines                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tolerances (summary)

The full per-field tolerance table is in
`../REPRODUCIBILITY_CHECKLIST.md` §8. The high-level summary:

| Group | Strict mode | Toy mode | Rationale |
|---|---|---|---|
| Table 13 latency / energy | rtol = 0.20 / 0.25 | same | Hardware simulator calibration bound |
| Table 13 speedup / efficiency | rtol = 0.05 | same | Derived ratios; noise propagates |
| Table 14 detection rates | rtol = 0.01 | rtol = 0.20 | Rates are empirical; CI is tighter than 1 % |
| Table 15 timings | rtol = 0.05 | same | CPU-bound ZK wall-clock |
| Table 15 soundness / size | exact | exact | Construction parameters, not measurements |
| Table 16 accuracy | rtol = 0.01 | rtol = 0.20 | ±0.12 % CI on 99.4 % |
| Table 16 CI half-width | rtol = 0.10 | **skipped** | Full-corpus-only field |
| Table 16 epochs | exact | exact | Integer |
| Table 16 dropout / adversarial | rtol = 0.02 | rtol = 0.20 | 2 % head-room |
| Table 17 means / stds | rtol = 0.05 | rtol = 0.20 | TDA estimates |
| Table 17 p-values | bracket-preserving | same | `< 0.001` is a bracket, not a point |
| Figure 8 accuracy | atol = 0.10 | same | Absolute tolerance (values transition 0 → ~1.0) |
| Figures 9 and 10 | rtol = 0.05 | same | Deterministic curves |

**Skipped fields.** In toy mode, `table_16.global_model_accuracy_ci_95_pct`
is skipped with a printed warning. Rationale: the ±0.12 % CI on 99.4 %
accuracy requires the millions of rows in the full 32 GB corpus; the
toy dataset has 2,000 rows total (400 in the test split), so the CI is
naturally ~0.69 %, not 0.12 %. The mismatch is a scale effect, not a
bug. See `../REPRODUCIBILITY_CHECKLIST.md` §9 for the corresponding
Known Limitations note.

---

## Anti-confusion discipline

Every field in every expected file is annotated (in the JSON itself
for tables, in the CSV column names for figures) with the thesis
section it comes from, so a reviewer can cross-check without the PDF.

**Field-tag legend (tables only):**

- `targets.*` — the value reported in the thesis (the reference).
- `metric.*`, `metrics.*`, `p_values.*` — the observed values from
  the current run (compared against `targets.*`).
- `notes.*` — provenance and caveats.

For figures, the CSV column names carry the method names; the
`illustrative` column in `figure_9_expected.csv` marks which curves
are *not* measured (per §4.27.3).

**What this directory does NOT contain:**

- `table_17a_expected.json` — Table 17a is a configuration-only table
  (no measured values to verify). It is produced by
  `scripts/reproduce_tables/table_17a_closest_methods.py` for the
  reviewer's inspection but is not part of the verification flow.
- `figure_7_expected.csv` — Figure 7 is a Graphviz schematic (no
  numeric values to verify). It is produced by
  `scripts/reproduce_figures/figure_7_architecture.py` but is not
  part of the verification flow.

---

## Known limitations

1. **Figure expected files are self-referential.** Running the figure
   scripts overwrites the committed reference. See the "Note on
   figures" block in §Contents and the workflow diagram in §Workflow.

2. **Toy-mode CI skip.** Covered above.

3. **Verification is per-file, not per-field-selection.** A reviewer
   who wants to check only one table can move the other expected
   files out of the way, or run the corresponding reproduce script
   and then verify — the verifier skips missing files (prints SKIP,
   does not fail).

4. **The figure_8 expected CSV is noise-free.** The script itself
   adds seed-dependent Gaussian noise on top of the deterministic
   base curve. The committed expected CSV has the noise-free values;
   verification uses `atol = 0.10` on `accuracy_mean`, which covers
   the ±0.06 three-sigma bound of the noise. See `figure_8_expected.csv`
   header and File 20f for the per-epoch values.

---

## License

The files in this directory are MIT-licensed (see
`../LICENSE`). They are derivative of the pipeline's output and do
not contain any dataset rows. See `../data/README_data.md` §6 for
the dataset-level checksum discipline.

---

*End of `expected_outputs/README.md`.*
