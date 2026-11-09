<!--
README_data.md — PAFA-DoH Reproducibility Artifact
Directory-level documentation for the data/ layer.
═══════════════════════════════════════════════════════════════════════════

Companion to:
  README.md                       (File 1 of 74)
  REPRODUCIBILITY_CHECKLIST.md    (File 2 of 74)
  LICENSE                         (File 3 of 74)
  requirements.txt                (File 4 of 74)
  environment.yml                 (File 5 of 74)
  Dockerfile                      (File 6 of 74)
  setup.py                        (File 7 of 74)
  .gitignore                      (File 8 of 74)
  configs/default_pafa_config.yaml (File 9, master config)
  configs/*.yaml                  (Files 9b–9h)

Thesis chapter: Chapter 4 (PAFA-DoH)
Citation DOI:   10.1016/j.neunet.2025.108343   (Neural Networks, April 2026)
License:        MIT (this file, as part of the artifact's documentation)
Artifact version: 1.0.0

This file is the directory-level entry point for the data/ layer. It
documents every file in the directory, the data flow through the four
pipelines (Zenodo, CIC, toy, checksum-verify), the commit-vs-generate
boundary, the checksum mechanism, and the seven most common errors.

It does NOT repeat the run instructions from README.md §3 and §5; it
cross-references them. It does NOT repeat the verdicts from
REPRODUCIBILITY_CHECKLIST.md §3–§6; it cross-references them.
═══════════════════════════════════════════════════════════════════════════
-->

# Data Layer — PAFA-DoH Reproducibility Artifact

Directory-level documentation for `data/`.
Companion to the top-level [`README.md`](../README.md), [`REPRODUCIBILITY_CHECKLIST.md`](../REPRODUCIBILITY_CHECKLIST.md), and [`configs/default_pafa_config.yaml`](../configs/default_pafa_config.yaml).

---

## 1. Purpose & scope

The `data/` directory is the **input side** of the PAFA-DoH artifact. It
provides:

- **Two dataset downloaders** (Zenodo DoH corpus, CIC-DoH2023) — Files 10b.
- **Two dataset preprocessors** (Zenodo → training source; CIC → schema-parity check) — Files 10c, 10d.
- **One splitter** (seeded, stratified, deterministic) — File 10e.
- **One synthetic toy generator** (2,000 rows, no network) — File 10f.
- **Shared utilities** (47-column schema, SHA-256, artifact-root resolution, `write_split()`) — File 10a.
- **Schema templates** (two header-only CSVs) — Files 10h–10i.
- **A checksum manifest** — File 10j.
- **Two placeholders** (`data/raw/.gitkeep`, `data/processed/.gitkeep`) — Files 10l–10m.
- **This document** — File 10g.

Scope is **Chapter 4 (PAFA-DoH) only**. No other framework's data layer is in this directory. The citation DOI for the chapter is `10.1016/j.neunet.2025.108343` (Neural Networks, April 2026).

---

## 2. File inventory

Every file in `data/` is listed below. The `Role` column distinguishes
**script** (executable Python), **document** (Markdown), **template**
(header-only CSV, committed), **manifest** (checksum list, committed), and
**placeholder** (empty file that pins a directory, committed).

| File | Role | Chapter section | Purpose |
|---|---|---|---|
| `data_utils.py` | script | §4.24.1, §4.25.1 | 47-column schema, SHA-256 helpers, artifact-root resolver, `write_split()`. Single source of truth for the schema. |
| `download_datasets.py` | script | §4.24.1, §4.25.1 | Downloads the two pinned datasets (Zenodo DOI, CIC URL), verifies checksums, and delegates the `toy` sub-command to `make_toy_dataset.py`. |
| `preprocess_doh2023.py` | script | §4.24.1, §4.25.1 | Preprocesses CIC-DoH2023 into the 47-column schema. This is the **schema-parity check**, not the training source. May fill non-derivable columns with a sentinel; `--strict-schema` turns those fills into errors. |
| `preprocess_zenodo.py` | script | §4.24.1, §4.25.1 | Preprocesses the 32 GB Zenodo DoH corpus into the 47-column schema under Dask. This is the **training source**. Zero sentinel fills — `to_schema()` fails closed if any column would require one. |
| `split_data.py` | script | §4.25.1, Table 16 | Seeded stratified train/val/test split. Consumes the output of either preprocessor. Writes `split_{train,val,test}.csv` and `split_manifest.json`. |
| `make_toy_dataset.py` | script | README §3, §4 | Generates the 1,000 + 1,000 row toy dataset used by the two-minute smoke test. Two modes: `synthetic` (default) and `from-full`. |
| `README_data.md` | document | (this file) | This document. Directory-level documentation. |
| `toy_dataset/benign_sample.csv` | template | README §3 | Header-only. The 47-column schema as a CSV. Used by `make_toy_dataset.py` as the schema template and by `data_utils.py::read_toy_sample()` as the schema validator fixture. |
| `toy_dataset/malicious_sample.csv` | template | README §3 | Header-only. Same schema as `benign_sample.csv`. |
| `CHECKSUMS.txt` | manifest | §4.24.1, §4.25.1 | SHA-256 manifest for the two downloaded datasets. Read by `data_utils.py::verify_checksums()`. |
| `README.md` | document | (this file) | Short pointer to `README_data.md` for anyone who lands in `data/` first. |
| `raw/.gitkeep` | placeholder | — | Pins `data/raw/` so a fresh clone has the empty directory that the downloader writes into. |
| `processed/.gitkeep` | placeholder | — | Pins `data/processed/` so a fresh clone has the empty directory that the preprocessors and splitter write into. |

---

## 3. Data flow

There are **four** independent flows through `data/`. The first is the
training path (Zenodo), the second is the schema-parity path (CIC), the
third is the toy-demo path, and the fourth is the verify-only path.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         FLOW 1 — TRAINING (ZENODO)                       │
└──────────────────────────────────────────────────────────────────────────┘

   download_datasets.py zenodo
            │
            ▼
   data/raw/zenodo/*.{csv,json,jsonl,parquet}        (≈ 32 GB)
            │
            ▼
   preprocess_zenodo.py            ── enforce_forbidden_observables (File 10c)
            │                       ── drop_malformed_flows
            │                       ── compute_features (8 families)
            │                       ── normalize (two-pass, global stats)
            │                       ── to_schema (fail-closed, zero sentinels)
            ▼
   data/processed/zenodo_doh.csv
   data/processed/zenodo_doh.parquet.d/
            │
            ▼
   split_data.py --source zenodo
            │
            ▼
   data/processed/split_{train,val,test}.csv
   data/processed/split_manifest.json
            │
            ▼
   scripts/train_pafa.py --mode full

┌──────────────────────────────────────────────────────────────────────────┐
│                    FLOW 2 — SCHEMA-PARITY CHECK (CIC)                    │
└──────────────────────────────────────────────────────────────────────────┘

   download_datasets.py cic_doh2023
            │
            ▼
   data/raw/cic_doh2023/*.csv + _landing_page.html   (≈ 2.5 GB)
            │
            ▼
   preprocess_doh2023.py           ── enforce_forbidden_observables
            │                       ── drop_malformed_flows
            │                       ── map_columns (alias table)
            │                       ── derive_missing_features
            │                       ── normalize
            ▼
   data/processed/cic_doh2023.csv
            │
            ▼
   split_data.py --source cic_doh2023
            │
            ▼
   data/processed/split_{train,val,test}.csv
   data/processed/split_manifest.json
   (used only to confirm 47-col schema parity; NOT used to reproduce
    Chapter 4's reported numbers)

┌──────────────────────────────────────────────────────────────────────────┐
│                        FLOW 3 — TOY DEMO (SYNTHETIC)                     │
└──────────────────────────────────────────────────────────────────────────┘

   make_toy_dataset.py --mode synthetic --seed 42
            │
            ▼
   data/processed/toy_dataset/benign_sample.csv      (1,000 rows)
   data/processed/toy_dataset/malicious_sample.csv   (1,000 rows)
   data/processed/toy_dataset/combined.csv           (2,000 rows)
            │
            ▼
   scripts/train_pafa.py --mode toy

┌──────────────────────────────────────────────────────────────────────────┐
│                       FLOW 4 — VERIFY (CHECKSUMS ONLY)                   │
└──────────────────────────────────────────────────────────────────────────┘

   download_datasets.py verify
            │
            ▼
   data_utils.py::verify_checksums(CHECKSUMS.txt, data/raw/)
            │
            ▼
   exit 0 (all match) or exit 1 (mismatch reported)
```

**Which flow a reviewer should run** depends on what they are
reproducing:

| Reviewer goal | Flow to run | Approx. runtime |
|---|---|---|
| Confirm the environment is correctly installed | Flow 3 (toy demo) | ≈ 2 min |
| Reproduce Tables 15, 17, 17a; Figures 7–10 | Flow 3 + the corresponding `scripts/reproduce_*` | ≈ 15 min |
| Reproduce Tables 14, 16; Figures 8–10 | Flow 1 (full) + `scripts/train_pafa.py --mode full` | ≈ 3–4 h |
| Confirm the 47-column schema is stable | Flow 2 (CIC) | ≈ 15 min + 2.5 GB download |
| Verify a corpus that was downloaded earlier | Flow 4 | < 1 min |

The exact steps for Flow 1 are in README.md §5. The exact steps for
Flow 3 are in README.md §3.

---

## 4. Directory layout

```
data/
├── data_utils.py                  (File 10a)  — shared utilities
├── download_datasets.py           (File 10b)  — dataset downloader
├── preprocess_doh2023.py          (File 10c)  — CIC preprocessor
├── preprocess_zenodo.py           (File 10d)  — Zenodo preprocessor
├── split_data.py                  (File 10e)  — seeded splitter
├── make_toy_dataset.py            (File 10f)  — toy generator
├── README_data.md                 (File 10g)  — this file
├── CHECKSUMS.txt                  (File 10j)  — SHA-256 manifest
├── README.md                      (File 10k)  — pointer to README_data.md
│
├── toy_dataset/                   — COMMITTED schema templates
│   ├── benign_sample.csv          (File 10h)  — header only
│   └── malicious_sample.csv       (File 10i)  — header only
│
├── raw/                           — NOT committed (downloaded on demand)
│   └── .gitkeep                   (File 10l)
│
└── processed/                     — NOT committed (generated)
    └── .gitkeep                   (File 10m)
```

The `.gitkeep` files exist because Git does not track empty
directories. Without them, a fresh clone would have no `data/raw/` and
no `data/processed/`, and `download_datasets.py` would have to create
them at run time (which it does anyway, but the directory-check
messages in `resolve_artifact_root()` would be less informative).

---

## 5. What is committed vs generated

The boundary between committed and generated files is enforced by
`.gitignore` (File 8) rules 6–8. The table below states it
explicitly, because it is the single most common source of confusion
in the first hour of using the artifact.

| Path | Committed? | Why |
|---|---|---|
| `data/toy_dataset/*.csv` | **Yes**, header-only | The header row is the canonical 47-column schema as a CSV. It is the schema template that `make_toy_dataset.py` and `data_utils.py::read_toy_sample()` read. Zero data rows, so the file is ~2 KB and never churns. **Never overwritten by the toy generator** (which writes to `data/processed/toy_dataset/`). |
| `data/raw/*` | **No**, except `data/raw/.gitkeep` | Downloaded on demand from the pinned DOI / URL by `download_datasets.py`. The 32 GB Zenodo corpus and 2.5 GB CIC bundle are not redistributed by this repository (see LICENSE §2). Re-downloadable, so committing them would be wasteful and would republish CC-BY 4.0 data inside an MIT-licensed repository. |
| `data/processed/*` | **No**, except `data/processed/.gitkeep` | Generated by the preprocessors (`preprocess_*.py`) and the splitter (`split_data.py`). Re-generated deterministically from the raw corpus, the config, and the seed. Committing them would duplicate data and hide bugs behind stale files. |
| `data/CHECKSUMS.txt` | **Yes** | The SHA-256 manifest for the two downloaded datasets. Pinned, diff-able, and read by `data_utils.py::verify_checksums()`. The manifest is the record of "what the corpus was" at the time the expected outputs were produced. |

**Consequences of this boundary:**

1. After a fresh `git clone`, `data/processed/` is empty. You must run
   Flow 1 or Flow 3 to populate it. The toy demo (Flow 3) does this in
   ≈ 2 s with no network.
2. After `python data/download_datasets.py zenodo`, `data/raw/zenodo/`
   is populated but `data/processed/` is still empty. You must run
   `python data/preprocess_zenodo.py` to populate the latter.
3. `git status --ignored` after the full Flow 1 pipeline shows
   `data/raw/zenodo/`, `data/processed/zenodo_doh.*` and
   `data/processed/split_*.csv` as ignored (`!!`), and shows
   `data/toy_dataset/*.csv` and `data/CHECKSUMS.txt` as tracked (no
   prefix). See `.gitignore` (File 8) rule 7 for the `!` exceptions
   that make this work.

---

## 6. Checksum verification

`data/CHECKSUMS.txt` (File 10j) is a plain-text manifest of SHA-256
digests for the two downloaded datasets. It exists so that a reviewer
can prove that the 32 GB they downloaded is the 32 GB the expected
outputs were produced against, without trusting the network, the CDN,
or the DOI registry.

**Format.** One record per line. Blank lines and lines beginning with
`#` are ignored. The two-column format is:

```
<sha256-hex-digest>  <relative-path-from-data-raw>
```

The digest is 64 lowercase hex characters. The separator is exactly two
spaces (POSIX `sha256sum` output format; the two-space convention
disambiguates from the single-space format used by BSD `md5` for
filenames that contain spaces).

**Example (two fabricated records, for illustration only — the real
manifest will contain the Zenodo and CIC file digests):**

```
# PAFA-DoH — SHA-256 manifest for data/raw/
# Generated by download_datasets.py on 2026-11-09T14:02:00Z.
# Verify with: python data/download_datasets.py verify
#
4f8b6c8a1a9e2d3c7b5f4e2d1c0b9a8e7d6c5b4a3f2e1d0c9b8a7f6e5d4c3b2a  zenodo/record_files.tar.zst
1c2b3a4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b  cic_doh2023/cic-doh2023.csv
```

**Verification command.** The simplest form is:

```bash
python data/download_datasets.py verify
```

which reads `data/CHECKSUMS.txt`, hashes every file under
`data/raw/`, and exits 0 if every digest matches and 1 otherwise. On a
mismatch, the error names the file, the expected digest, and the
observed digest, and instructs the user to delete the file and
re-download.

The equivalent shell command (for a reviewer who wants to bypass the
Python wrapper) is:

```bash
cd data/raw && sha256sum -c ../CHECKSUMS.txt
```

on Linux / macOS, or:

```powershell
Get-FileHash -Algorithm SHA256 -Path <file>
```

on Windows.

**When to run it.** After `download_datasets.py zenodo` and after
`download_datasets.py cic_doh2023`. The downloader runs it
automatically; the manual invocation is only needed if a reviewer
wants to re-verify a corpus that was downloaded in a previous session.

**What a mismatch means.** A SHA-256 mismatch almost always means one
of three things:

1. **A partial download.** The downloader's resume path (HTTP `Range`)
   was interrupted, and the file is short. Delete the file and re-run
   `download_datasets.py zenodo --force`.
2. **A source-side change.** The Zenodo record or the CIC bundle was
   updated upstream after the manifest was pinned. This is rare but
   possible; check `data/CHECKSUMS.txt` and the git log of the
   repository for the file, and if the upstream change is real,
   open an issue on the repository rather than editing the manifest
   locally.
3. **A corrupted local copy.** A filesystem error, a concurrent write,
   or an antivirus tool that modified the file. Delete and re-download.

**What a mismatch does NOT mean.** It does not mean the manifest is
wrong. The manifest is committed and pinned; if the artifact's expected
outputs were produced against a different corpus, that would be a bug
in the artifact, and the correct response is to open an issue, not to
edit the manifest.

---

## 7. Cross-reference to the other top-level documents

The table below maps each data-layer file to the sections of `README.md`
and `REPRODUCIBILITY_CHECKLIST.md` that refer to it, so a reviewer who
starts from one of those documents can find the file-level detail.

| Data file | README.md section | REPRODUCIBILITY_CHECKLIST.md section |
|---|---|---|
| `data_utils.py` | §3 Quick start, §5 Full reproduction | §2 (platform matrix), §8 (tolerance table — the `write_split()` determinism guarantee is part of §7 "Determinism guarantees") |
| `download_datasets.py` | §5 step 1, §7 Dataset acquisition | §6 (dataset dependency map) |
| `preprocess_doh2023.py` | §5 step 2 | §6 (dataset dependency map — CIC row), §9 (Known limitations — schema-parity only) |
| `preprocess_zenodo.py` | §5 step 2, §7 | §3 (Table 14, 16, 17 rows), §6 (Zenodo row), §10 (reproduction-time estimates — preprocessing row) |
| `split_data.py` | §5 step 3 | §7 (Determinism guarantees — the splitter is deterministic), §8 (tolerance table — `split_manifest.json` is the manifest read by `verification_script.py`) |
| `make_toy_dataset.py` | §3 Quick start, §4 Accuracy warning | §2 (platform matrix — toy mode runtime), §10 (reproduction-time estimates — toy-mode rows) |
| `README_data.md` | §2 Repository structure | §1 Purpose & scope |
| `toy_dataset/*.csv` | §2 Repository structure | §6 (Toy dataset row) |
| `CHECKSUMS.txt` | §7 Dataset acquisition | §6 (dataset dependency map — `verify_checksums()` note) |
| `raw/.gitkeep`, `processed/.gitkeep` | §2 Repository structure | (not referenced; placeholders are transparent to reviewers) |

---

## 8. Troubleshooting

Seven errors account for essentially every data-layer problem a
reviewer can hit. Each is one paragraph, one-line diagnosis, and a
remedy. The remedy is always one of: (a) set `PAFA_ROOT`, (b) run the
downloader, (c) run the preprocessor, (d) delete the corrupt file, or
(e) pass `--force`. If the error is not one of the seven, open an
issue with the command, the environment, and the full traceback.

### 8.1 `Could not resolve the PAFA-DoH artifact root`

**Cause:** you are running a data-layer script from a directory that
is not a checkout of this repository and `$PAFA_ROOT` is not set.

**Fix:** either `cd` to the repository root and re-run, or set
`PAFA_ROOT`:

```bash
PAFA_ROOT=$(pwd) python data/data_utils.py
```

The error message names every path that was tried. The resolver's
three-step contract is documented in `data_utils.py::resolve_artifact_root`
and in the footer of `README.md` (File 1) and `setup.py` (File 7).

### 8.2 `Checksum mismatch for ...`

**Cause:** the downloaded file does not match the pinned SHA-256 in
`data/CHECKSUMS.txt`. See §6 above for the three possible causes.

**Fix:** delete the file and re-download with `--force`:

```bash
rm -rf data/raw/zenodo/<file>
python data/download_datasets.py zenodo --force
```

If the mismatch persists on a fresh download, check the git log for
`data/CHECKSUMS.txt` — a real upstream change would require a manifest
update and a re-pin, which is a maintainer decision, not a user fix.

### 8.3 `No .csv files found in .../cic_doh2023`

**Cause:** `download_datasets.py cic_doh2023` saves the CIC landing
page as `_landing_page.html` and prints a manual-extraction
instruction. The user has not yet placed the CSVs in the directory.

**Fix:** open the landing page, obtain the CSVs, and place them
directly in `data/raw/cic_doh2023/`:

```bash
ls data/raw/cic_doh2023/
# _landing_page.html  <the CSVs you extracted>
```

Then re-run `python data/preprocess_doh2023.py`.

### 8.4 `No data files (...csv, .json, .jsonl, .parquet) found in .../zenodo`

**Cause:** `download_datasets.py zenodo` was not run, or it wrote to a
nested directory (e.g. `data/raw/zenodo/zenodo/`), or it was
interrupted before writing anything.

**Fix:** re-run the downloader, or move the files up one level:

```bash
ls data/raw/zenodo/
# <the files>    OR    zenodo/    <-- if you see this nested dir, move it:
# mv data/raw/zenodo/zenodo/* data/raw/zenodo/ && rmdir data/raw/zenodo/zenodo
```

### 8.5 `Existing split file(s) present: ...`

**Cause:** `split_data.py` was already run, and the default policy is
to refuse to overwrite a split that a reviewer might be relying on.

**Fix:** re-run with `--force`:

```bash
python data/split_data.py --force
```

The `--force` flag overwrites all three CSVs and the manifest. A
reviewer who only wants to regenerate the manifest (without
re-splitting) should run the splitter with `--force` anyway; the split
is deterministic given the seed and the input, so re-running it
produces byte-identical output.

### 8.6 `Toy-mode accuracy is ~85-90%` warning

**Cause:** not an error. The toy demo is deliberately a smoke test,
not a reproduction of Chapter 4's headline numbers. See README §4.

**Fix:** ignore it. To reach the reported numbers, run the full
pipeline (Flow 1) and train on the full 32 GB corpus, as in README
§5.

### 8.7 `Strict-schema mode: N column(s) would be filled with a sentinel`

**Cause:** `preprocess_doh2023.py --strict-schema` was passed on a
CIC release that is missing one or more columns that the
preprocessor cannot derive from flow-level data (e.g.,
`entropy_subdomain`, `tls_version`). See File 10c's docstring for the
full list of derivable-vs-sentinel columns on the CIC path.

**Fix:** either drop `--strict-schema` (accept the sentinel fills,
which is the default and is what the schema-parity check needs), or
switch to the Zenodo path (`preprocess_zenodo.py`), which is the
training source and produces a schema-complete output with zero
sentinel fills.

---

## 9. Cross-references in this directory

The script files in this directory refer to each other, to the
configs, and to the top-level documents. The map below is the
summary.

| From | To | Why |
|---|---|---|
| `data_utils.py` | `configs/default_pafa_config.yaml` | Schema, split fractions, seed, threat-model observables |
| `download_datasets.py` | `data_utils.py` | `resolve_artifact_root`, `verify_checksums` |
| `download_datasets.py` | `configs/default_pafa_config.yaml` | Dataset DOIs/URLs |
| `download_datasets.py` | `data/make_toy_dataset.py` (via `toy` sub-command) | Toy generation |
| `preprocess_doh2023.py` | `data_utils.py` | Schema, dtype canonicalisation, split fractions |
| `preprocess_doh2023.py` | `configs/default_pafa_config.yaml` | Paths, `threat_model.forbidden_observables` |
| `preprocess_zenodo.py` | `data_utils.py` | Schema, dtype canonicalisation |
| `preprocess_zenodo.py` | `data/preprocess_doh2023.py` | `enforce_forbidden_observables`, `SENTINEL_NUMERIC` |
| `preprocess_zenodo.py` | `configs/default_pafa_config.yaml` | Paths |
| `split_data.py` | `data_utils.py` | `write_split`, `SplitResult`, fraction defaults |
| `split_data.py` | `data/preprocess_doh2023.py` | `enforce_forbidden_observables` |
| `split_data.py` | `configs/default_pafa_config.yaml` | Seed, paths, processed-corpus location |
| `make_toy_dataset.py` | `data_utils.py` | Schema, `validate_schema`, `resolve_artifact_root` |
| `make_toy_dataset.py` | `configs/default_pafa_config.yaml` | Seed |
| `README_data.md` | `README.md`, `REPRODUCIBILITY_CHECKLIST.md`, `LICENSE` | Cross-document navigation |

---

*End of `data/README_data.md`.*
