# `data/` — PAFA-DoH Reproducibility Artifact

This directory is the **input side** of the PAFA-DoH reproducibility artifact (Chapter 4). It provides the dataset downloaders, preprocessors, the seeded train/val/test splitter, the toy-dataset generator, and the shared utilities every data-layer script depends on.

**Full documentation is in [`README_data.md`](README_data.md).** It covers:

- the file inventory (13 files),
- the four data flows (Zenodo training, CIC schema-parity, toy demo, checksum-verify),
- the committed-vs-generated boundary,
- how `CHECKSUMS.txt` works and what a mismatch means,
- a cross-reference to `README.md` and `REPRODUCIBILITY_CHECKLIST.md`,
- and a seven-item troubleshooting guide.

---

## Quick reference

| File | Purpose |
|---|---|
| `data_utils.py` (10a) | 47-column schema, SHA-256 helpers, artifact-root resolver, `write_split()`. Single source of truth for the schema. |
| `download_datasets.py` (10b) | Downloads the two pinned datasets and delegates the `toy` sub-command to the toy generator. |
| `preprocess_zenodo.py` (10d) | Zenodo DoH corpus → 47-column schema, Dask-based. **Training source.** |
| `preprocess_doh2023.py` (10c) | CIC-DoH2023 → 47-column schema. **Schema-parity check**, not the training source. |
| `split_data.py` (10e) | Seeded, stratified train/val/test split. Deterministic given the seed. |
| `make_toy_dataset.py` (10f) | 2,000-row synthetic toy dataset for the two-minute smoke test. |
| `toy_dataset/*.csv` (10h–10i) | Header-only schema templates. **Committed; never overwritten.** |
| `CHECKSUMS.txt` (10j) | SHA-256 manifest for the two downloaded datasets. |
| `raw/` | Downloaded datasets. **Not committed** (empty in a fresh clone; see `.gitkeep`). |
| `processed/` | Preprocessor and splitter output. **Not committed** (empty in a fresh clone; see `.gitkeep`). |

---

## Where to start

- **Just want to see the pipeline run?** README.md §3 (two-minute toy demo). `python data/make_toy_dataset.py --mode synthetic --seed 42` then `python scripts/train_pafa.py --config configs/default_pafa_config.yaml --mode toy`.
- **Reproducing the reported numbers?** README.md §5 (full reproduction). Start with `python data/download_datasets.py zenodo`.

---

## License note

The code in this directory is MIT-licensed (see [`../LICENSE`](../LICENSE)). The two datasets are CC-BY 4.0 upstream and are **not redistributed** by this repository (LICENSE §2). The toy CSVs `data/toy_dataset/*.csv` are CC-BY 4.0, header-only, and committed as schema templates.
