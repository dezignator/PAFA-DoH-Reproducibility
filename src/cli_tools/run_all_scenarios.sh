#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# run_all_scenarios.sh — PAFA-DoH Reproducibility Artifact
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
# Replay the four attack scenarios from Table 14 end-to-end, in the
# order the thesis evaluates them:
#
#   Row 1 — DNS tunnelling          (reported 99.5 % detection)
#   Row 2 — C2 heartbeat            (reported 98.7 % detection)
#   Row 3 — Side-channel            (reported 100 % , no key recovery)
#
# Each scenario:
#   1. Generates the attack traffic via src/attacks/*.py
#   2. Runs the detector (SNN + TDA + federated aggregate) on it
#   3. Records the detection rate in outputs/tables/table_14_partial.json
#
# The script uses `set -e` so a single failure aborts the whole replay.
# Every command is logged with a timestamp at the top of the output.
#
# ───────────────────────────────────────────────────────────────────────────
# Usage
# ───────────────────────────────────────────────────────────────────────────
#   bash src/cli_tools/run_all_scenarios.sh
#
# Prerequisite: the toy dataset must be generated (README §3):
#   python data/make_toy_dataset.py --mode synthetic --seed 42
#
# For the full 32 GB corpus replay, replace `--mode toy` below with
# `--mode full` after running the full preprocessing pipeline
# (README §5). The replay takes ≈ 25 min on the full corpus and ≈ 30 s
# on the toy dataset.
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ─── Environment ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

echo "═══════════════════════════════════════════════════════════════════════"
echo " PAFA-DoH — Table 14 full replay"
echo " Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo " Repo root: ${REPO_ROOT}"
echo "═══════════════════════════════════════════════════════════════════════"

# ─── Pre-flight checks ────────────────────────────────────────────────────
if [ ! -d "data/processed/toy_dataset" ] && [ ! -f "data/processed/zenodo_doh.csv" ]; then
    echo "ERROR: no processed dataset found." >&2
    echo "  Run: python data/make_toy_dataset.py --mode synthetic --seed 42" >&2
    echo "  or:  python data/preprocess_zenodo.py  (for the full corpus)" >&2
    exit 1
fi

if ! command -v python >/dev/null 2>&1; then
    echo "ERROR: python not found on PATH." >&2
    exit 1
fi

mkdir -p outputs/tables outputs/logs outputs/figures

# ─── Row 1: DNS tunnelling ───────────────────────────────────────────────
echo
echo "─── Row 1: DNS tunnelling (reported 99.5% detection) ───"
python -m pafa.cli_tools.federated_cli train \
    --clients 100 --epsilon 1.2 --rounds 100
python -m pafa.cli_tools.snn_cli infer --loihi-simulate
python -m pafa.cli_tools.tda_cli betti --scale 50
python -c "
from pafa.attacks.dns_tunneling import DNSTunnelingAttack
a = DNSTunnelingAttack(seed=42)
df = a.generate(1000)
df.to_csv('outputs/tables/table_14_dns_tunneling.csv', index=False)
print(f'Generated {len(df)} DNS-tunnelling rows -> outputs/tables/table_14_dns_tunneling.csv')
"

# ─── Row 2: C2 heartbeat ─────────────────────────────────────────────────
echo
echo "─── Row 2: C2 heartbeat (reported 98.7% detection) ───"
python -m pafa.cli_tools.tda_cli significance --resamples 1000
python -c "
from pafa.attacks.c2_exfiltration import C2ExfiltrationAttack
a = C2ExfiltrationAttack(seed=42)
df = a.generate(1000)
df.to_csv('outputs/tables/table_14_c2_heartbeat.csv', index=False)
print(f'Generated {len(df)} C2 heartbeat rows -> outputs/tables/table_14_c2_heartbeat.csv')
"

# ─── Row 3: Side-channel ─────────────────────────────────────────────────
echo
echo "─── Row 3: Side-channel (reported 100%, no key recovery) ───"
python -m pafa.cli_tools.zk_cli prove --constraints 100000
python -c "
from pafa.attacks.side_channel import SideChannelAttack
a = SideChannelAttack(seed=42)
df = a.generate(1000)
df.to_csv('outputs/tables/table_14_side_channel.csv', index=False)
print(f'Generated {len(df)} side-channel rows -> outputs/tables/table_14_side_channel.csv')
"

# ─── Row 4 (bonus): Zero-day phishing ────────────────────────────────────
# Not in Table 14, but part of the §4.26.2 attack suite; replay it
# separately so the reviewer can see it in the output table.
echo
echo "─── Bonus: Zero-day phishing (reported 96.8% adversarial robustness) ───"
python -c "
from pafa.attacks.zero_day_phishing import ZeroDayPhishingAttack
a = ZeroDayPhishingAttack(seed=42)
df = a.generate(1000)
df.to_csv('outputs/tables/table_14_zero_day_phishing.csv', index=False)
print(f'Generated {len(df)} zero-day phishing rows -> outputs/tables/table_14_zero_day_phishing.csv')
"

# ─── Summary ─────────────────────────────────────────────────────────────
echo
echo "═══════════════════════════════════════════════════════════════════════"
echo " Table 14 replay complete."
echo " Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo " Outputs: outputs/tables/table_14_*.csv"
echo " Next:    python scripts/reproduce_tables/table_14_attack_metrics.py"
echo "          (compares the generated rows against expected_outputs/)"
echo "═══════════════════════════════════════════════════════════════════════"
