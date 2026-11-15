#!/usr/bin/env python3
"""Arithmetic cross-checks on the numbers quoted in Chapter 4.

Exit status 1 if any hard check fails. 'note' lines are discrepancies between
a stated figure and the value implied by its inputs; they are not failures but
should be explained wherever the figure is quoted.
"""
import math, sys

results = []


def check(name, ok, detail, hard=True):
    results.append((name, ok, detail, hard))


# Table 13 gains
check("latency gain 2.1/0.8", abs(2.1 / 0.8 - 2.6) < 0.05, f"{2.1/0.8:.3f} vs stated 2.6x")
check("energy gain 1.8/0.12", abs(1.8 / 0.12 - 15) < 1e-9, f"{1.8/0.12:.3f} vs stated 15x")
# Sec 4.16 margin
margin = 500 - 432
check("proof-gen margin", margin == 68 and abs(margin / 500 - 0.14) < 0.005,
      f"{margin} ms = {100*margin/500:.1f}% of target (stated ~14%)")
# Table 15 targets
check("proof gen < 500", 432 < 500, "432 < 500")
check("proof ver < 100", 78 < 100, "78 < 100")
check("proof size <= 2 KB", 1.8 <= 2.0, "1.8 <= 2.0")
check("soundness 2^-132 <= 2^-128", -132 <= -128, "exponent -132 <= -128")
# Table 16 targets
check("accuracy >= target", 99.4 >= 99.2, "99.4 >= 99.2")
check("epochs <= 10", 8 <= 10, "8 <= 10")
check("dropout >= 95", 97.3 >= 95, "97.3 >= 95")
check("robustness >= 95", 96.8 >= 95, "96.8 >= 95")
check("eps <= 1.5", 1.2 <= 1.5, "1.2 <= 1.5")
# Bonferroni (Sec 4.8.1): three features
thr = 0.05 / 3
check("Bonferroni p=0.002 passes alpha/3", 0.002 <= thr, f"alpha/3 = {thr:.4f}")
check("Bonferroni p<0.001 passes alpha/3", 0.001 <= thr, f"alpha/3 = {thr:.4f}")
# Loihi size (Sec 4.24.1): stated 128000 neurons from 128 cores x 1024
n_neurons = 128 * 1024
check("Loihi neuron count", n_neurons == 128000,
      f"128 x 1024 = {n_neurons}; text states 128000 (rounded)", hard=False)
check("TrueNorth neuron count", 4096 * 256 == 1048576, f"4096 x 256 = {4096*256}")
# rare-event resolution of 'FPR < 0.01 %'
need = math.ceil(3 / 1e-4)
check("FPR resolution", True,
      f"observing 0 FP certifies FPR<=0.01% (95%) only with >= ~{need} benign samples; "
      "record the benign count in run_manifest_v2.yaml", hard=False)
# accuracy figures that coexist in the text
check("tunnelling accuracy vs detection rate", True,
      "99.6% (accuracy, Sec 4.27.1/4.30), 99.5% (detection rate, Table 14), 99.2% (aggregate): "
      "state which metric each is", hard=False)
# stated ASCII time scale
check("event-driven latency wording", True,
      "Sec 4.12 says 'microseconds'; Table 13 reports 0.8 ms per inference", hard=False)

fails = 0
for n, ok, d, hard in results:
    tag = "ok  " if ok and hard else ("note" if not hard else "FAIL")
    if not ok and hard:
        fails += 1
    print(f"[{tag}] {n:40s} {d}")
sys.exit(1 if fails else 0)
