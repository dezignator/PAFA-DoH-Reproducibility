#!/usr/bin/env python3
"""End-to-end exercise of Secs 4.7-4.8 on SYNTHETIC flows.

flows -> point cloud -> windows -> VR persistence -> (beta_1, entropy, W_2)
-> permutation tests (Eq. 42) -> Bonferroni over three features.

Output is a pipeline check. It is not Table 17; those values come from the
Zenodo DoH data and the ns-3/dnscat2 testbed.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pafa_v2 import flows_v2 as fl, tda_v2 as tda, stats_v2 as st


def features(gen, n_win, width, rng, eps, max_scale, ref, representation="point"):
    rows = []
    for _ in range(n_win):
        if representation == "point":              # Sec 4.7.1 cloud
            ts, size, names = gen(width, rng)
            X = fl.flow_to_cloud(ts, size, names)
        else:                                      # delay embedding of query counts
            X = fl.delay_embed(gen(width, rng), 6, 1)
        rows.append(tda.summarize_window(X, eps, max_scale, ref))
    return np.array(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=30)
    ap.add_argument("--width", type=int, default=40)
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--max-scale", type=float, default=2.5)
    ap.add_argument("--resamples", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260101)
    ap.add_argument("--representation", choices=["point", "delay"], default="point",
                    help="point = Sec 4.7.1 cloud; delay = sliding-window embedding of per-bin query counts (comparison only)")
    ap.add_argument("--out", default="tda_pipeline_demo_v2.json")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    # reference diagram from a benign split disjoint from the evaluation windows
    if a.representation == "point":
        ref_pts = fl.flow_to_cloud(*fl.synth_benign(a.width, rng))
        gb, ga = fl.synth_benign, (lambda n, r: fl.synth_tunnel(n, r))
    else:
        ref_pts = fl.delay_embed(fl.synth_benign_counts(a.width, rng), 6, 1)
        gb, ga = fl.synth_benign_counts, fl.synth_beacon_bursty
    ref = tda.vr_persistence(ref_pts, a.max_scale)[1]
    ben = features(gb, a.windows, a.width, rng, a.eps, a.max_scale, ref, a.representation)
    atk = features(ga, a.windows, a.width, rng, a.eps, a.max_scale, ref, a.representation)
    names = ["betti1", "persistence_entropy", "wasserstein_to_benign_ref"]
    res = {"seed": a.seed, "windows": a.windows, "width": a.width, "eps": a.eps,
           "max_scale": a.max_scale, "resamples": a.resamples, "representation": a.representation, "features": {}}
    pv = []
    for j, nm in enumerate(names):
        p, t = st.permutation_pvalue(ben[:, j], atk[:, j], n_resamples=a.resamples, seed=a.seed + j)
        pv.append(p)
        res["features"][nm] = {"benign_mean": float(ben[:, j].mean()), "benign_sd": float(ben[:, j].std(ddof=1)),
                               "attack_mean": float(atk[:, j].mean()), "attack_sd": float(atk[:, j].std(ddof=1)),
                               "p": p}
    thr, rej = st.bonferroni(pv, 0.05)
    res["bonferroni_threshold"] = thr
    for nm, r in zip(names, rej):
        res["features"][nm]["reject_after_bonferroni"] = bool(r)
    res["note"] = "synthetic flows; pipeline check only"
    Path(a.out).write_text(json.dumps(res, indent=2))
    for nm in names:
        f = res["features"][nm]
        print(f"{nm:28s} benign {f['benign_mean']:.3f}+-{f['benign_sd']:.3f}  "
              f"attack {f['attack_mean']:.3f}+-{f['attack_sd']:.3f}  p={f['p']:.4g}")
    print(f"bonferroni alpha/3 = {thr:.4f}")


if __name__ == "__main__":
    main()
