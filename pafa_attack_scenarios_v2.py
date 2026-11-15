#!/usr/bin/env python3
"""PAFA-DoH Chapter 4, attack-scenario and detection experiment suite (v2).

Simulates the detection side of the framework (Secs 4.7-4.14, 4.26, 4.30) on
SYNTHETIC traffic and produces Table 14 / Table 17 / Sec 4.30 style tables:

  A1  scenario catalogue        what each simulated attack does
  A2  detector comparison       rule-based, statistical, supervised ML, topological (TDA),
                                spiking (LIF+STDP), and the PAFA fusion; every detector is
                                calibrated to the SAME benign false-positive rate (1 %) on a
                                separate benign calibration set; TPR with Wilson 95 % CIs
  A3  ablation                  PAFA fusion with one component removed
  A4  topological features      Betti-1 / persistence entropy / Wasserstein, benign vs attack,
                                permutation test + Bonferroni (Table 17 style)
  A5  multi-phase lifecycle     14 h session: stealth set-up -> C2 beaconing -> query-flood
                                exfiltration -> subdomain enumeration; per-phase detection
  A6  side-channel hardening    simulated CPA key recovery against Hamming-weight leakage with
                                no / static / SNN-adaptive noise injection (Eqs. 47-48)
  A7  cost per window           measured CPU time of every detector

Data: pafa_v2/windows_v2.py. The scenarios, their parameters and the 'mimic'
family are my construction. They are not the dnscat2/ns-3/Zenodo testbed of the
thesis, and the CPA experiment is a software model of leakage, not ChipWhisperer
captures. The tables show how the *components* behave when every detector faces
the same traffic at the same false-positive budget; they are not Table 14/17.

usage:
    python3 experiments/pafa_attack_scenarios_v2.py --out build/attack
    python3 experiments/pafa_attack_scenarios_v2.py --quick
"""
import argparse
import json
import warnings
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pafa_v2 import windows_v2 as wv                              # noqa: E402
from pafa_v2 import flows_v2 as fl, tda_v2 as tda, snn_v2 as snn   # noqa: E402
from pafa_v2 import stats_v2 as st, metrics_v2 as mt              # noqa: E402
from pafa_v2.sidechannel_v2 import adaptive_threshold             # noqa: E402

FPR_BUDGET = 0.01
ATTACKS = ["tunnel", "beacon", "slow_exfil", "mimic"]


# ==========================================================================
# extra scenario used only by the lifecycle experiment: subdomain enumeration
# ==========================================================================
def gen_enum(rng, w=wv.W):
    ts = np.cumsum(rng.exponential(rng.uniform(0.05, 0.3), w))
    start = int(rng.integers(0, 500))
    names = [f"host{start + i}.corp{rng.integers(0, 2)}.victim.org" for i in range(w)]
    size = rng.normal(95, 5, w)
    qt = ["A"] * w
    return ts, np.clip(size, 40, 512), names, qt


def gen_any(kind, rng):
    return gen_enum(rng) if kind == "enum" else wv.gen_window(kind, rng)


# ==========================================================================
# TDA features from the inter-arrival series (delay embedding, Secs 4.7-4.8)
# ==========================================================================
EPS_BETTI = 0.5
MAX_SCALE = 1.2


def tda_window(win):
    ts = win[0]
    x = wv.iat_series(ts)
    x = (x - x.mean()) / (x.std() + 1e-9)
    E = fl.delay_embed(x, 4, 2)
    dg = tda.vr_persistence(E, max_scale=MAX_SCALE)[1]
    f = tda.finite(dg)
    life = (f[:, 1] - f[:, 0]) if len(f) else np.array([0.0])
    return dg, np.array([tda.betti_at(dg, EPS_BETTI), tda.persistence_entropy(dg), life.max(), len(f),
                         life.sum()])


TDA_NAMES = ["betti1", "persistence_entropy", "max_lifetime", "n_bars", "total_persistence"]


# ==========================================================================
# spiking detector: log-burst encoding -> LIF layer with lateral inhibition,
# unsupervised pair-STDP on benign windows (Secs 4.9-4.11)
# ==========================================================================
class SNNDetector:
    NH, NCH, T_STEPS, SEG = 16, 3, 520, 3
    TAU_M, TAU_S, V_TH, INH, T_REF = 10.0, 5.0, 1.0, 0.35, 2

    def __init__(self, seed=0, tau_enc=2.0, alpha=0.5, n_burst=3, eta=0.02, w_max=1.6):
        self.rng = np.random.default_rng(seed)
        self.tau_enc, self.alpha, self.n_burst, self.eta, self.w_max = tau_enc, alpha, n_burst, eta, w_max
        self.W = self.rng.uniform(0.3, 1.0, (self.NCH, self.NH))
        self.mu = self.sd = None

    def _zfit(self, wins):
        F = np.vstack([self._chan_raw(w) for w in wins])
        self.mu, self.sd = F.mean(0), F.std(0) + 1e-9

    @staticmethod
    def _chan_raw(win):
        ts, size, names, _ = win
        iat = np.diff(ts, prepend=ts[0] - 1.0)
        ent = np.array([fl.label_entropy(n.split(".")[0]) for n in names])
        return np.column_stack([np.log(iat + 1e-3), size, ent])

    def encode(self, win):
        """Per-channel spike steps (1 ms grid), Eq. (43) with feature magnitude z^2."""
        Z = (self._chan_raw(win) - self.mu) / self.sd
        trains = []
        for c in range(self.NCH):
            t_arr = 10.0 * np.arange(len(Z)) + 5.0
            t, _ = snn.log_burst_encode(t_arr, Z[:, c] ** 2, self.tau_enc, self.alpha, self.n_burst)
            steps = np.clip(np.floor(t).astype(int), 0, self.T_STEPS - 1)
            trains.append(steps)
        return trains

    def run(self, trains, learn=False):
        T, NH, NCH = self.T_STEPS, self.NH, self.NCH
        inp = np.zeros((T, NCH))
        for c, s in enumerate(trains):
            np.add.at(inp[:, c], s, 1.0)
        V = np.zeros(NH); syn = np.zeros(NH); ref = np.zeros(NH)
        dec = np.exp(-1.0 / self.TAU_S)
        out_times = [[] for _ in range(NH)]
        for t in range(T):
            syn = syn * dec + inp[t] @ self.W
            V += (-V + 2.0 * syn) / self.TAU_M
            V[ref > 0] = 0.0
            ref = np.maximum(ref - 1, 0)
            f = V >= self.V_TH
            if f.any():
                for j in np.nonzero(f)[0]:
                    out_times[j].append(t)
                V[f] = 0.0
                ref[f] = self.T_REF
                V[~f] -= self.INH * f.sum()
                V = np.maximum(V, -1.0)
        if learn:
            for c in range(NCH):
                pre = trains[c]
                for j in range(NH):
                    if out_times[j] and len(pre):
                        dw = snn.stdp_delta_w(pre.astype(float), out_times[j], self.eta, 20.0, 20.0)
                        self.W[c, j] = snn.apply_bounded(self.W[c, j], dw / max(1.0, len(pre) / 20), 0.0, self.w_max)
        counts = np.zeros((self.SEG, NH))
        edges = np.linspace(0, T, self.SEG + 1)
        for j in range(NH):
            if out_times[j]:
                counts[:, j] = np.histogram(out_times[j], bins=edges)[0]
        return counts.ravel()

    def fit(self, benign_wins, train_n=250):
        self._zfit(benign_wins)
        for w in benign_wins[:train_n]:
            self.run(self.encode(w), learn=True)
        R = np.array([self.run(self.encode(w)) for w in benign_wins[train_n:train_n + 400]])
        self.m = R.mean(0)
        C = np.cov(R.T) + 1e-1 * np.eye(R.shape[1])
        self.Ci = np.linalg.inv(C)

    def score(self, win):
        r = self.run(self.encode(win))
        d = r - self.m
        return float(np.sqrt(d @ self.Ci @ d))


# ==========================================================================
# detectors (all return an anomaly score; larger = more anomalous)
# ==========================================================================
def rule_score(Xraw):
    """Static thresholds fixed in advance (not tuned on any data):
    flag long high-entropy labels, or TXT-heavy traffic with high entropy."""
    i = {n: k for k, n in enumerate(wv.FEATURE_NAMES)}
    s1 = (Xraw[:, i["ent_mean"]] - 3.5) + 0.5 * (Xraw[:, i["label_len"]] - 2.0)
    s2 = (Xraw[:, i["txt_frac"]] - 0.3) + (Xraw[:, i["ent_mean"]] - 3.0)
    return np.maximum(s1, s2)


class Mahalanobis:
    def fit(self, X):
        self.mu = X.mean(0)
        self.Ci = np.linalg.inv(np.cov(X.T) + 1e-3 * np.eye(X.shape[1]))
        return self

    def score(self, X):
        d = X - self.mu
        return np.sqrt(np.einsum("ij,jk,ik->i", d, self.Ci, d))


class Logistic:
    def fit(self, X, y, iters=800, lr=0.3, l2=1e-3):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        w = np.zeros(Z.shape[1]); b = 0.0
        for _ in range(iters):
            p = 1 / (1 + np.exp(-(Z @ w + b)))
            w -= lr * (Z.T @ (p - y) / len(y) + l2 * w); b -= lr * (p - y).mean()
        self.w, self.b = w, b
        return self

    def score(self, X):
        return ((X - self.mu) / self.sd) @ self.w + self.b


def threshold_at_fpr(benign_scores, fpr=FPR_BUDGET):
    return float(np.quantile(benign_scores, 1 - fpr))


def tpr_ci(scores, thr):
    k = int((scores > thr).sum()); n = len(scores)
    lo, hi = mt.wilson(k, n)
    return k / n, lo, hi


def auc(neg, pos):
    s = np.concatenate([neg, pos]); y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    return mt.auc(s, y)


# ==========================================================================
# build all data once
# ==========================================================================
def build(cfg, rng):
    t0 = time.time()
    n = cfg["n"]
    D = {}
    def gen(kind, m):
        return [gen_any(kind, rng) for _ in range(m)]
    D["ben_snn"] = gen("benign", 650)               # SNN training / reference
    D["ben_cal"] = gen("benign", n["cal"])          # threshold calibration (FPR budget)
    D["ben_test"] = gen("benign", n["ben_test"])
    D["ben_sup"] = gen("benign", n["sup"])
    D["sup_tunnel"] = gen("tunnel", n["sup"] // 2)  # supervised baseline sees only these two
    D["sup_beacon"] = gen("beacon", n["sup"] // 2)
    for a in ATTACKS:
        D[f"test_{a}"] = gen(a, n["atk"])
    print(f"  generated windows in {time.time()-t0:.1f}s")
    return D


def feats(wins):
    return np.array([wv.window_features(*w) for w in wins])


def tda_feats(wins):
    dgs, F = [], []
    for w in wins:
        d, f = tda_window(w)
        dgs.append(d); F.append(f)
    return dgs, np.array(F)


# ==========================================================================
# experiments
# ==========================================================================
def catalogue():
    rows = [
        ("benign", "browsing bursts (73 %), periodic telemetry polling (15 %), CDN hash labels (12 %)", "-"),
        ("tunnel", "dnscat2-like: 28-52 char base32 labels, TXT/CNAME/MX heavy, ~0.15-1 s spacing", "4.26.1"),
        ("beacon", "C2 heartbeat: 2-10 s period, 4-15 % jitter, short hex labels, constant small payload", "4.26.4"),
        ("slow_exfil", "low-and-slow: 12-45 s mean spacing, 18-30 char labels, TXT heavy", "4.26.1"),
        ("mimic", "benign-looking names/sizes/types, hidden cyclic rate modulation (zero-day style)", "4.2.1, 4.26.2"),
        ("enum", "subdomain enumeration host0..hostN, sequential, fast (lifecycle phase 4 only)", "4.30.4"),
    ]
    return pd.DataFrame(rows, columns=["scenario", "behaviour (simulated)", "thesis section"])


def a2_detectors(D, cfg, out):
    t0 = time.time()
    Xb_cal, Xb_te = feats(D["ben_cal"]), feats(D["ben_test"])
    Xtest = {a: feats(D[f"test_{a}"]) for a in ATTACKS}
    print(f"  window features {time.time()-t0:.1f}s"); t1 = time.time()

    # TDA features / diagrams
    dg_cal, T_cal = tda_feats(D["ben_cal"]); dg_te, T_te = tda_feats(D["ben_test"])
    dg_a, T_a = {}, {}
    for a in ATTACKS:
        dg_a[a], T_a[a] = tda_feats(D[f"test_{a}"])
    # medoid benign reference diagram for the Wasserstein score
    mu_t, sd_t = T_cal.mean(0), T_cal.std(0) + 1e-9
    zc = np.abs((T_cal - mu_t) / sd_t).sum(1)
    ref_idx = int(np.argsort(zc)[len(zc) // 2])
    ref_dg = dg_cal[ref_idx]
    wass = lambda dgs: np.array([tda.wasserstein(d, ref_dg, p=2.0) for d in dgs])
    W_cal, W_te = wass(dg_cal), wass(dg_te)
    W_a = {a: wass(dg_a[a]) for a in ATTACKS}
    print(f"  TDA {time.time()-t1:.1f}s"); t2 = time.time()

    def tda_score(Tm, Wv):
        z = np.abs((Tm - mu_t) / sd_t)
        return z[:, 1] + z[:, 2] + z[:, 3] + (Wv - W_cal.mean()) / (W_cal.std() + 1e-9)

    # SNN
    snn_det = SNNDetector(seed=cfg["seed"])
    snn_det.fit(D["ben_snn"])
    S_cal = np.array([snn_det.score(w) for w in D["ben_cal"]])
    S_te = np.array([snn_det.score(w) for w in D["ben_test"]])
    S_a = {a: np.array([snn_det.score(w) for w in D[f"test_{a}"]]) for a in ATTACKS}
    print(f"  SNN {time.time()-t2:.1f}s")

    # statistical + supervised
    Xb_sup = feats(D["ben_sup"])
    maha = Mahalanobis().fit(Xb_sup)
    Xs = np.vstack([Xb_sup, feats(D["sup_tunnel"]), feats(D["sup_beacon"])])
    ys = np.r_[np.zeros(len(Xb_sup)), np.ones(len(Xs) - len(Xb_sup))]
    sup = Logistic().fit(Xs, ys)

    det = {}
    det["Rule-based (static thresholds)"] = (rule_score(Xb_cal), rule_score(Xb_te), {a: rule_score(Xtest[a]) for a in ATTACKS})
    det["Statistical (Mahalanobis, benign only)"] = (maha.score(Xb_cal), maha.score(Xb_te), {a: maha.score(Xtest[a]) for a in ATTACKS})
    det["Supervised ML (trained on tunnel+beacon)"] = (sup.score(Xb_cal), sup.score(Xb_te), {a: sup.score(Xtest[a]) for a in ATTACKS})
    det["TDA (persistence, delay-embedded IAT)"] = (tda_score(T_cal, W_cal), tda_score(T_te, W_te), {a: tda_score(T_a[a], W_a[a]) for a in ATTACKS})
    det["SNN (LIF + STDP, benign-trained)"] = (S_cal, S_te, S_a)

    # PAFA fusion: each component normalised by its benign-calibration 99th percentile,
    # fused by max (any component can raise the alarm). No attack labels are used.
    comps = ["Statistical (Mahalanobis, benign only)", "TDA (persistence, delay-embedded IAT)", "SNN (LIF + STDP, benign-trained)"]
    ref = {c: (np.median(det[c][0]), np.quantile(det[c][0], 0.99)) for c in comps}
    def fuse(which, key, a=None):
        parts = []
        for c in which:
            s = det[c][key] if a is None else det[c][2][a]
            m, q = ref[c]
            parts.append((s - m) / (q - m + 1e-9))
        return np.max(parts, axis=0)
    det["PAFA fusion (stat + TDA + SNN, max)"] = (fuse(comps, 0), fuse(comps, 1), {a: fuse(comps, 2, a) for a in ATTACKS})
    ablate = {}
    for drop in comps:
        rest = [c for c in comps if c != drop]
        ablate["fusion without " + drop.split(" (")[0]] = (fuse(rest, 0), fuse(rest, 1), {a: fuse(rest, 2, a) for a in ATTACKS})
    state = dict(det=det, ablate=ablate, maha=maha, snn=snn_det, sup=sup, ref_dg=ref_dg,
                 T_cal=T_cal, T_te=T_te, T_a=T_a, W_cal=W_cal, W_te=W_te, W_a=W_a,
                 mu_t=mu_t, sd_t=sd_t, tda_score=tda_score, comps=comps, fuse=fuse)

    def table(detmap):
        rows = []
        for name, (cal, te, atk) in detmap.items():
            thr = threshold_at_fpr(cal)
            fpr_k = int((te > thr).sum()); fpr_lo, fpr_hi = mt.wilson(fpr_k, len(te))
            row = {"detector": name, "FPR (benign test)": f"{fpr_k/len(te):.3%} [{fpr_lo:.2%},{fpr_hi:.2%}]"}
            tprs = []
            for a in ATTACKS:
                t, lo, hi = tpr_ci(atk[a], thr)
                tprs.append(t)
                row[f"TPR {a}"] = f"{t:.1%} [{lo:.0%},{hi:.0%}]"
            row["mean TPR"] = f"{np.mean(tprs):.1%}"
            row["mean AUC"] = f"{np.mean([auc(te, atk[a]) for a in ATTACKS]):.3f}"
            rows.append(row)
        return pd.DataFrame(rows)

    return table(det), table(ablate), state


def a4_topology(state, cfg):
    T_te, T_a, W_te, W_a = state["T_te"], state["T_a"], state["W_te"], state["W_a"]
    rows = []
    names = ["betti1", "persistence_entropy", "wasserstein_to_ref"]
    for a in ATTACKS:
        ben = [T_te[:, 0], T_te[:, 1], W_te]
        atk = [T_a[a][:, 0], T_a[a][:, 1], W_a[a]]
        pv = []
        for j, nm in enumerate(names):
            p, _ = st.permutation_pvalue(ben[j], atk[j], n_resamples=cfg["resamples"], seed=cfg["seed"] + j)
            pv.append(p)
        thr, rej = st.bonferroni(pv, 0.05)
        for j, nm in enumerate(names):
            rows.append({"scenario": a, "feature": nm,
                         "benign mean ± sd": f"{ben[j].mean():.2f} ± {ben[j].std(ddof=1):.2f}",
                         "attack mean ± sd": f"{atk[j].mean():.2f} ± {atk[j].std(ddof=1):.2f}",
                         "cohen_d": f"{st.cohens_d(atk[j], ben[j]):+.2f}",
                         "p (perm)": f"{pv[j]:.4f}", "reject @ alpha/3": "yes" if rej[j] else "no"})
    return pd.DataFrame(rows)


def a5_lifecycle(state, cfg, rng):
    """14 h session. Per hour: 20 windows, each from the phase's attack with prob 0.7, else benign."""
    phases = [(0, 4, "mimic", "stealth set-up"), (4, 7, "beacon", "C2 beaconing"),
              (7, 11, "tunnel", "query-flood exfiltration"), (11, 14, "enum", "subdomain enumeration")]
    det = state["det"]
    thr = {n: threshold_at_fpr(v[0]) for n, v in det.items()}
    snn_det, maha, sup, tda_score = state["snn"], state["maha"], state["sup"], state["tda_score"]
    mu_t, sd_t, ref_dg = state["mu_t"], state["sd_t"], state["ref_dg"]
    W_cal = state["W_cal"]
    comps, fuse = state["comps"], state["fuse"]
    rows, timeline = [], []
    for h in range(14):
        kind = [p[2] for p in phases if p[0] <= h < p[1]][0]
        wins, lab = [], []
        for _ in range(cfg["lc_per_hour"]):
            if rng.random() < 0.7:
                wins.append(gen_any(kind, rng)); lab.append(1)
            else:
                wins.append(gen_any("benign", rng)); lab.append(0)
        X = feats(wins)
        dg, Tm = tda_feats(wins)
        Wv = np.array([tda.wasserstein(d, ref_dg, 2.0) for d in dg])
        Sv = np.array([snn_det.score(w) for w in wins])
        sc = {"Rule-based (static thresholds)": rule_score(X),
              "Statistical (Mahalanobis, benign only)": maha.score(X),
              "Supervised ML (trained on tunnel+beacon)": sup.score(X),
              "TDA (persistence, delay-embedded IAT)": tda_score(Tm, Wv),
              "SNN (LIF + STDP, benign-trained)": Sv}
        # fused score
        ref = {c: (np.median(det[c][0]), np.quantile(det[c][0], 0.99)) for c in comps}
        parts = [(sc[c] - ref[c][0]) / (ref[c][1] - ref[c][0] + 1e-9) for c in comps]
        sc["PAFA fusion (stat + TDA + SNN, max)"] = np.max(parts, axis=0)
        lab = np.array(lab)
        row = {"hour": h + 1, "phase": kind}
        for n, s in sc.items():
            flag = s > thr[n]
            row[n] = float(flag[lab == 1].mean()) if (lab == 1).any() else np.nan
            row[n + "|fp"] = float(flag[lab == 0].mean()) if (lab == 0).any() else np.nan
        rows.append(row)
        timeline.append((h + 1, kind, float(np.mean(sc["PAFA fusion (stat + TDA + SNN, max)"][lab == 1])),
                         float(np.mean(sc["PAFA fusion (stat + TDA + SNN, max)"][lab == 0]))))
    df = pd.DataFrame(rows)
    names = [n for n in det.keys()]
    out = []
    for lo, hi, kind, label in phases:
        sub = df[(df.hour > lo) & (df.hour <= hi)]
        r = {"phase": f"{label} (h{lo+1}-{hi}, {kind})"}
        for n in names:
            r[n] = f"{100*np.nanmean(sub[n]):.1f}%"
        out.append(r)
    r = {"phase": "benign false-alarm rate (all hours)"}
    for n in names:
        r[n] = f"{100*np.nanmean(df[n + '|fp']):.1f}%"
    out.append(r)
    return pd.DataFrame(out), timeline


# ---- A6: side channel ----------------------------------------------------
def _sbox():
    # AES S-box from GF(2^8) inverse + affine map
    def gmul(a, b):
        p = 0
        for _ in range(8):
            if b & 1: p ^= a
            hi = a & 0x80
            a = (a << 1) & 0xFF
            if hi: a ^= 0x1B
            b >>= 1
        return p
    inv = [0] * 256
    for a in range(1, 256):
        for b in range(1, 256):
            if gmul(a, b) == 1:
                inv[a] = b; break
    sb = []
    for x in range(256):
        b = inv[x]
        s = b ^ ((b << 1) | (b >> 7)) & 0xFF ^ ((b << 2) | (b >> 6)) & 0xFF ^ ((b << 3) | (b >> 5)) & 0xFF ^ ((b << 4) | (b >> 4)) & 0xFF ^ 0x63
        sb.append(s & 0xFF)
    return np.array(sb, dtype=np.uint8)


SBOX = _sbox()
HW = np.array([bin(i).count("1") for i in range(256)], dtype=float)


def cpa_success(n_traces, noise_sd, rng, trials, key_byte=0x2B, gain=1.0):
    """Fraction of trials in which first-round CPA ranks the true key byte first."""
    ok = 0
    for _ in range(trials):
        pt = rng.integers(0, 256, n_traces).astype(np.uint8)
        leak = gain * HW[SBOX[pt ^ key_byte]] + rng.normal(0, noise_sd, n_traces)
        H = HW[SBOX[pt[:, None] ^ np.arange(256, dtype=np.uint8)[None, :]]]
        Hc = H - H.mean(0); lc = leak - leak.mean()
        corr = (Hc * lc[:, None]).sum(0) / (np.linalg.norm(Hc, axis=0) * np.linalg.norm(lc) + 1e-12)
        ok += int(np.argmax(np.abs(corr)) == key_byte)
    return ok / trials


def a6_sidechannel(cfg, rng):
    base_sd = 1.0
    gain = 1.0
    sig_norm = np.sqrt(np.var(HW[SBOX[np.arange(256, dtype=np.uint8)]]) * 1000)     # ||s|| over 1000 samples (leak std ~1.41)
    levels = [("none (measurement noise only)", 0.0),
              ("static injection, amplitude x2.5", 2.5),
              ("static injection, amplitude x8", 8.0),
              ("static injection, amplitude x25", 25.0),
              ("static injection, amplitude x50", 50.0)]
    n_grid = [25, 50, 100, 200, 400, 800, 1600, 3200]
    rows = []
    for name, mult in levels:
        sd = base_sd * np.sqrt(1 + mult ** 2) if mult else base_sd
        ratio = sig_norm / (sd * np.sqrt(1000))
        ratio0 = sig_norm / (base_sd * np.sqrt(1000))
        row = {"countermeasure": name, "total noise sd": f"{sd:.2f}",
               "||s||/||n|| (Eq. 47)": f"{ratio:.3f}",
               "ratio reduction": f"{100*(1-ratio/ratio0):.1f}%"}
        for n in n_grid:
            row[f"n={n}"] = f"{cpa_success(n, sd, rng, cfg['cpa_trials']):.2f}"
        rows.append(row)
    # adaptive: noise only while the detector's threat level T(t) is high.
    # attack present for fraction q of traces (T=1 there), benign otherwise (T=0).
    # V_th(t) = V_th0 + beta*T(t) (Eq. 48) sets the controller firing, modelled here as
    # injection amplitude m_hi when T high and m_lo when low.
    vth = adaptive_threshold(1.0, 0.8, np.array([0.0, 1.0]))
    m_hi, m_lo = 50.0, 2.5
    adaptive = {"countermeasure": "SNN-adaptive: x50 while T(t)~1, x2.5 otherwise",
                "total noise sd": f"{base_sd*np.sqrt(1+m_hi**2):.2f} (alert) / {base_sd*np.sqrt(1+m_lo**2):.2f} (idle)",
                "||s||/||n|| (Eq. 47)": f"{sig_norm/(base_sd*np.sqrt(1+m_hi**2)*np.sqrt(1000)):.3f} (alert)",
                "ratio reduction": f"{100*(1-1/np.sqrt(1+m_hi**2)):.1f}% (alert)"}
    for n in n_grid:
        adaptive[f"n={n}"] = f"{cpa_success(n, base_sd*np.sqrt(1+m_hi**2), rng, cfg['cpa_trials']):.2f} (alert)"
    rows.append(adaptive)
    df = pd.DataFrame(rows)
    # cost: mean injected noise power over a day with attack duty cycle q
    cost = []
    for q in [0.01, 0.05, 0.2]:
        static_hi = m_hi ** 2
        adapt = q * m_hi ** 2 + (1 - q) * m_lo ** 2
        cost.append({"attack duty cycle": f"{q:.0%}", "static x50 injected power": f"{static_hi:.0f}",
                     "adaptive injected power": f"{adapt:.0f}", "saving": f"{100*(1-adapt/static_hi):.1f}%",
                     "V_th idle/alert (Eq. 48)": f"{vth[0]:.2f}/{vth[1]:.2f}"})
    return df, pd.DataFrame(cost)


# ---- A7: cost --------------------------------------------------------------
def a7_cost(D, state):
    wins = D["ben_test"][:60]
    X = feats(wins)
    def tm(f, reps=1):
        t = time.perf_counter()
        for _ in range(reps): f()
        return (time.perf_counter() - t) / (reps * len(wins)) * 1e6
    rows = []
    rows.append(("Window feature extraction (17 feats)", tm(lambda: [wv.window_features(*w) for w in wins])))
    rows.append(("Rule-based", tm(lambda: rule_score(X), 50)))
    rows.append(("Mahalanobis", tm(lambda: state["maha"].score(X), 50)))
    rows.append(("Supervised logistic", tm(lambda: state["sup"].score(X), 50)))
    rows.append(("TDA: embed + H0/H1 + Wasserstein", tm(lambda: [tda.wasserstein(tda_window(w)[0], state["ref_dg"]) for w in wins])))
    rows.append(("SNN: encode + 520-step LIF sim (python)", tm(lambda: [state["snn"].score(w) for w in wins])))
    return pd.DataFrame(rows, columns=["stage", "us per window (CPU, 1 core)"]).round(1)


# ---- plots -----------------------------------------------------------------
def plots(state, timeline, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    det = state["det"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, a in zip(axs, ["mimic", "slow_exfil"]):
        for name, (cal, te, atk) in det.items():
            g, T, F = mt.roc(np.concatenate([te, atk[a]]), np.r_[np.zeros(len(te)), np.ones(len(atk[a]))])
            ax.plot(F, T, label=name.split(" (")[0], lw=1.4)
        ax.plot([0, 1], [0, 1], "k:", lw=.6); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(f"ROC: {a}")
        ax.grid(alpha=.3)
    axs[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "a2_roc.png", dpi=140); plt.close(fig)
    h = [t[0] for t in timeline]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(h, [t[2] for t in timeline], "o-", label="fused score, attack windows")
    ax.plot(h, [t[3] for t in timeline], "s--", label="fused score, benign windows")
    ax.axhline(1.0, color="r", lw=.7, ls=":", label="~benign 99th pct")
    for x, lab in [(0.5, "stealth"), (4.5, "C2"), (7.5, "exfil"), (11.5, "enum")]:
        ax.text(x, ax.get_ylim()[1] * .92, lab, fontsize=8)
    ax.set_xlabel("hour"); ax.set_ylabel("normalised score"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax.set_title("A5 lifecycle (synthetic)")
    fig.tight_layout(); fig.savefig(out / "a5_lifecycle.png", dpi=140); plt.close(fig)


def md(df, title):
    return f"### {title}\n\n{df.to_markdown(index=False)}\n"


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="build/attack")
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    cfg = dict(seed=a.seed, resamples=3000, lc_per_hour=24, cpa_trials=60,
               n=dict(cal=900, ben_test=1500, sup=900, atk=250))
    if a.quick:
        cfg.update(resamples=300, lc_per_hour=6, cpa_trials=10, n=dict(cal=150, ben_test=200, sup=200, atk=50))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    report = ["# Attack scenarios and detection (synthetic traffic)\n",
              f"seed={a.seed}; every detector calibrated to {FPR_BUDGET:.0%} FPR on a separate benign calibration set "
              f"({cfg['n']['cal']} windows); evaluation on {cfg['n']['ben_test']} benign and {cfg['n']['atk']} windows per attack. "
              "Wilson 95 % intervals in brackets.\n"]
    cat = catalogue(); cat.to_csv(out / "a1_catalogue.csv", index=False)
    report.append(md(cat, "A1 Scenario catalogue")); print(cat.to_string(index=False)); print()
    print("building data ..."); D = build(cfg, rng)
    print("detectors ...")
    t2, t3, state = a2_detectors(D, cfg, out)
    t2.to_csv(out / "a2_detector_comparison.csv", index=False)
    t3.to_csv(out / "a3_ablation.csv", index=False)
    report.append(md(t2, "A2 Detector comparison at a common 1 % false-positive budget"))
    report.append(md(t3, "A3 Ablation of the PAFA fusion (one component removed)"))
    print(t2.to_string(index=False)); print(); print(t3.to_string(index=False)); print()
    t4 = a4_topology(state, cfg); t4.to_csv(out / "a4_topology.csv", index=False)
    report.append(md(t4, "A4 Topological features, benign vs attack (permutation test, Bonferroni alpha/3)"))
    print(t4.to_string(index=False)); print()
    t5, timeline = a5_lifecycle(state, cfg, rng); t5.to_csv(out / "a5_lifecycle.csv", index=False)
    report.append(md(t5, "A5 Multi-phase lifecycle: per-phase detection rate of attack windows"))
    print(t5.to_string(index=False)); print()
    t6, t6c = a6_sidechannel(cfg, rng)
    t6.to_csv(out / "a6_sidechannel_cpa.csv", index=False); t6c.to_csv(out / "a6_noise_cost.csv", index=False)
    report.append(md(t6, "A6 Simulated CPA key recovery: success rate vs number of traces (injected amplitude is a multiple of the measurement-noise sd; the multiples were chosen to span the 60-98 % Eq. 47 reduction range, so the CPA columns are the informative result)"))
    report.append(md(t6c, "A6b Injected noise power: static vs threat-adaptive"))
    print(t6.to_string(index=False)); print(); print(t6c.to_string(index=False)); print()
    t7 = a7_cost(D, state); t7.to_csv(out / "a7_cost.csv", index=False)
    report.append(md(t7, "A7 Measured CPU cost per window"))
    print(t7.to_string(index=False))
    plots(state, timeline, out)
    report.append(f"\nWall time {time.time()-t0:.0f}s.\n")
    (out / "attack_report_v2.md").write_text("\n".join(report))
    json.dump(cfg, open(out / "attack_config_v2.json", "w"))


if __name__ == "__main__":
    main()
