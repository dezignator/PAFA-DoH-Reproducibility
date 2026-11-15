#!/usr/bin/env python3
"""PAFA-DoH Chapter 4, federated-learning experiment suite (v2).

Simulates the training side of the framework on SYNTHETIC DoH windows and
produces the tables that Chapter 4 reports for it (Table 16 style):

  E1  convergence            centralized vs FedAvg vs DP-FL (eps = 1.2)
  E2  privacy-utility        eps sweep at fixed K, rounds
  E3  client dropout         0..40 % of clients absent each round
  E4  Byzantine clients      mean / clipped mean / geometric median /
                             geometric median + detector cross-check
  E5  adversarial robustness feature-space evasion (FGSM) with and without
                             adversarial augmentation inside the FL loop
  E6  scale of the client population needed for eps = 1.2

Everything is numpy. Clients are trained in one batched pass (einsum over the
client axis), so a 400-client round costs a few milliseconds.

What the numbers mean. The data come from pafa_v2/windows_v2.py, not from the
Zenodo/ns-3/dnscat2 testbed, and the model is a small MLP over 17 window
features. The tables therefore show how the *mechanisms* behave (aggregation,
clipping, noise, robust aggregation, augmentation) under controlled conditions;
they are not the values of Table 16. Where the thesis target is >= 99.2 %
accuracy, compare the trend and the cost of each mechanism, not the absolute
figure.

usage:
    python3 experiments/pafa_federated_experiments_v2.py --out build/fed
    python3 experiments/pafa_federated_experiments_v2.py --quick        # smoke run
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pafa_v2 import windows_v2 as wv                      # noqa: E402
from pafa_v2.dp_accountant_v2 import eps_from_rdp, sigma_for_target   # noqa: E402
from pafa_v2.byzantine_v2 import geometric_median          # noqa: E402
from pafa_v2.metrics_v2 import wilson                      # noqa: E402

TRAIN_KINDS = ["benign", "tunnel", "beacon", "slow_exfil"]   # 'mimic' is held out (zero-day)
ALL_KINDS = wv.SCENARIOS
D = len(wv.FEATURE_NAMES)
H = 24


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def build_data(cache, n_train=16000, n_test=3000, n_val=600, n_public=400, seed=0):
    cache = Path(cache)
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    rng = np.random.default_rng(seed)
    # train pools per family (benign 40 %)
    share = {"benign": 0.40, "tunnel": 0.20, "beacon": 0.20, "slow_exfil": 0.20}
    pools = {}
    for kind in TRAIN_KINDS:
        n = int(n_train * share[kind])
        X, _, _ = wv.make_dataset({kind: n}, rng, kinds=[kind])
        pools[kind] = X
    out = {f"pool_{k}": v for k, v in pools.items()}
    tshare = {"benign": 0.5, "tunnel": 0.125, "beacon": 0.125, "slow_exfil": 0.125, "mimic": 0.125}
    Xt, yt, kt = wv.make_dataset({k: int(n_test * v) for k, v in tshare.items()}, rng, kinds=ALL_KINDS)
    out.update(test_X=Xt, test_y=yt, test_k=kt)
    # validation split used by the detector cross-check; public split = labeled seed set
    # for the reference detector and for feature standardisation statistics
    Xv, yv, kv = wv.make_dataset({k: n_val // 4 for k in TRAIN_KINDS}, rng, kinds=TRAIN_KINDS)
    Xp, yp, kp = wv.make_dataset({k: n_public // 4 for k in TRAIN_KINDS}, rng, kinds=TRAIN_KINDS)
    out.update(val_X=Xv, val_y=yv, pub_X=Xp, pub_y=yp)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


class Scaler:
    def __init__(self, X):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9

    def __call__(self, X):
        return (X - self.mu) / self.sd


# --------------------------------------------------------------------------
# batched MLP (parameters as one flat vector per client)
# --------------------------------------------------------------------------
NP = D * H + H + H + 1


def unpack(theta):
    """theta: (..., NP) -> W1 (...,D,H), b1 (...,H), W2 (...,H), b2 (...,)"""
    i = 0
    W1 = theta[..., i:i + D * H].reshape(theta.shape[:-1] + (D, H)); i += D * H
    b1 = theta[..., i:i + H]; i += H
    W2 = theta[..., i:i + H]; i += H
    b2 = theta[..., i]
    return W1, b1, W2, b2


def init_theta(rng):
    W1 = rng.normal(0, 1 / np.sqrt(D), (D, H))
    W2 = rng.normal(0, 1 / np.sqrt(H), H)
    return np.concatenate([W1.ravel(), np.zeros(H), W2, [0.0]])


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def forward(theta, X):
    """theta (NP,), X (n,D) -> prob (n,)"""
    W1, b1, W2, b2 = unpack(theta)
    A = np.tanh(X @ W1 + b1)
    return sigmoid(A @ W2 + b2)


def forward_batched(Theta, X):
    """Theta (K,NP), X (K,n,D) or (n,D) -> prob (K,n)"""
    W1, b1, W2, b2 = unpack(Theta)
    if X.ndim == 2:
        Z = np.einsum("nd,kdh->knh", X, W1) + b1[:, None]
    else:
        Z = np.einsum("knd,kdh->knh", X, W1) + b1[:, None]
    A = np.tanh(Z)
    return sigmoid(np.einsum("knh,kh->kn", A, W2) + b2[:, None])


def grad_batched(Theta, X, y):
    """Mean BCE gradient per client. X (K,n,D), y (K,n) -> (K,NP)"""
    W1, b1, W2, b2 = unpack(Theta)
    n = X.shape[1]
    Z = np.einsum("knd,kdh->knh", X, W1) + b1[:, None]
    A = np.tanh(Z)
    p = sigmoid(np.einsum("knh,kh->kn", A, W2) + b2[:, None])
    dl = (p - y) / n
    gW2 = np.einsum("knh,kn->kh", A, dl)
    gb2 = dl.sum(1)
    dZ = dl[:, :, None] * W2[:, None, :] * (1 - A ** 2)
    gW1 = np.einsum("knd,knh->kdh", X, dZ)
    gb1 = dZ.sum(1)
    return np.concatenate([gW1.reshape(len(X), -1), gb1, gW2, gb2[:, None]], axis=1)


def input_grad(theta, X, y):
    """d loss / d x for the single global model (used by FGSM)."""
    W1, b1, W2, b2 = unpack(theta)
    A = np.tanh(X @ W1 + b1)
    p = sigmoid(A @ W2 + b2)
    dZ = ((p - y)[:, None] * W2[None, :]) * (1 - A ** 2)
    return dZ @ W1.T


def fgsm(theta, X, y, eps):
    return X + eps * np.sign(input_grad(theta, X, y))


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def auc_rank(s, y):
    s = np.asarray(s)
    r = pd.Series(s).rank().values
    n1, n0 = int(y.sum()), int((1 - y).sum())
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def evaluate(theta, X, y, kinds):
    p = forward(theta, X)
    pred = p >= 0.5
    known = kinds != "mimic"
    out = {"acc": float((pred == y).mean()),
           "acc_known": float((pred[known] == y[known]).mean()),
           "fpr": float(pred[y == 0].mean()),
           "tpr": float(pred[y == 1].mean()),
           "auc": float(auc_rank(p, y))}
    for k in np.unique(kinds):
        if k != "benign":
            out[f"recall_{k}"] = float(pred[kinds == k].mean())
    return out


# --------------------------------------------------------------------------
# federated simulation
# --------------------------------------------------------------------------
class FedSim:
    """K clients with label-skewed local data, equal local size N/K."""

    def __init__(self, data, scaler, K, seed, dirichlet=0.7):
        self.rng = np.random.default_rng(seed)
        self.K = K
        self.scaler = scaler
        pools = {k: scaler(data[f"pool_{k}"]) for k in TRAIN_KINDS}
        N = sum(len(v) for v in pools.values())
        self.n = max(8, N // K)
        mix = self.rng.dirichlet(dirichlet * np.array([3.0, 1, 1, 1]) , size=K)   # benign-heavy prior
        X = np.empty((K, self.n, D))
        y = np.empty((K, self.n))
        for c in range(K):
            fam = self.rng.choice(len(TRAIN_KINDS), self.n, p=mix[c])
            for j, kind in enumerate(TRAIN_KINDS):
                idx = np.nonzero(fam == j)[0]
                if len(idx):
                    X[c, idx] = pools[kind][self.rng.integers(0, len(pools[kind]), len(idx))]
                    y[c, idx] = 0.0 if kind == "benign" else 1.0
        self.X, self.y = X, y
        self.Xte = scaler(data["test_X"])
        self.yte, self.kte = data["test_y"], data["test_k"]
        self.Xval = scaler(data["val_X"])
        self.yval = data["val_y"]
        self.Xpub = scaler(data["pub_X"])
        self.ypub = data["pub_y"]

    # ---- local training (all clients at once) ----
    def local_deltas(self, theta, idx, epochs=2, bs=20, lr=0.2, adv_eps=0.0, flip=None):
        """Return (len(idx), NP) model deltas after local SGD from the global theta."""
        Kc = len(idx)
        Th = np.tile(theta, (Kc, 1))
        X, y = self.X[idx], self.y[idx].copy()
        if flip is not None:                      # label-flipping clients
            y[flip] = 1.0 - y[flip]
        n = X.shape[1]
        bs = min(bs, n)
        for _ in range(epochs):
            perm = self.rng.permutation(n)
            for s in range(0, n - bs + 1, bs):
                b = perm[s:s + bs]
                Xb, yb = X[:, b], y[:, b]
                if adv_eps > 0:                    # augment attack samples with FGSM from the *global* model
                    am = yb == 1
                    Xadv = Xb.copy()
                    flat = Xb.reshape(-1, D)
                    g = input_grad(theta, flat, yb.reshape(-1)).reshape(Xb.shape)
                    Xadv[am] = (Xb + adv_eps * np.sign(g))[am]
                    Xb = np.concatenate([Xb, Xadv[:, :, :]], axis=1)
                    yb = np.concatenate([yb, yb], axis=1)
                Th -= lr * grad_batched(Th, Xb, yb)
        return Th - theta


def clip_rows(D_, C):
    nrm = np.linalg.norm(D_, axis=1, keepdims=True)
    return D_ * np.minimum(1.0, C / np.maximum(nrm, 1e-12))


def reference_detector(sim):
    """Logistic regression on the small public labeled set: stands in for the
    topological detector of Sec 4.6.3 as the cross-check reference."""
    X, y = sim.Xpub, sim.ypub
    w = np.zeros(D); b = 0.0
    for _ in range(400):
        p = sigmoid(X @ w + b)
        w -= 0.3 * (X.T @ (p - y) / len(y)); b -= 0.3 * (p - y).mean()
    return (sigmoid(sim.Xval @ w + b) >= 0.5).astype(int)


def run_fl(sim, seed, rounds=30, aggregator="mean", eps=None, delta=1e-5, C=3.5,
           dropout=0.0, byz_frac=0.0, byz_mode="signflip", byz_scale=5.0,
           adv_eps=0.0, cross_thr=0.75, lr_server=1.0, track=True):
    rng = np.random.default_rng(seed + 1000)
    sim.rng = np.random.default_rng(seed + 2000)
    theta = init_theta(rng)
    K = sim.K
    byz = np.zeros(K, bool)
    if byz_frac > 0:
        byz[rng.choice(K, int(round(byz_frac * K)), replace=False)] = True
    sigma = sigma_for_target(rounds, eps, delta) if eps is not None else 0.0
    det = reference_detector(sim) if aggregator == "geomed_xcheck" else None
    hist, excl_log = [], []
    for t in range(rounds):
        active = np.nonzero(rng.random(K) >= dropout)[0]
        if len(active) < 2:
            continue
        flip = byz[active] if byz_mode == "labelflip" else None
        d = sim.local_deltas(theta, active, adv_eps=adv_eps, flip=flip)
        if byz_mode in ("signflip", "noise") and byz.any():
            m = byz[active]
            if byz_mode == "signflip":
                d[m] = -byz_scale * d[m]
            else:
                d[m] = rng.normal(0, byz_scale, d[m].shape)
        # --- aggregation ---
        if aggregator == "mean":
            upd = d.mean(0)
        elif aggregator == "clipmean":
            upd = clip_rows(d, C).mean(0)
        elif aggregator in ("geomed", "geomed_xcheck"):
            dd = clip_rows(d, C)
            if aggregator == "geomed_xcheck":
                cand = theta + dd
                pred = (forward_batched(cand, sim.Xval) >= 0.5).astype(int)
                agree = (pred == det[None, :]).mean(1)
                keep = agree >= cross_thr
                if keep.sum() < 3:
                    keep[:] = True
                excl_log.append(float((~keep).mean()))
                dd = dd[keep]
            upd, _ = geometric_median(dd)
        elif aggregator == "dpfl":
            dd = clip_rows(d, C)
            noise = rng.normal(0, sigma * C, NP)
            upd = (dd.sum(0) + noise) / len(dd)          # noise inside the average
        elif aggregator == "dpfl_geomed":
            dd = clip_rows(d, C)
            gm, _ = geometric_median(dd)
            upd = gm + rng.normal(0, sigma * C / len(dd), NP)
        else:
            raise ValueError(aggregator)
        theta = theta + lr_server * upd
        if track:
            e = evaluate(theta, sim.Xte, sim.yte, sim.kte)
            e["round"] = t + 1
            hist.append(e)
    final = evaluate(theta, sim.Xte, sim.yte, sim.kte)
    final["eps_spent"] = eps_from_rdp(rounds, sigma, delta)[0] if sigma > 0 else float("inf")
    final["sigma"] = sigma
    if excl_log:
        final["xcheck_excluded_frac"] = float(np.mean(excl_log))
    return theta, final, hist


def central_baseline(sim, seed, rounds=30, lr=0.2):
    """Pooled-data SGD with the same number of passes as the FL run (2 local epochs
    x rounds), mini-batch 64. Upper reference: no privacy, no federation."""
    rng = np.random.default_rng(seed + 3000)
    X = sim.X.reshape(-1, D); y = sim.y.reshape(-1)
    theta = init_theta(rng)
    hist = []
    for r in range(rounds):
        for _ in range(2):
            perm = rng.permutation(len(X))
            for s in range(0, len(X) - 64 + 1, 64):
                b = perm[s:s + 64]
                theta = theta - lr * grad_batched(theta[None], X[b][None], y[b][None])[0]
        e = evaluate(theta, sim.Xte, sim.yte, sim.kte); e["round"] = r + 1
        hist.append(e)
    return theta, hist[-1], hist


def attack_accuracy(theta, sim, eps_adv):
    """Accuracy and attack recall on the test set after FGSM on attack windows."""
    X = sim.Xte.copy(); y = sim.yte.astype(float)
    m = y == 1
    X[m] = fgsm(theta, X[m], y[m], eps_adv)
    p = forward(theta, X)
    pred = p >= 0.5
    return {"acc_adv": float((pred == y).mean()), "recall_adv": float(pred[m].mean())}


# --------------------------------------------------------------------------
# aggregation helpers for repeated runs
# --------------------------------------------------------------------------
def ms(vals):
    a = np.asarray(vals, float)
    if len(a) < 2:
        return f"{a.mean():.4f}"
    h = 1.96 * a.std(ddof=1) / np.sqrt(len(a))
    return f"{a.mean():.4f} ± {h:.4f}"


def repeat(fn, seeds):
    rows = [fn(s) for s in seeds]
    keys = rows[0].keys()
    return {k: [r[k] for r in rows] for k in keys}


# --------------------------------------------------------------------------
# experiments
# --------------------------------------------------------------------------
def exp1_convergence(data, scaler, cfg):
    rows, curves = [], {}
    sim = FedSim(data, scaler, cfg["K"], 0)
    for name, kw in [("centralized (no FL)", None),
                     ("FedAvg (no privacy)", dict(aggregator="mean")),
                     ("FL clip + geometric median", dict(aggregator="geomed")),
                     ("DP-FL eps=1.2", dict(aggregator="dpfl", eps=1.2))]:
        finals, hs = [], []
        for s in cfg["seeds"]:
            if kw is None:
                _, f, h = central_baseline(sim, s, cfg["rounds"])
            else:
                _, f, h = run_fl(sim, s, cfg["rounds"], **kw)
            finals.append(f); hs.append([x["acc_known"] for x in h])
        H_ = np.array(hs)
        curves[name] = H_.mean(0)
        thr95 = [int(np.argmax(h >= 0.95)) + 1 if (h >= 0.95).any() else None for h in H_]
        thr90 = [int(np.argmax(h >= 0.90)) + 1 if (h >= 0.90).any() else None for h in H_]
        rows.append({
            "method": name,
            "acc_known_families": ms([f["acc_known"] for f in finals]),
            "acc_incl_zero-day": ms([f["acc"] for f in finals]),
            "final_auc": ms([f["auc"] for f in finals]),
            "tpr": ms([f["tpr"] for f in finals]),
            "fpr": ms([f["fpr"] for f in finals]),
            "recall_mimic (zero-day)": ms([f["recall_mimic"] for f in finals]),
            "rounds_to_90%(known)": str(thr90),
            "rounds_to_95%(known)": str(thr95),
        })
    return pd.DataFrame(rows), curves


def exp2_privacy(data, scaler, cfg):
    rows = []
    sim = FedSim(data, scaler, cfg["K"], 0)
    for eps in [0.25, 0.5, 1.2, 3.0, 8.0, None]:
        fs = repeat(lambda s: run_fl(sim, s, cfg["rounds"], aggregator="dpfl", eps=eps, track=False)[1], cfg["seeds"])
        sig = fs["sigma"][0]
        rows.append({"eps_target": "inf (no noise)" if eps is None else eps,
                     "sigma": "-" if eps is None else f"{sig:.2f}",
                     "noise_std_on_avg": "-" if eps is None else f"{sig * 3.5 / cfg['K']:.3f}",
                     "acc_known": ms(fs["acc_known"]), "acc_all": ms(fs["acc"]), "auc": ms(fs["auc"]), "fpr": ms(fs["fpr"]),
                     "recall_mimic": ms(fs["recall_mimic"])})
    return pd.DataFrame(rows)


def exp3_dropout(data, scaler, cfg):
    rows = []
    sim = FedSim(data, scaler, cfg["K"], 0)
    base = None
    for p in [0.0, 0.1, 0.2, 0.3, 0.4]:
        fs = repeat(lambda s: run_fl(sim, s, cfg["rounds"], aggregator="mean", dropout=p, track=False)[1], cfg["seeds"])
        a = np.mean(fs["acc_known"]); base = a if base is None else base
        rows.append({"dropout": f"{int(p*100)}%", "acc_known": ms(fs["acc_known"]), "auc": ms(fs["auc"]),
                     "recovery_vs_0%_dropout": f"{100 * a / base:.2f}%"})
    for p in [0.2]:
        fs = repeat(lambda s: run_fl(sim, s, cfg["rounds"], aggregator="dpfl", eps=1.2, dropout=p, track=False)[1], cfg["seeds"])
        rows.append({"dropout": f"{int(p*100)}% (DP eps=1.2)", "acc_known": ms(fs["acc_known"]), "auc": ms(fs["auc"]),
                     "recovery_vs_0%_dropout": "-"})
    return pd.DataFrame(rows)


def exp4_byzantine(data, scaler, cfg):
    rows = []
    sim = FedSim(data, scaler, cfg["K"], 0)
    for mode in ["signflip", "labelflip"]:
        for frac in [0.0, 0.1, 0.2, 0.3]:
            row = {"attack": mode, "byz_frac": f"{int(frac*100)}%"}
            for agg in ["mean", "clipmean", "geomed", "geomed_xcheck"]:
                if frac == 0 and mode == "labelflip":
                    continue
                fs = repeat(lambda s: run_fl(sim, s, cfg["rounds"], aggregator=agg, byz_frac=frac,
                                             byz_mode=mode, track=False)[1], cfg["seeds"])
                row[agg] = ms(fs["acc_known"])
                if agg == "geomed_xcheck":
                    row["xcheck_excluded"] = f"{np.mean(fs.get('xcheck_excluded_frac', [0])):.2f}"
            if frac == 0 and mode == "labelflip":
                continue
            rows.append(row)
    return pd.DataFrame(rows)


def exp5_adversarial(data, scaler, cfg):
    rows = []
    sim = FedSim(data, scaler, cfg["K"], 0)
    models = {"FL standard": 0.0, "FL + adversarial augmentation (eps_adv=0.3)": 0.3}
    for name, ae in models.items():
        fs_all = []
        for s in cfg["seeds"]:
            th, f, _ = run_fl(sim, s, cfg["rounds"], aggregator="mean", adv_eps=ae, track=False)
            r = {"clean_acc_known": f["acc_known"], "clean_recall": f["tpr"]}
            for ev in [0.1, 0.2, 0.3, 0.5]:
                r[f"recall@{ev}"] = attack_accuracy(th, sim, ev)["recall_adv"]
            fs_all.append(r)
        keys = fs_all[0].keys()
        row = {"training": name}
        row.update({k: ms([r[k] for r in fs_all]) for k in keys})
        rows.append(row)
    return pd.DataFrame(rows)


def exp6_scale(data, scaler, cfg):
    rows = []
    for K in cfg["scale_K"]:
        sim = FedSim(data, scaler, K, 0)
        fs = repeat(lambda s: run_fl(sim, s, cfg["rounds"], aggregator="dpfl", eps=1.2, track=False)[1], cfg["seeds"])
        sig = fs["sigma"][0]
        rows.append({"clients_K": K, "samples_per_client": sim.n, "sigma": f"{sig:.2f}",
                     "noise_std_on_avg": f"{sig*3.5/K:.3f}", "acc_known_at_eps_1.2": ms(fs["acc_known"]),
                     "auc": ms(fs["auc"])})
    return pd.DataFrame(rows)


def to_md(df, title):
    return f"### {title}\n\n{df.to_markdown(index=False)}\n"


def plot_curves(curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for k, v in curves.items():
        ax.plot(np.arange(1, len(v) + 1), v, label=k)
    ax.set_xlabel("round / epoch-pair"); ax.set_ylabel("test accuracy")
    ax.set_ylim(0.5, 1.0); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("E1 convergence (synthetic windows)")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="build/fed")
    ap.add_argument("--K", type=int, default=400)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="", help="comma list of e1..e6")
    a = ap.parse_args()
    cfg = dict(K=a.K, rounds=a.rounds, seeds=list(range(a.seeds)), scale_K=[100, 400, 1600])
    if a.quick:
        cfg.update(K=100, rounds=10, seeds=[0], scale_K=[100, 200])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = build_data(out / ("data_quick.npz" if a.quick else "data.npz"),
                      **(dict(n_train=4000, n_test=800, n_val=200, n_public=200) if a.quick else {}))
    scaler = Scaler(data["pub_X"])
    todo = set(a.only.split(",")) if a.only else {"e1", "e2", "e3", "e4", "e5", "e6"}
    md = ["# Federated-learning experiments (synthetic windows)\n",
          f"K={cfg['K']} clients, {cfg['rounds']} rounds, seeds={cfg['seeds']}, "
          f"train families={TRAIN_KINDS}, held-out zero-day family='mimic'. "
          "Mean ± 95 % CI over seeds.\n"]
    if "e1" in todo:
        df, curves = exp1_convergence(data, scaler, cfg)
        df.to_csv(out / "e1_convergence.csv", index=False); plot_curves(curves, out / "e1_convergence.png")
        md.append(to_md(df, "E1 Convergence and final quality")); print(df.to_string(index=False)); print()
    if "e2" in todo:
        df = exp2_privacy(data, scaler, cfg); df.to_csv(out / "e2_privacy_utility.csv", index=False)
        md.append(to_md(df, "E2 Privacy-utility trade-off (delta=1e-5, clip C=3.5)")); print(df.to_string(index=False)); print()
    if "e3" in todo:
        df = exp3_dropout(data, scaler, cfg); df.to_csv(out / "e3_dropout.csv", index=False)
        md.append(to_md(df, "E3 Client dropout")); print(df.to_string(index=False)); print()
    if "e4" in todo:
        df = exp4_byzantine(data, scaler, cfg); df.to_csv(out / "e4_byzantine.csv", index=False)
        md.append(to_md(df, "E4 Byzantine clients: accuracy (known families) by aggregator")); print(df.to_string(index=False)); print()
    if "e5" in todo:
        df = exp5_adversarial(data, scaler, cfg); df.to_csv(out / "e5_adversarial.csv", index=False)
        md.append(to_md(df, "E5 Evasion robustness (attack-window recall under FGSM)")); print(df.to_string(index=False)); print()
    if "e6" in todo:
        df = exp6_scale(data, scaler, cfg); df.to_csv(out / "e6_client_scale.csv", index=False)
        md.append(to_md(df, "E6 Client population needed at eps=1.2")); print(df.to_string(index=False)); print()
    md.append(f"\nWall time {time.time()-t0:.0f}s. Synthetic data; see module docstring.\n")
    (out / "federated_report_v2.md").write_text("\n".join(md))
    json.dump(cfg, open(out / "federated_config_v2.json", "w"))


if __name__ == "__main__":
    main()
