"""Synthetic DoH traffic scenarios and per-window features, shared by the
experiment scripts in experiments/.

All traffic here is SIMULATED. Scenario parameters are mine, chosen to give
realistic-looking overlap (periodic benign polling, CDN hash labels, a mimicry
attack with benign-like statistics). They are not the Zenodo/ns-3/dnscat2
testbed of Sec 4.25-4.26, so results built on them are not Table 14-17.
"""
import math
import numpy as np
from .flows_v2 import label_entropy

W = 48                      # queries per window
_B32 = list("abcdefghijklmnopqrstuvwxyz234567")
_HEX = list("0123456789abcdef")
_WORDS = ["mail", "www", "api", "cdn", "img", "login", "static", "update", "pay", "docs",
          "shop", "news", "maps", "video", "cloud", "auth", "files", "chat"]
_QT = {"A": 0, "AAAA": 1, "TXT": 2, "MX": 3, "CNAME": 4, "NULL": 5}
SCENARIOS = ["benign", "tunnel", "beacon", "slow_exfil", "mimic"]


def _rl(rng, alphabet, n):
    return "".join(rng.choice(alphabet, n))


def _qtypes(rng, n, probs):
    k = list(_QT)
    return [k[i] for i in rng.choice(len(k), n, p=probs)]


def gen_window(kind, rng, w=W):
    """Return (ts, size, qnames, qtypes) for one window."""
    if kind == "benign":
        sub = rng.random()
        if sub < 0.15:            # periodic telemetry / NTP-like polling (benign, low entropy)
            p = rng.uniform(2, 10)
            ts = np.cumsum(p * (1 + rng.normal(0, 0.05, w)).clip(0.5))
            names = [f"time{rng.integers(0, 3)}.vendor{rng.integers(0, 4)}.com" for _ in range(w)]
            size = rng.normal(100, 6, w)
        elif sub < 0.27:          # CDN / hash-like labels (benign, high entropy)
            ts = np.cumsum(rng.lognormal(-1.0, 0.9, w))
            names = [f"{_rl(rng, _HEX, int(rng.integers(8, 20)))}.cdn{rng.integers(0, 5)}.net" for _ in range(w)]
            size = rng.normal(130, 25, w)
        else:                     # ordinary browsing bursts
            ts = np.cumsum(rng.lognormal(-1.0, 1.1, w))
            names = [f"{rng.choice(_WORDS)}{rng.integers(0, 60)}.site{rng.integers(0, 30)}.org" for _ in range(w)]
            size = rng.normal(120, 28, w)
        qt = _qtypes(rng, w, [0.62, 0.30, 0.03, 0.02, 0.025, 0.005])
    elif kind == "tunnel":        # dnscat2-like
        ts = np.cumsum(rng.exponential(rng.uniform(0.15, 1.0), w))
        L = int(rng.integers(28, 52))
        names = [f"{_rl(rng, _B32, L)}.t{rng.integers(0, 3)}.evil.net" for _ in range(w)]
        size = rng.normal(235, 15, w)
        qt = _qtypes(rng, w, [0.05, 0.02, 0.5, 0.1, 0.25, 0.08])
    elif kind == "beacon":        # C2 heartbeat: periodic, short labels, small payload
        p = rng.uniform(2, 10)
        ts = np.cumsum(p * (1 + rng.normal(0, rng.uniform(0.04, 0.15), w)).clip(0.4))
        names = [f"c{_rl(rng, _HEX, 6)}.upd.evil.net" for _ in range(w)]
        size = rng.normal(110, 6, w)
        qt = _qtypes(rng, w, [0.55, 0.05, 0.35, 0.02, 0.02, 0.01])
    elif kind == "slow_exfil":    # low-and-slow
        ts = np.cumsum(rng.exponential(rng.uniform(12, 45), w))
        names = [f"{_rl(rng, _B32, int(rng.integers(18, 30)))}.s.evil.net" for _ in range(w)]
        size = rng.normal(150, 12, w)
        qt = _qtypes(rng, w, [0.1, 0.05, 0.6, 0.05, 0.15, 0.05])
    elif kind == "mimic":         # statistics tuned to benign; hidden cyclic rate modulation
        base = rng.lognormal(-1.0, 0.8, w)
        mod = 1 + 0.9 * np.sin(2 * np.pi * np.arange(w) / rng.integers(6, 9))
        ts = np.cumsum(base * mod)
        names = [f"{rng.choice(_WORDS)}{rng.integers(0, 60)}.site{rng.integers(0, 30)}.org" for _ in range(w)]
        size = rng.normal(122, 27, w)
        qt = _qtypes(rng, w, [0.60, 0.30, 0.05, 0.02, 0.025, 0.005])
    else:
        raise ValueError(kind)
    return ts, np.clip(size, 40, 512), names, qt


def window_features(ts, size, names, qt):
    """17 features per window: timing, size, label statistics, periodicity."""
    iat = np.diff(ts)
    iat = iat if len(iat) else np.array([1.0])
    ent = np.array([label_entropy(n.split(".")[0]) for n in names])
    lab = [n.split(".")[0] for n in names]
    L = np.array([len(x) for x in lab], float)
    x = (iat - iat.mean()) / (iat.std() + 1e-9)
    ac = [float(np.mean(x[:-k] * x[k:])) for k in range(1, 9)]
    q = np.array([_QT[t] for t in qt])
    return np.array([
        np.log(iat.mean() + 1e-3), iat.std() / (iat.mean() + 1e-9),
        size.mean() / 100, size.std() / 30,
        ent.mean(), ent.std(), ent.max(),
        L.mean() / 10, len(set(lab)) / len(lab),
        ac[0], max(ac[1:]), float(np.argmax(np.abs(ac)) + 1) / 8,
        float(np.mean(q == 2)), float(np.mean(q >= 3)),
        float(np.mean(np.array([len(set(n.split('.')[-2:])) for n in names]))) - 1.0,
        np.log1p(np.percentile(iat, 90)), np.log1p(np.percentile(iat, 10)),
    ])


FEATURE_NAMES = ["log_mean_iat", "iat_cv", "size_mean", "size_sd", "ent_mean", "ent_sd", "ent_max",
                 "label_len", "uniq_label_frac", "ac1", "ac_peak", "ac_peak_lag", "txt_frac",
                 "rare_qtype_frac", "zone_variety", "log_p90_iat", "log_p10_iat"]


def make_dataset(n_per_class, rng, kinds=SCENARIOS, w=W):
    X, y, k = [], [], []
    for kind in kinds:
        n = n_per_class[kind] if isinstance(n_per_class, dict) else n_per_class
        for _ in range(n):
            X.append(window_features(*gen_window(kind, rng, w)))
            y.append(0 if kind == "benign" else 1)
            k.append(kind)
    return np.array(X), np.array(y), np.array(k)


def iat_series(ts):
    d = np.diff(ts)
    return np.log(d + 1e-3)
