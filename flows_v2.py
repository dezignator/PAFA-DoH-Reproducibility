"""DoH flow -> point cloud (Sec 4.7.1) and synthetic flow generators.

Point coordinates per query/response pair: inter-arrival time, payload size,
Shannon entropy of the subdomain label. The thesis also lists "protocol-
specific parameters"; they are not enumerated there, so extra columns can be
appended via `extra`.

The generators are SYNTHETIC. They exist for unit tests and for exercising
the pipeline end to end; numbers computed from them are not thesis results.
"""
import math
from collections import Counter
import numpy as np


def label_entropy(label):
    if not label:
        return 0.0
    c = Counter(label)
    n = len(label)
    return -sum((v / n) * math.log2(v / n) for v in c.values())


def subdomain_label(qname, zone_labels=2):
    parts = qname.rstrip(".").split(".")
    return ".".join(parts[:-zone_labels]) if len(parts) > zone_labels else ""


def flow_to_cloud(ts, size, qname, extra=None, standardize=True):
    ts = np.asarray(ts, float)
    iat = np.diff(ts, prepend=ts[0])
    ent = np.array([label_entropy(subdomain_label(q)) for q in qname])
    cols = [iat, np.asarray(size, float), ent]
    if extra is not None:
        cols += [np.asarray(c, float) for c in extra]
    X = np.column_stack(cols)
    if standardize:
        mu, sd = X.mean(axis=0), X.std(axis=0)
        X = (X - mu) / np.where(sd > 0, sd, 1.0)
    return X


def windows(X, width, stride=None):
    stride = stride or width
    return [X[i:i + width] for i in range(0, len(X) - width + 1, stride)]


_B32 = "abcdefghijklmnopqrstuvwxyz234567"
_WORDS = ["mail", "www", "api", "cdn", "img", "login", "static", "update", "pay", "docs"]


def synth_benign(n, rng, zone="example.org"):
    ts = np.cumsum(rng.lognormal(mean=-1.0, sigma=1.0, size=n))
    size = rng.normal(120, 25, n).clip(40, 512)
    names = [f"{rng.choice(_WORDS)}{rng.integers(0, 50)}.{zone}" for _ in range(n)]
    return ts, size, names


def synth_tunnel(n, rng, zone="t.example.net", period=1.0, jitter=0.05, label_len=40):
    ts = np.cumsum(period + rng.normal(0, jitter, n).clip(-0.5 * period, None))
    size = rng.normal(230, 8, n).clip(150, 512)
    names = ["".join(rng.choice(list(_B32), label_len)) + "." + zone for _ in range(n)]
    return ts, size, names


# ---- optional representation: NOT the Sec 4.7.1 point cloud -----------------
def count_series(ts, bin_width):
    """Queries per time bin."""
    ts = np.asarray(ts, float)
    edges = np.arange(ts.min(), ts.max() + bin_width, bin_width)
    return np.histogram(ts, bins=edges)[0].astype(float)


def delay_embed(x, dim=6, lag=1):
    """Sliding-window (Takens) embedding of a 1-D series; a periodic series
    traces a closed curve, which H1 picks up. Provided as a comparison
    representation: Sec 4.7.1 builds the cloud from per-query coordinates,
    where nothing forces a periodic beacon to form a loop."""
    x = np.asarray(x, float)
    n = len(x) - (dim - 1) * lag
    if n <= 0:
        raise ValueError("series too short")
    E = np.column_stack([x[i * lag:i * lag + n] for i in range(dim)])
    E = E - E.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(E, axis=1, keepdims=True)
    return E / np.where(nrm > 0, nrm, 1.0)


def synth_beacon_bursty(n_bins, rng, period_bins=7, base=0.5, amp=4.0):
    """Beacon with a periodic query-rate modulation (synthetic)."""
    t = np.arange(n_bins)
    lam = base + amp * (np.sin(2 * np.pi * t / period_bins) > 0.3)
    return rng.poisson(lam).astype(float)


def synth_benign_counts(n_bins, rng, lam=1.5):
    return rng.poisson(lam, n_bins).astype(float)
