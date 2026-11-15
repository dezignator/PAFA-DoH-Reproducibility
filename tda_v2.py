"""Vietoris-Rips persistent homology (H0, H1), Betti numbers, persistence
entropy and p-Wasserstein distance between diagrams: thesis Eqs. (40)-(41),
Table 17.

Self-contained (no ripser/gudhi) so the artifact has no compiled dependency.
H1 is computed by Z/2 boundary-matrix reduction on the edge/triangle
filtration, which is O(n^3) triangles: use n <= ~60 points per window, or
cap the scale with `max_scale`.  For larger clouds use ripser/gudhi and
compare against this module on small inputs (tests/test_tda_v2.py does so
against analytic cases).
"""
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform


def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def vr_persistence(X, max_scale=np.inf, max_dim=1):
    """Return dict {0: (n0,2), 1: (n1,2)} of (birth, death). Infinite bars
    are returned with death = np.inf."""
    X = np.asarray(X, float)
    n = len(X)
    D = squareform(pdist(X))
    edges = [(D[i, j], i, j) for i in range(n) for j in range(i + 1, n) if D[i, j] <= max_scale]
    edges.sort()
    eidx = {(i, j): k for k, (_, i, j) in enumerate(edges)}
    # --- H0 (union-find) and classification of edges ---------------------
    parent = list(range(n))
    h0, positive = [], []
    for k, (w, i, j) in enumerate(edges):
        ri, rj = _find(parent, i), _find(parent, j)
        if ri != rj:
            parent[ri] = rj
            h0.append((0.0, w))
        else:
            positive.append(k)
    ncomp = len({_find(parent, i) for i in range(n)})
    h0 += [(0.0, np.inf)] * ncomp
    out = {0: np.array(h0, float).reshape(-1, 2)}
    if max_dim < 1:
        return out
    # --- H1: reduce triangle boundaries over Z/2 ---------------------------
    tris = []
    for i, j, k in itertools.combinations(range(n), 3):
        e1, e2, e3 = eidx.get((i, j)), eidx.get((i, k)), eidx.get((j, k))
        if e1 is None or e2 is None or e3 is None:
            continue
        tris.append((max(edges[e1][0], edges[e2][0], edges[e3][0]), (e1, e2, e3)))
    tris.sort(key=lambda t: t[0])
    low_owner = {}
    paired_edges = set()
    h1 = []
    for f_tri, bd in tris:
        col = set(bd)
        while col:
            low = max(col)
            if low in low_owner:
                col ^= low_owner[low]
            else:
                low_owner[low] = col
                paired_edges.add(low)
                b = edges[low][0]
                if f_tri > b:
                    h1.append((b, f_tri))
                break
    for k in positive:
        if k not in paired_edges:
            h1.append((edges[k][0], np.inf))
    out[1] = np.array(h1, float).reshape(-1, 2)
    return out


def finite(dgm):
    dgm = np.asarray(dgm, float).reshape(-1, 2)
    return dgm[np.isfinite(dgm[:, 1])]


def betti_at(dgm, eps):
    """beta_k(eps) = #{(b,d): b <= eps < d}   (Sec 4.7.1)."""
    dgm = np.asarray(dgm, float).reshape(-1, 2)
    return int(np.sum((dgm[:, 0] <= eps) & (eps < dgm[:, 1])))


def persistence_entropy(dgm):
    """E = -sum p_i log p_i, p_i = persistence_i / total  (Table 17)."""
    d = finite(dgm)
    if len(d) == 0:
        return 0.0
    life = d[:, 1] - d[:, 0]
    life = life[life > 0]
    if len(life) == 0:
        return 0.0
    p = life / life.sum()
    return float(-(p * np.log(p)).sum())


def wasserstein(d1, d2, p=2.0, ground="l2"):
    """Eq. (41): p-Wasserstein between finite diagrams with diagonal matching.
    ground: 'l2' (Euclidean on the (b,d) plane) or 'linf'. The thesis says
    "the metric on the plane of birth-death pairs" without naming it, so the
    choice is explicit here."""
    a, b = finite(d1), finite(d2)
    na, nb = len(a), len(b)
    if na == 0 and nb == 0:
        return 0.0
    if ground == "l2":
        diag = lambda x: (x[:, 1] - x[:, 0]) / np.sqrt(2.0)
        dist = lambda x, y: np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2)
    elif ground == "linf":
        diag = lambda x: (x[:, 1] - x[:, 0]) / 2.0
        dist = lambda x, y: np.abs(x[:, None, :] - y[None, :, :]).max(axis=2)
    else:
        raise ValueError(ground)
    C = np.zeros((na + nb, na + nb))
    if na and nb:
        C[:na, :nb] = dist(a, b) ** p
    if na:
        C[:na, nb:] = np.inf
        C[np.arange(na), nb + np.arange(na)] = diag(a) ** p
    if nb:
        C[na:, :nb] = np.inf
        C[na + np.arange(nb), np.arange(nb)] = diag(b) ** p
    # diagonal-to-diagonal block is free
    C[na:, nb:] = 0.0
    big = np.nanmax(C[np.isfinite(C)]) * 10 + 1.0
    C = np.where(np.isinf(C), big, C)
    r, c = linear_sum_assignment(C)
    return float(C[r, c].sum() ** (1.0 / p))


def summarize_window(X, eps_betti, max_scale=np.inf, ref_dgm1=None, p=2.0):
    """Per-window feature triple used in Table 17: (beta_1, entropy, W_p)."""
    dg = vr_persistence(X, max_scale=max_scale)[1]
    w = wasserstein(dg, ref_dgm1, p=p) if ref_dgm1 is not None else np.nan
    return betti_at(dg, eps_betti), persistence_entropy(dg), w
