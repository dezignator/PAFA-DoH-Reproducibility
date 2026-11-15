"""Weighted geometric median by IRLS (Weiszfeld), thesis Eq. (38), and the
detector cross-check of Sec 4.6.3.

    g_hat = argmin_g  sum_k w_k ||g - g_k||_2
"""
import numpy as np


def geometric_median(G, w=None, tol=1e-9, max_iter=500, eps=1e-12):
    G = np.asarray(G, dtype=float)
    K = G.shape[0]
    w = np.ones(K) if w is None else np.asarray(w, dtype=float)
    if np.any(w < 0):
        raise ValueError("weights must be >= 0")
    g = np.average(G, axis=0, weights=w)
    for it in range(max_iter):
        d = np.linalg.norm(G - g, axis=1)
        r = w / np.maximum(d, eps)          # per-iteration reweighting
        g_new = (r[:, None] * G).sum(axis=0) / r.sum()
        if np.linalg.norm(g_new - g) < tol * max(1.0, np.linalg.norm(g)):
            return g_new, it + 1
        g = g_new
    return g, max_iter


def cross_check(pred_client, pred_detector, agreement_threshold):
    """Sec 4.6.3: exclude a client update from the round if its predictions on
    the held-out validation split agree with the topological detector on fewer
    than `agreement_threshold` of the samples. The thesis does not state the
    threshold, so it is a required argument.

    pred_client: (K, N) 0/1 predictions of each client's candidate model.
    pred_detector: (N,) 0/1 detector output.
    returns boolean mask of admitted clients.
    """
    agree = (np.asarray(pred_client) == np.asarray(pred_detector)[None, :]).mean(axis=1)
    return agree >= agreement_threshold, agree
