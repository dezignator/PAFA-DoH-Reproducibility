"""Spike encoding, LIF neuron, pair-based STDP: thesis Eqs. (43)-(46).

Software reference only. Nothing here runs on Loihi; latency/energy figures in
Table 13 come from hardware and are not produced by this module.
"""
import numpy as np


def log_burst_encode(t_arr, feat_sqnorm, tau, alpha, n_spikes):
    """Eq. (43).  t_k = t_{k-1} + tau*log(1 + ||p||^2/alpha), t_0 = t_arr.

    t_arr: (P,) packet arrival times; feat_sqnorm: (P,) squared L2 norm of the
    feature vector (the thesis defines ||p_i||_2 as the squared norm);
    n_spikes: int or (P,) burst sizes N_i.  Returns the merged, sorted spike
    train S(t) = sum_i S_i(t) and the packet index of each spike.
    """
    t_arr = np.asarray(t_arr, float)
    sq = np.asarray(feat_sqnorm, float)
    N = np.broadcast_to(np.asarray(n_spikes), t_arr.shape).astype(int)
    isi = tau * np.log1p(sq / alpha)                  # constant inside a burst
    times, owner = [], []
    for i in range(len(t_arr)):
        k = np.arange(1, N[i] + 1)
        times.append(t_arr[i] + k * isi[i])
        owner.append(np.full(N[i], i))
    times = np.concatenate(times)
    owner = np.concatenate(owner)
    o = np.argsort(times, kind="stable")
    return times[o], owner[o]


def lif_simulate(I_syn, dt, tau_m, v_rest, v_th, v_reset, tau_ref,
                 sigma_noise=0.0, rng=None, v0=None):
    """Eqs. (44)-(45), Euler-Maruyama.

    tau_m dV = -(V - V_rest) dt + I dt + sigma*sqrt(tau_m*dt) dW   (convention:
    xi(t) is white noise with intensity sigma; the thesis does not fix the
    discretisation, so it is stated here).  I_syn: (steps,) or (steps, N).
    Returns spike_steps list per neuron and the V trace.
    """
    I = np.atleast_2d(np.asarray(I_syn, float).T).T if np.ndim(I_syn) == 1 else np.asarray(I_syn, float)
    if I.ndim == 1:
        I = I[:, None]
    steps, N = I.shape
    rng = rng or np.random.default_rng(0)
    V = np.full(N, v_rest if v0 is None else v0, float)
    ref = np.zeros(N)
    spikes = [[] for _ in range(N)]
    trace = np.empty((steps, N))
    for s in range(steps):
        active = ref <= 0
        dV = (-(V - v_rest) + I[s]) * dt / tau_m
        if sigma_noise > 0:
            dV = dV + sigma_noise * np.sqrt(dt / tau_m) * rng.standard_normal(N)
        V = np.where(active, V + dV, v_reset)
        ref = np.where(active, ref, ref - dt)
        fired = active & (V >= v_th)
        for j in np.nonzero(fired)[0]:
            spikes[j].append(s)
        V = np.where(fired, v_reset, V)
        ref = np.where(fired, tau_ref, ref)
        trace[s] = V
    return spikes, trace


def lif_period_analytic(I, tau_m, v_rest, v_th, v_reset):
    """Noiseless inter-spike period for constant I (excl. refractory)."""
    return tau_m * np.log((I - (v_reset - v_rest)) / (I - (v_th - v_rest)))


def stdp_delta_w(t_pre, t_post, eta, tau_plus=20.0, tau_minus=20.0):
    """Eq. (46), all-to-all pairing:
    dw = eta * sum_{ti,tj} [ exp(-(tj-ti)/tau+) H(tj-ti) - exp(-(ti-tj)/tau-) H(ti-tj) ]
    Times in ms.  Equal-time pairs contribute 0 (H(0) taken as 0)."""
    ti = np.asarray(t_pre, float)[:, None]
    tj = np.asarray(t_post, float)[None, :]
    d = tj - ti
    pot = np.where(d > 0, np.exp(-np.clip(d, 0, None) / tau_plus), 0.0)
    dep = np.where(d < 0, np.exp(-np.clip(-d, 0, None) / tau_minus), 0.0)
    return float(eta * (pot - dep).sum())


def stdp_delta_w_nearest(t_pre, t_post, eta, tau_plus=20.0, tau_minus=20.0):
    """Nearest-neighbour pairing variant (Sec 4.11 says the pairing convention
    is a configuration choice): each post spike pairs with the closest
    preceding pre spike (potentiation) and the closest following pre spike
    (depression)."""
    pre = np.sort(np.asarray(t_pre, float))
    dw = 0.0
    for tj in np.asarray(t_post, float):
        k = np.searchsorted(pre, tj, side="left")
        if k > 0:
            dw += np.exp(-(tj - pre[k - 1]) / tau_plus)
        k2 = np.searchsorted(pre, tj, side="right")
        if k2 < len(pre):
            dw -= np.exp(-(pre[k2] - tj) / tau_minus)
    return float(eta * dw)


def apply_bounded(w, dw, w_min=0.0, w_max=1.0):
    return float(np.clip(w + dw, w_min, w_max))
