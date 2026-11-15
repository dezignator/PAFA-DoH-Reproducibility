"""Primal uSVP core-SVP estimate for LWE/RLWE parameters (Alkim et al. 2016
style). Used to check the Sec 4.21 / Table 17a statement "n=256, q=4096 targeting
128-bit post-quantum security".

This is a back-of-envelope lower-effort estimator (core-SVP, ring structure
ignored, no dual/hybrid attacks). For a publishable claim run the Albrecht et
al. lattice-estimator and quote its output; this file only makes the check
executable and shows the arithmetic.
"""
import math


def delta_bkz(beta):
    return ((math.pi * beta) ** (1.0 / beta) * beta / (2 * math.pi * math.e)) ** (1.0 / (2 * (beta - 1)))


def primal_beta(n, q, sigma, m_max=None):
    """Smallest BKZ block size beta (over sample count m) with
    sigma*sqrt(beta) <= delta^(2beta-d-1) * q^(m/d),  d = n+m+1."""
    m_max = m_max or 2 * n
    best = None
    for m in range(max(1, n // 2), m_max + 1):
        d = n + m + 1
        lo, hi = 50, d
        ok = lambda b: sigma * math.sqrt(b) <= delta_bkz(b) ** (2 * b - d - 1) * q ** (m / d)
        if not ok(hi):
            continue
        while lo < hi:
            mid = (lo + hi) // 2
            if ok(mid):
                hi = mid
            else:
                lo = mid + 1
        if best is None or lo < best[0]:
            best = (lo, m)
    return best


def core_svp_bits(beta):
    return {"classical": 0.292 * beta, "quantum": 0.265 * beta}


def report(n, q, sigma):
    b, m = primal_beta(n, q, sigma)
    bits = core_svp_bits(b)
    return {"n": n, "q": q, "sigma": sigma, "beta": b, "m": m,
            "classical_bits": round(bits["classical"], 1),
            "quantum_bits": round(bits["quantum"], 1)}


def sigma_from_cbd(eta):
    return math.sqrt(eta / 2.0)


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--q", type=int, default=4096)
    ap.add_argument("--sigma", type=float, default=3.2)
    a = ap.parse_args()
    print(json.dumps(report(a.n, a.q, a.sigma), indent=2))
