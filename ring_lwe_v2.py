"""Ring-LWE encryption of Eq. (58) in R_q = Z_q[x]/(x^n + 1).

    c = (a, a*s + e + floor(q/t) * m)   (mod q),   t = 2 reproduces the
    "high-order bit" encoding of Sec 4.21.

Functional reference for additive homomorphism and noise growth. NOT a
production scheme (no CCA transform, no constant-time arithmetic) and the
default (n=256, q=4096) is the parameter pair stated in Table 17a; see
lattice_estimate_v2.py for what the core-SVP estimate says about it.
"""
import numpy as np


def negacyclic_mul(a, b, q):
    n = len(a)
    full = np.convolve(a.astype(np.int64), b.astype(np.int64))
    res = full[:n].copy()
    res[: len(full) - n] -= full[n:]
    return res % q


class RLWE:
    def __init__(self, n=256, q=4096, sigma=3.2, t=2, seed=None):
        self.n, self.q, self.sigma, self.t = n, q, sigma, t
        self.delta = q // t
        self.rng = np.random.default_rng(seed)
        self.s = self._small()

    def _small(self):
        return np.rint(self.rng.normal(0, self.sigma, self.n)).astype(np.int64)

    def encrypt(self, m):
        m = np.asarray(m, dtype=np.int64) % self.t
        if len(m) != self.n:
            raise ValueError("message length must equal n")
        a = self.rng.integers(0, self.q, self.n, dtype=np.int64)
        e = self._small()
        c1 = (negacyclic_mul(a, self.s, self.q) + e + self.delta * m) % self.q
        return a, c1

    def decrypt(self, ct):
        a, c1 = ct
        v = (c1 - negacyclic_mul(a, self.s, self.q)) % self.q
        return np.rint(v / self.delta).astype(np.int64) % self.t

    def noise(self, ct, m):
        """Centered noise of a ciphertext given the true plaintext (diagnostic)."""
        a, c1 = ct
        v = (c1 - negacyclic_mul(a, self.s, self.q) - self.delta * (np.asarray(m) % self.t)) % self.q
        return np.where(v > self.q // 2, v - self.q, v)

    # --- homomorphic operations -------------------------------------------
    def add(self, ct1, ct2):
        return (ct1[0] + ct2[0]) % self.q, (ct1[1] + ct2[1]) % self.q

    def mul_plain_scalar(self, ct, k):
        """Multiply by a public small integer: noise scales by |k|."""
        return (ct[0] * k) % self.q, (ct[1] * k) % self.q

    def decrypt_budget_bits(self, ct, m):
        """log2 of (delta/2) / max|noise|: decryption is correct while > 0."""
        mx = np.abs(self.noise(ct, m)).max()
        return float(np.log2((self.delta / 2) / max(mx, 1)))
