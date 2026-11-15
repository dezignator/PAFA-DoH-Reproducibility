"""Merkle commitment (Eq. 52) and FRI folding, thesis Secs 4.17-4.18.

Educational reference over the Goldilocks field p = 2^64 - 2^32 + 1. It shows
commit -> fold -> query-consistency; it is not a STARK prover, and the
parameters returned by `fri_soundness_bits_conjectured` are the folklore
conjecture, not a proven bound. The thesis records blow-up factor, number of
queries and hash function in the run manifest; they are explicit arguments here.
"""
import hashlib
import secrets

P = 2**64 - 2**32 + 1
GEN = 7                                   # generator of the full multiplicative group


def H(*parts):
    h = hashlib.sha256()
    for x in parts:
        h.update(x if isinstance(x, bytes) else int(x).to_bytes(8, "big"))
    return h.digest()


def root_of_unity(n):
    assert n & (n - 1) == 0 and (P - 1) % n == 0
    return pow(GEN, (P - 1) // n, P)


def poly_eval(coeffs, x):
    r = 0
    for c in reversed(coeffs):
        r = (r * x + c) % P
    return r


def domain(n):
    w = root_of_unity(n)
    return [pow(w, i, P) for i in range(n)]


class Merkle:
    """Eq. (52): root over H(f(w^i)), i = 0..n-1."""
    def __init__(self, values):
        self.n = len(values)
        assert self.n & (self.n - 1) == 0
        self.layers = [[H(v) for v in values]]
        while len(self.layers[-1]) > 1:
            L = self.layers[-1]
            self.layers.append([H(L[i], L[i + 1]) for i in range(0, len(L), 2)])
        self.root = self.layers[-1][0]

    def open(self, idx):
        path = []
        for L in self.layers[:-1]:
            path.append(L[idx ^ 1])
            idx //= 2
        return path

    @staticmethod
    def verify(root, idx, value, path):
        h = H(value)
        for sib in path:
            h = H(h, sib) if idx % 2 == 0 else H(sib, h)
            idx //= 2
        return h == root


def fold(evals, alpha):
    """One FRI round. With f(x) = f_e(x^2) + x f_o(x^2):
       f'(w^{2i}) = (f(w^i)+f(-w^i))/2 + alpha (f(w^i)-f(-w^i))/(2 w^i).
    Degree bound halves. `evals` are values on the order-n subgroup."""
    n = len(evals)
    h = n // 2
    w = root_of_unity(n)
    inv2 = pow(2, P - 2, P)
    out = []
    for i in range(h):
        x = pow(w, i, P)
        a, b = evals[i], evals[i + h]            # f(x), f(-x) since w^h = -1
        e = (a + b) * inv2 % P
        o = (a - b) * inv2 % P * pow(x, P - 2, P) % P
        out.append((e + alpha * o) % P)
    return out


def _challenge(transcript):
    return int.from_bytes(hashlib.sha256(transcript).digest(), "big") % P


def fri_prove(coeffs, blowup=4):
    """Commit phase. Returns layers (evaluations), roots, alphas (Fiat-Shamir)."""
    d1 = len(coeffs)
    n = 1
    while n < d1 * blowup:
        n *= 2
    ev = [poly_eval(coeffs, x) for x in domain(n)]
    layers, trees, alphas = [ev], [Merkle(ev)], []
    transcript = trees[0].root
    while len(layers[-1]) > blowup:
        a = _challenge(transcript)
        alphas.append(a)
        nxt = fold(layers[-1], a)
        layers.append(nxt)
        trees.append(Merkle(nxt))
        transcript = trees[-1].root
    return layers, trees, alphas


def fri_verify(layers, trees, alphas, n_queries=16, rng=None):
    """Query phase + final-layer check. The last layer must be constant
    (degree 0) when the committed polynomial has degree < n0/blowup."""
    rng = rng or secrets.SystemRandom()
    last = layers[-1]
    if any(v != last[0] for v in last):
        return False, "final layer not constant"
    n0 = len(layers[0])
    for _ in range(n_queries):
        idx = rng.randrange(n0 // 2)
        for r in range(len(layers) - 1):
            n = len(layers[r])
            h = n // 2
            i = idx % h
            a, b = layers[r][i], layers[r][i + h]
            if not (Merkle.verify(trees[r].root, i, a, trees[r].open(i)) and
                    Merkle.verify(trees[r].root, i + h, b, trees[r].open(i + h))):
                return False, f"merkle opening failed, layer {r}"
            x = pow(root_of_unity(n), i, P)
            inv2 = pow(2, P - 2, P)
            exp = ((a + b) * inv2 + alphas[r] * (a - b) * inv2 % P * pow(x, P - 2, P)) % P
            if exp != layers[r + 1][i]:
                return False, f"fold mismatch, layer {r}"
            idx = i
    return True, "ok"


def fri_soundness_bits_conjectured(blowup, n_queries):
    """Folklore conjecture: per-query soundness ~ 1/blowup, so
    bits ~ n_queries * log2(blowup). Not a theorem; do not quote as one."""
    import math
    return n_queries * math.log2(blowup)
