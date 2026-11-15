"""R1CS -> QAP over a prime field: thesis Eqs. (50)-(51).

Illustrative only: checks satisfiability and the divisibility condition
sum_j z_j u_j(X) * sum_j z_j v_j(X) - sum_j z_j w_j(X) = h(X) t(X)
on a toy circuit. No pairing, no trusted setup, no Groth16 prover. It does not
produce the 432 ms / 78 ms / 1.8 KB figures of Table 15.
"""
P = 21888242871839275222246405745257275088548364400416034343698204186575808495617  # BN254 scalar field


def inv(a, p=P):
    return pow(a % p, p - 2, p)


# polynomials are lists of coefficients, lowest degree first
def padd(a, b, p=P):
    n = max(len(a), len(b))
    return [((a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0)) % p for i in range(n)]


def pscale(a, c, p=P):
    return [(x * c) % p for x in a]


def pmul(a, b, p=P):
    r = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            r[i + j] = (r[i + j] + x * y) % p
    return r


def ptrim(a):
    while len(a) > 1 and a[-1] == 0:
        a = a[:-1]
    return a


def pdivmod(a, b, p=P):
    a, b = ptrim(a[:]), ptrim(b[:])
    if len(a) < len(b):
        return [0], a
    q = [0] * (len(a) - len(b) + 1)
    ib = inv(b[-1], p)
    for i in range(len(a) - len(b), -1, -1):
        c = (a[i + len(b) - 1] * ib) % p
        q[i] = c
        for j, y in enumerate(b):
            a[i + j] = (a[i + j] - c * y) % p
    return ptrim(q), ptrim(a[:len(b) - 1] or [0])


def lagrange(xs, ys, p=P):
    out = [0]
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        num, den = [1], 1
        for j, xj in enumerate(xs):
            if j != i:
                num = pmul(num, [(-xj) % p, 1], p)
                den = (den * (xi - xj)) % p
        out = padd(out, pscale(num, yi * inv(den, p) % p, p), p)
    return ptrim(out)


def r1cs_satisfied(A, B, C, z, p=P):
    dot = lambda row: sum(a * b for a, b in zip(row, z)) % p
    return all((dot(a) * dot(b) - dot(c)) % p == 0 for a, b, c in zip(A, B, C))


def qap_check(A, B, C, z, p=P):
    """Return (divides, remainder). Evaluation points r_i = 1..m."""
    m, nv = len(A), len(z)
    rs = list(range(1, m + 1))
    def combo(M):
        poly = [0]
        for j in range(nv):
            col = lagrange(rs, [M[i][j] for i in range(m)], p)
            poly = padd(poly, pscale(col, z[j], p), p)
        return poly
    a, b, c = combo(A), combo(B), combo(C)
    num = padd(pmul(a, b, p), pscale(c, p - 1, p), p)
    t = [1]
    for r in rs:
        t = pmul(t, [(-r) % p, 1], p)
    h, rem = pdivmod(num, t, p)
    return rem == [0], rem, h


def toy_circuit():
    """x^3 + x + 5 = y.  z = [1, y, x, v1, v2],  v1 = x*x,  v2 = v1*x.
    The classic worked example; a stand-in for the resolver-policy predicate."""
    A = [[0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [5, 0, 1, 0, 1]]
    B = [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 0, 0, 0, 0]]
    C = [[0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 1, 0, 0, 0]]
    return A, B, C


def toy_witness(x):
    v1 = x * x
    v2 = v1 * x
    y = v2 + x + 5
    return [1, y, x, v1, v2]
