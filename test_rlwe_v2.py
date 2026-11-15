import unittest
import numpy as np
from pafa_v2.ring_lwe_v2 import RLWE, negacyclic_mul
from pafa_v2 import lattice_estimate_v2 as L


class TestRing(unittest.TestCase):
    def test_negacyclic_identity(self):
        n, q = 8, 97
        x = np.zeros(n, dtype=np.int64); x[1] = 1          # polynomial "x"
        xn1 = np.zeros(n, dtype=np.int64); xn1[n - 1] = 1  # x^(n-1)
        out = negacyclic_mul(x, xn1, q)                    # x^n = -1
        self.assertEqual(out[0], q - 1)
        self.assertTrue(np.all(out[1:] == 0))

    def test_commutative_distributive(self):
        rng = np.random.default_rng(0)
        n, q = 16, 4096
        a, b, c = (rng.integers(0, q, n) for _ in range(3))
        self.assertTrue(np.array_equal(negacyclic_mul(a, b, q), negacyclic_mul(b, a, q)))
        lhs = negacyclic_mul(a, (b + c) % q, q)
        rhs = (negacyclic_mul(a, b, q) + negacyclic_mul(a, c, q)) % q
        self.assertTrue(np.array_equal(lhs, rhs))


class TestRLWE(unittest.TestCase):
    def test_roundtrip(self):
        k = RLWE(seed=1)
        m = np.random.default_rng(2).integers(0, 2, k.n)
        self.assertTrue(np.array_equal(k.decrypt(k.encrypt(m)), m))

    def test_additive_hom_xor(self):
        k = RLWE(seed=3)
        r = np.random.default_rng(4)
        m1, m2 = r.integers(0, 2, k.n), r.integers(0, 2, k.n)
        s = k.add(k.encrypt(m1), k.encrypt(m2))
        self.assertTrue(np.array_equal(k.decrypt(s), (m1 + m2) % 2))

    def test_integer_plaintext_modulus(self):
        k = RLWE(n=256, q=4096, sigma=1.0, t=16, seed=5)
        r = np.random.default_rng(6)
        m1, m2 = r.integers(0, 4, k.n), r.integers(0, 4, k.n)
        s = k.add(k.encrypt(m1), k.encrypt(m2))
        self.assertTrue(np.array_equal(k.decrypt(s), m1 + m2))
        s3 = k.mul_plain_scalar(k.encrypt(m1), 3)
        self.assertTrue(np.array_equal(k.decrypt(s3), (3 * m1) % 16))

    def test_noise_grows_with_additions(self):
        k = RLWE(seed=7)
        m = np.zeros(k.n, dtype=np.int64)
        ct = k.encrypt(m)
        n1 = np.abs(k.noise(ct, m)).max()
        acc = ct
        for _ in range(63):
            acc = k.add(acc, k.encrypt(m))
        n64 = np.abs(k.noise(acc, m)).max()
        self.assertGreater(n64, n1)
        self.assertGreater(k.decrypt_budget_bits(acc, m), 0)

    def test_wrong_key_fails(self):
        k1, k2 = RLWE(seed=8), RLWE(seed=9)
        m = np.random.default_rng(1).integers(0, 2, k1.n)
        self.assertGreater(np.mean(k2.decrypt(k1.encrypt(m)) != m), 0.3)


class TestEstimator(unittest.TestCase):
    def test_kyber512_calibration(self):
        r = L.report(512, 3329, L.sigma_from_cbd(3))
        self.assertAlmostEqual(r["beta"], 406, delta=6)
        self.assertAlmostEqual(r["classical_bits"], 118, delta=2)

    def test_monotone_in_dimension(self):
        a = L.report(256, 4096, 3.2)["beta"]
        b = L.report(512, 4096, 3.2)["beta"]
        c = L.report(1024, 4096, 3.2)["beta"]
        self.assertTrue(a < b < c)


if __name__ == "__main__":
    unittest.main()
