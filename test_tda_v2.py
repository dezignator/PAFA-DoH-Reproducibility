import unittest
import numpy as np
from pafa_v2 import tda_v2 as tda


def circle(n, r=1.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    P = np.c_[r * np.cos(th), r * np.sin(th)]
    return P + rng.normal(0, noise, P.shape)


class TestPH(unittest.TestCase):
    def test_circle_one_long_h1(self):
        dg = tda.vr_persistence(circle(24), max_scale=2.0)
        h1 = tda.finite(dg[1])
        life = h1[:, 1] - h1[:, 0]
        self.assertGreaterEqual(len(h1), 1)
        self.assertGreater(life.max(), 0.8)          # ~ sqrt(3) - small chord
        self.assertEqual(int((life > 0.5).sum()), 1)

    def test_two_clusters_h0(self):
        rng = np.random.default_rng(3)
        X = np.vstack([rng.normal(0, 0.05, (15, 2)), rng.normal(5, 0.05, (15, 2))])
        dg = tda.vr_persistence(X, max_dim=0)[0]
        fin = tda.finite(dg)
        self.assertEqual(int(np.sum(np.isinf(dg[:, 1]))), 1)
        self.assertGreater(fin[:, 1].max(), 4.0)      # the bridge between clusters
        self.assertEqual(int((fin[:, 1] > 1.0).sum()), 1)

    def test_h0_count(self):
        X = np.random.default_rng(1).normal(size=(20, 3))
        dg = tda.vr_persistence(X, max_dim=0)[0]
        self.assertEqual(len(dg), 20)                  # n bars: n-1 finite + 1 inf

    def test_blob_has_no_long_cycle(self):
        X = np.random.default_rng(2).normal(size=(30, 2))
        dg = tda.vr_persistence(X, max_scale=1.5)
        h1 = tda.finite(dg[1])
        life = h1[:, 1] - h1[:, 0] if len(h1) else np.array([0.0])
        self.assertLess(life.max(), 0.6)

    def test_betti_at(self):
        dg = np.array([[0.1, 0.9], [0.2, 0.3], [0.5, np.inf]])
        self.assertEqual(tda.betti_at(dg, 0.25), 2)
        self.assertEqual(tda.betti_at(dg, 0.6), 2)
        self.assertEqual(tda.betti_at(dg, 0.05), 0)

    def test_entropy_uniform(self):
        dg = np.array([[0, 1.0], [0, 1.0], [0, 1.0], [0, 1.0]])
        self.assertAlmostEqual(tda.persistence_entropy(dg), np.log(4))
        self.assertEqual(tda.persistence_entropy(np.zeros((0, 2))), 0.0)


class TestWasserstein(unittest.TestCase):
    def test_identity_zero(self):
        d = np.array([[0.0, 1.0], [0.2, 0.9]])
        self.assertAlmostEqual(tda.wasserstein(d, d), 0.0, places=12)

    def test_single_point_to_diagonal(self):
        d = np.array([[0.0, 2.0]])
        self.assertAlmostEqual(tda.wasserstein(d, np.zeros((0, 2)), p=2), 2.0 / np.sqrt(2.0))
        self.assertAlmostEqual(tda.wasserstein(d, np.zeros((0, 2)), p=2, ground="linf"), 1.0)

    def test_symmetry_and_triangle(self):
        rng = np.random.default_rng(0)
        def rd(n):
            b = rng.uniform(0, 1, n); return np.c_[b, b + rng.uniform(0.05, 1, n)]
        a, b, c = rd(5), rd(6), rd(4)
        ab, ba = tda.wasserstein(a, b), tda.wasserstein(b, a)
        self.assertAlmostEqual(ab, ba, places=10)
        self.assertLessEqual(tda.wasserstein(a, c), ab + tda.wasserstein(b, c) + 1e-9)

    def test_translation_cost(self):
        a = np.array([[0.0, 4.0]]); b = np.array([[0.0, 4.5]])
        self.assertAlmostEqual(tda.wasserstein(a, b, p=2), 0.5)


if __name__ == "__main__":
    unittest.main()
