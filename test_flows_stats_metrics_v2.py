import unittest
import numpy as np
from pafa_v2 import flows_v2 as fl, stats_v2 as st, metrics_v2 as m


class TestFlows(unittest.TestCase):
    def test_entropy(self):
        self.assertEqual(fl.label_entropy("aaaa"), 0.0)
        self.assertAlmostEqual(fl.label_entropy("abcd"), 2.0)
        self.assertEqual(fl.label_entropy(""), 0.0)

    def test_subdomain(self):
        self.assertEqual(fl.subdomain_label("a.b.example.org."), "a.b")
        self.assertEqual(fl.subdomain_label("example.org"), "")

    def test_cloud_shape_and_standardised(self):
        rng = np.random.default_rng(0)
        X = fl.flow_to_cloud(*fl.synth_benign(100, rng))
        self.assertEqual(X.shape, (100, 3))
        self.assertTrue(np.allclose(X.mean(axis=0), 0, atol=1e-9))

    def test_tunnel_entropy_higher(self):
        rng = np.random.default_rng(0)
        _, _, nb = fl.synth_benign(200, rng)
        _, _, nt = fl.synth_tunnel(200, rng)
        eb = np.mean([fl.label_entropy(fl.subdomain_label(q)) for q in nb])
        et = np.mean([fl.label_entropy(fl.subdomain_label(q, 3)) for q in nt])
        self.assertGreater(et, eb)

    def test_windows(self):
        X = np.arange(30).reshape(10, 3)
        self.assertEqual(len(fl.windows(X, 4, 2)), 4)


class TestStats(unittest.TestCase):
    def test_perm_detects_shift(self):
        rng = np.random.default_rng(1)
        p, _ = st.permutation_pvalue(rng.normal(0, 1, 40), rng.normal(2, 1, 40), n_resamples=2000)
        self.assertLess(p, 0.01)

    def test_perm_null_uniformish(self):
        rng = np.random.default_rng(2)
        p, _ = st.permutation_pvalue(rng.normal(0, 1, 40), rng.normal(0, 1, 40), n_resamples=2000, seed=5)
        self.assertGreater(p, 0.01)

    def test_bonferroni(self):
        thr, rej = st.bonferroni([0.0004, 0.002, 0.04], 0.05)
        self.assertAlmostEqual(thr, 0.05 / 3)
        self.assertEqual(rej, [True, True, False])

    def test_ci_contains_mean(self):
        x = np.random.default_rng(3).normal(5, 1, 200)
        lo, hi = st.bootstrap_ci(x)
        self.assertTrue(lo < 5.0 < hi)


class TestMetrics(unittest.TestCase):
    def test_confusion(self):
        tp, tn, fp, fn = m.confusion([1, 1, 0, 0, 1], [1, 0, 0, 1, 1])
        self.assertEqual((tp, tn, fp, fn), (2, 1, 1, 1))
        self.assertAlmostEqual(m.accuracy(tp, tn, fp, fn), 0.6)
        self.assertAlmostEqual(m.fpr(tp, tn, fp, fn), 0.5)

    def test_threshold_perfect(self):
        s = np.r_[np.random.default_rng(0).normal(0, 1, 500), np.random.default_rng(1).normal(8, 1, 500)]
        y = np.r_[np.zeros(500), np.ones(500)]
        g, T, F = m.best_threshold(s, y)
        self.assertGreater(T, 0.99)
        self.assertLess(F, 0.01)
        self.assertGreater(m.auc(s, y), 0.999)

    def test_rule_of_three(self):
        n = m.min_negatives_for_fpr_bound(1e-4, 0.95, fp=0)
        self.assertTrue(29000 < n < 31000)          # ~ 29,956 (exact CP), 3/1e-4 = 30,000

    def test_wilson(self):
        lo, hi = m.wilson(0, 1000)
        self.assertAlmostEqual(lo, 0.0, places=12)
        self.assertTrue(0.002 < hi < 0.005)


if __name__ == "__main__":
    unittest.main()
