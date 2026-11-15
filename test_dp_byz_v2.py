import math, unittest
import numpy as np
from pafa_v2 import dp_accountant_v2 as acc, dp_aggregation_v2 as agg, byzantine_v2 as byz


class TestAccountant(unittest.TestCase):
    def test_monotone_in_T(self):
        e = [acc.eps_from_rdp(T, 4.0, 1e-5)[0] for T in (10, 50, 200)]
        self.assertTrue(e[0] < e[1] < e[2])

    def test_monotone_in_sigma(self):
        e = [acc.eps_from_rdp(100, s, 1e-5)[0] for s in (2.0, 4.0, 8.0)]
        self.assertTrue(e[0] > e[1] > e[2])

    def test_closed_form_optimum(self):
        # continuous optimum: alpha* = 1 + sqrt(2 s^2 log(1/d)/T)
        T, s, d = 100, 5.0, 1e-5
        a = 1 + math.sqrt(2 * s * s * math.log(1 / d) / T)
        ref = T * a / (2 * s * s) + math.log(1 / d) / (a - 1)
        got, _ = acc.eps_from_rdp(T, s, d)
        self.assertAlmostEqual(got, ref, delta=0.02 * ref)

    def test_sigma_inverse(self):
        s = acc.sigma_for_target(8, 1.2, 1e-5)
        self.assertLessEqual(acc.eps_from_rdp(8, s, 1e-5)[0], 1.2 + 1e-6)
        self.assertGreater(acc.eps_from_rdp(8, s * 0.98, 1e-5)[0], 1.2)


class TestAggregation(unittest.TestCase):
    def test_clip(self):
        g = np.array([3.0, 4.0])
        self.assertAlmostEqual(np.linalg.norm(agg.clip_l2(g, 1.0)), 1.0)
        self.assertTrue(np.allclose(agg.clip_l2(g, 10.0), g))

    def test_noise_scale(self):
        rng = np.random.default_rng(1)
        ups = [np.zeros(2000) for _ in range(10)]
        a_in = agg.dp_aggregate(ups, 3.5, 1.0, rng, True)
        a_out = agg.dp_aggregate(ups, 3.5, 1.0, rng, False)
        self.assertAlmostEqual(a_in.std(), 3.5 / 10, delta=0.03)
        self.assertAlmostEqual(a_out.std(), 3.5, delta=0.2)
        self.assertEqual(agg.effective_noise_multiplier(1.0, 10, False), 10.0)


class TestByzantine(unittest.TestCase):
    def test_geomed_beats_mean(self):
        rng = np.random.default_rng(7)
        honest = rng.normal(1.0, 0.1, size=(40, 16))
        bad = rng.normal(-50.0, 1.0, size=(10, 16))      # 20 % Byzantine
        G = np.vstack([honest, bad])
        gm, _ = byz.geometric_median(G)
        err_gm = np.linalg.norm(gm - 1.0)
        err_mean = np.linalg.norm(G.mean(axis=0) - 1.0)
        self.assertLess(err_gm, 0.5)
        self.assertGreater(err_mean, 10 * err_gm)

    def test_weights_respected(self):
        G = np.array([[0.0], [10.0], [10.0]])
        g, _ = byz.geometric_median(G, w=[10.0, 1.0, 1.0])
        self.assertLess(g[0], 1.0)

    def test_cross_check(self):
        det = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        p = np.vstack([det, 1 - det, det])
        mask, agree = byz.cross_check(p, det, 0.9)
        self.assertEqual(mask.tolist(), [True, False, True])


if __name__ == "__main__":
    unittest.main()
