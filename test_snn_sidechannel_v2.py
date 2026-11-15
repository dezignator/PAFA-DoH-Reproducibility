import unittest
import numpy as np
from pafa_v2 import snn_v2 as snn, sidechannel_v2 as sc


class TestEncoding(unittest.TestCase):
    def test_isi_equal_in_burst_and_monotone_in_norm(self):
        t, own = snn.log_burst_encode([0.0, 100.0], [1.0, 100.0], tau=2.0, alpha=1.0, n_spikes=4)
        for p, sq in ((0, 1.0), (1, 100.0)):
            tt = t[own == p]
            d = np.diff(tt)
            self.assertTrue(np.allclose(d, 2.0 * np.log1p(sq)))
        self.assertGreater(np.diff(t[own == 1])[0], np.diff(t[own == 0])[0])

    def test_sorted_and_count(self):
        t, own = snn.log_burst_encode([0, 1, 2], [1, 2, 3], 1.0, 1.0, [2, 3, 4])
        self.assertTrue(np.all(np.diff(t) >= 0))
        self.assertEqual(len(t), 9)
        self.assertEqual(int((own == 2).sum()), 4)


class TestLIF(unittest.TestCase):
    def test_noiseless_period(self):
        dt, tau_m = 0.01, 20.0
        v_rest, v_th, v_reset, I = 0.0, 15.0, 0.0, 20.0
        steps = int(500 / dt)
        sp, _ = snn.lif_simulate(np.full(steps, I), dt, tau_m, v_rest, v_th, v_reset, 0.0)
        isi = np.diff(np.array(sp[0])) * dt
        ref = snn.lif_period_analytic(I, tau_m, v_rest, v_th, v_reset)
        self.assertGreater(len(isi), 5)
        self.assertAlmostEqual(isi.mean(), ref, delta=0.05)

    def test_subthreshold_silent(self):
        sp, _ = snn.lif_simulate(np.full(20000, 10.0), 0.01, 20.0, 0.0, 15.0, 0.0, 2.0)
        self.assertEqual(len(sp[0]), 0)

    def test_refractory_limits_rate(self):
        dt = 0.01
        a, _ = snn.lif_simulate(np.full(50000, 200.0), dt, 20.0, 0, 15, 0, 0.0)
        b, _ = snn.lif_simulate(np.full(50000, 200.0), dt, 20.0, 0, 15, 0, 5.0)
        self.assertGreater(len(a[0]), len(b[0]))


class TestSTDP(unittest.TestCase):
    def test_pre_before_post_potentiates(self):
        self.assertGreater(snn.stdp_delta_w([10.0], [15.0], 0.01), 0)

    def test_post_before_pre_depresses(self):
        self.assertLess(snn.stdp_delta_w([15.0], [10.0], 0.01), 0)

    def test_window_value(self):
        dw = snn.stdp_delta_w([0.0], [20.0], 1.0, 20.0, 20.0)
        self.assertAlmostEqual(dw, np.exp(-1.0))

    def test_symmetric_pair_cancels(self):
        dw = snn.stdp_delta_w([0.0, 40.0], [20.0], 1.0, 20.0, 20.0)
        self.assertAlmostEqual(dw, 0.0, places=12)

    def test_nearest_le_all_to_all_magnitude(self):
        pre = [0, 5, 10, 15]
        post = [12, 30]
        a = snn.stdp_delta_w(pre, post, 1.0)
        n = snn.stdp_delta_w_nearest(pre, post, 1.0)
        self.assertNotEqual(a, n)

    def test_bounds(self):
        self.assertEqual(snn.apply_bounded(0.99, 0.5), 1.0)
        self.assertEqual(snn.apply_bounded(0.01, -0.5), 0.0)


class TestSideChannel(unittest.TestCase):
    def test_reduction(self):
        self.assertAlmostEqual(sc.snr_reduction_pct(10.0, 0.2), 98.0)

    def test_adaptive_threshold(self):
        v = sc.adaptive_threshold(1.0, 0.5, [0.0, 1.0])
        self.assertEqual(v.tolist(), [1.0, 1.5])
        with self.assertRaises(ValueError):
            sc.adaptive_threshold(1.0, 0.5, [1.2])

    def test_objective(self):
        rng = np.random.default_rng(0)
        s = rng.normal(size=1000)
        n = sc.noise_to_hit_target(s, rng.normal(size=1000), 0.05)
        self.assertTrue(sc.meets_objective(s, n, 0.0500001))
        self.assertFalse(sc.meets_objective(s, n * 0.5, 0.05))

    def test_error_trace_shape(self):
        E, th = sc.stdp_error_trace(np.full(100, 0.01), 0.1, np.random.default_rng(1))
        self.assertEqual(E.shape, (100,))
        self.assertAlmostEqual(th[-1], 1.0)


if __name__ == "__main__":
    unittest.main()
