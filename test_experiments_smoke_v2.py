import importlib.util, sys, unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


atk = load("atk_v2", "experiments/pafa_attack_scenarios_v2.py")
fed = load("fed_v2", "experiments/pafa_federated_experiments_v2.py")
from pafa_v2 import windows_v2 as wv


class TestScenarios(unittest.TestCase):
    def test_features_shape_all_kinds(self):
        rng = np.random.default_rng(0)
        for k in wv.SCENARIOS:
            f = wv.window_features(*wv.gen_window(k, rng))
            self.assertEqual(f.shape, (len(wv.FEATURE_NAMES),))
            self.assertTrue(np.all(np.isfinite(f)))

    def test_tunnel_labels_longer_and_noisier_than_benign(self):
        rng = np.random.default_rng(1)
        i = wv.FEATURE_NAMES.index("ent_mean")
        b = np.mean([wv.window_features(*wv.gen_window("benign", rng))[i] for _ in range(60)])
        t = np.mean([wv.window_features(*wv.gen_window("tunnel", rng))[i] for _ in range(60)])
        self.assertGreater(t, b)


class TestSideChannelSim(unittest.TestCase):
    def test_sbox_reference_values(self):
        sb = atk.SBOX
        self.assertEqual([int(sb[0]), int(sb[1]), int(sb[0x53]), int(sb[0xff])], [0x63, 0x7c, 0xed, 0x16])
        self.assertEqual(len(set(sb.tolist())), 256)

    def test_cpa_degrades_with_noise(self):
        rng = np.random.default_rng(2)
        clean = atk.cpa_success(300, 1.0, rng, 20)
        noisy = atk.cpa_success(300, 40.0, rng, 20)
        self.assertGreater(clean, 0.9)
        self.assertLess(noisy, 0.3)


class TestFederatedPieces(unittest.TestCase):
    def test_batched_gradient_matches_finite_difference(self):
        rng = np.random.default_rng(3)
        th = fed.init_theta(rng)[None]
        X = rng.normal(size=(1, 10, fed.D)); y = (rng.random((1, 10)) > .5).astype(float)
        g = fed.grad_batched(th, X, y)[0]
        loss = lambda t: -np.mean(y[0] * np.log(fed.forward(t, X[0]) + 1e-12) + (1 - y[0]) * np.log(1 - fed.forward(t, X[0]) + 1e-12))
        for j in (0, 7, 100, fed.NP - 1):
            e = np.zeros(fed.NP); e[j] = 1e-5
            fd = (loss(th[0] + e) - loss(th[0] - e)) / 2e-5
            self.assertAlmostEqual(g[j], fd, places=5)

    def test_input_grad_matches_finite_difference(self):
        rng = np.random.default_rng(4)
        th = fed.init_theta(rng); X = rng.normal(size=(3, fed.D)); y = np.array([1., 0., 1.])
        g = fed.input_grad(th, X, y)
        loss = lambda Z: np.sum(-(y * np.log(fed.forward(th, Z)) + (1 - y) * np.log(1 - fed.forward(th, Z))))
        e = np.zeros_like(X); e[1, 5] = 1e-6
        self.assertAlmostEqual(g[1, 5], (loss(X + e) - loss(X - e)) / 2e-6, places=5)

    def test_clip_rows(self):
        d = np.array([[3., 4.], [0.3, 0.4]])
        c = fed.clip_rows(d, 1.0)
        self.assertAlmostEqual(np.linalg.norm(c[0]), 1.0); self.assertTrue(np.allclose(c[1], d[1]))


if __name__ == "__main__":
    unittest.main()
