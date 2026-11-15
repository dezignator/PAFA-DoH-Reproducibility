import random, unittest
from pafa_v2 import zk_r1cs_v2 as zk, merkle_fri_v2 as fri


class TestR1CS(unittest.TestCase):
    def test_valid_witness(self):
        A, B, C = zk.toy_circuit()
        z = zk.toy_witness(3)
        self.assertEqual(z[1], 35)
        self.assertTrue(zk.r1cs_satisfied(A, B, C, z))
        ok, rem, h = zk.qap_check(A, B, C, z)
        self.assertTrue(ok)

    def test_invalid_witness(self):
        A, B, C = zk.toy_circuit()
        z = zk.toy_witness(3)
        z[1] = 36
        self.assertFalse(zk.r1cs_satisfied(A, B, C, z))
        ok, rem, h = zk.qap_check(A, B, C, z)
        self.assertFalse(ok)

    def test_many_x(self):
        A, B, C = zk.toy_circuit()
        for x in (0, 1, 2, 7, 123456789):
            self.assertTrue(zk.qap_check(A, B, C, zk.toy_witness(x))[0])

    def test_lagrange(self):
        p = zk.lagrange([1, 2, 3], [5, 11, 19])      # 5,11,19 = x^2+3x+1 at 1,2,3
        self.assertEqual(p, [1, 3, 1])

    def test_poly_div(self):
        a = zk.pmul([zk.P - 1, 1], [zk.P - 2, 1])    # (x-1)(x-2)
        q, r = zk.pdivmod(a, [zk.P - 1, 1])
        self.assertEqual(r, [0])
        self.assertEqual(q, [zk.P - 2, 1])


class TestMerkle(unittest.TestCase):
    def test_open_verify(self):
        vals = list(range(16))
        t = fri.Merkle(vals)
        for i in (0, 5, 15):
            self.assertTrue(fri.Merkle.verify(t.root, i, vals[i], t.open(i)))
        self.assertFalse(fri.Merkle.verify(t.root, 5, 999, t.open(5)))
        self.assertFalse(fri.Merkle.verify(t.root, 4, vals[5], t.open(5)))


class TestFRI(unittest.TestCase):
    def test_root_of_unity(self):
        w = fri.root_of_unity(64)
        self.assertEqual(pow(w, 64, fri.P), 1)
        self.assertEqual(pow(w, 32, fri.P), fri.P - 1)

    def test_fold_halves_degree(self):
        coeffs = [random.randrange(fri.P) for _ in range(8)]     # degree 7
        layers, trees, alphas = fri.fri_prove(coeffs, blowup=4)
        self.assertEqual(len(layers[0]), 32)
        self.assertEqual(len(layers[-1]), 4)
        self.assertTrue(all(v == layers[-1][0] for v in layers[-1]))

    def test_low_degree_accepts(self):
        coeffs = [random.randrange(fri.P) for _ in range(16)]
        layers, trees, alphas = fri.fri_prove(coeffs, blowup=8)
        ok, msg = fri.fri_verify(layers, trees, alphas, 20, random.Random(1))
        self.assertTrue(ok, msg)

    def test_high_degree_rejected(self):
        n = 128
        evals = [random.randrange(fri.P) for _ in range(n)]      # not low degree
        layers = [evals]
        trees = [fri.Merkle(evals)]
        alphas = []
        tr = trees[0].root
        while len(layers[-1]) > 8:
            a = fri._challenge(tr); alphas.append(a)
            layers.append(fri.fold(layers[-1], a)); trees.append(fri.Merkle(layers[-1])); tr = trees[-1].root
        ok, msg = fri.fri_verify(layers, trees, alphas, 20, random.Random(1))
        self.assertFalse(ok)

    def test_tampered_fold_rejected(self):
        coeffs = [random.randrange(fri.P) for _ in range(16)]
        layers, trees, alphas = fri.fri_prove(coeffs, blowup=8)
        layers[1] = layers[1][:]
        layers[1][3] = (layers[1][3] + 1) % fri.P
        trees[1] = fri.Merkle(layers[1])
        ok, msg = fri.fri_verify(layers, trees, alphas, 400, random.Random(2))
        self.assertFalse(ok)

    def test_soundness_formula(self):
        self.assertEqual(fri.fri_soundness_bits_conjectured(8, 40), 120.0)


if __name__ == "__main__":
    unittest.main()
