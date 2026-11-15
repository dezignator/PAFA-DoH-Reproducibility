#!/usr/bin/env python3
"""PAFA-DoH Chapter 4: performance benchmarks and consolidated experiment report (v2).

1. (optional, --run) executes the federated and attack suites.
2. Benchmarks the cryptographic and proof components that exist as software
   references in this artifact (all measured on the machine running the script):
     B1  Ring-LWE encrypt / decrypt / add vs ring dimension, decryption success
     B2  Merkle + FRI commit / verify time and proof size vs polynomial degree
     B3  R1CS -> QAP divisibility check vs constraint count (scaling exponent)
     B4  DP accountant: epsilon vs rounds and noise multiplier
     B5  core-SVP estimate for candidate Ring-LWE parameter sets
     B6  federated training throughput vs client population
     B7  C LIF kernel throughput (if a C compiler is present)
3. Reads the CSV outputs of the two suites and writes one report
   (Markdown + self-contained HTML) with a claim-by-claim comparison table
   between the numbers quoted in Chapter 4 and what the simulations show.

Everything is measured or simulated here. Hardware-bound figures of the thesis
(Loihi/A100 latency and energy, ChipWhisperer SNR, Groth16 prover times) are
listed as 'not reproducible here' rather than approximated.

usage:
    python3 experiments/pafa_report_builder_v2.py --run --out build/report
    python3 experiments/pafa_report_builder_v2.py --fed build/fed --attack build/attack --out build/report
"""
import argparse
import base64
import html
import importlib.util
import math
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pafa_v2.ring_lwe_v2 import RLWE                                   # noqa: E402
from pafa_v2 import merkle_fri_v2 as fri, zk_r1cs_v2 as zk             # noqa: E402
from pafa_v2.dp_accountant_v2 import eps_from_rdp, sigma_for_target     # noqa: E402
from pafa_v2 import lattice_estimate_v2 as lat                          # noqa: E402


def best_of(f, reps=3):
    t = []
    for _ in range(reps):
        a = time.perf_counter(); f(); t.append(time.perf_counter() - a)
    return min(t)


# --------------------------------------------------------------------------
# B1 Ring-LWE
# --------------------------------------------------------------------------
def b1_rlwe():
    rows = []
    for n, q in [(256, 4096), (512, 4096), (1024, 12289)]:
        k = RLWE(n=n, q=q, sigma=3.2, seed=1)
        rng = np.random.default_rng(2)
        m = rng.integers(0, 2, n)
        ct = k.encrypt(m)
        t_enc = best_of(lambda: k.encrypt(m), 5)
        t_dec = best_of(lambda: k.decrypt(ct), 5)
        ct2 = k.encrypt(m)
        t_add = best_of(lambda: k.add(ct, ct2), 20)
        ok, N = 0, 200
        for i in range(N):
            m1, m2 = rng.integers(0, 2, n), rng.integers(0, 2, n)
            ok += int(np.array_equal(k.decrypt(k.add(k.encrypt(m1), k.encrypt(m2))), (m1 + m2) % 2))
        acc = np.zeros(n, dtype=np.int64); c = k.encrypt(acc)
        for _ in range(255):
            c = k.add(c, k.encrypt(acc))
        rows.append({"n": n, "q": q, "enc (ms)": f"{t_enc*1e3:.2f}", "dec (ms)": f"{t_dec*1e3:.2f}",
                     "add (us)": f"{t_add*1e6:.1f}", "ciphertext (bytes)": 2 * n * math.ceil(math.log2(q) / 8),
                     "hom-add success (200 trials)": f"{ok}/{N}",
                     "noise budget after 256 adds (bits)": f"{k.decrypt_budget_bits(c, acc):.1f}"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# B2 Merkle + FRI
# --------------------------------------------------------------------------
def b2_fri():
    import random
    rows = []
    for k in [4, 5, 6, 7, 8]:
        d1 = 2 ** k
        for blow in [4, 8]:
            coeffs = [random.randrange(fri.P) for _ in range(d1)]
            t_prove = best_of(lambda: fri.fri_prove(coeffs, blowup=blow), 2)
            layers, trees, alphas = fri.fri_prove(coeffs, blowup=blow)
            nq = 20
            t_ver = best_of(lambda: fri.fri_verify(layers, trees, alphas, nq, random.Random(1)), 2)
            ok, _ = fri.fri_verify(layers, trees, alphas, nq, random.Random(1))
            size = 0
            for r in range(len(layers) - 1):
                depth = int(math.log2(len(layers[r])))
                size += nq * 2 * (8 + 32 * depth)
            size += 32 * len(trees) + 8 * blow
            rows.append({"degree bound": d1, "blow-up": blow, "domain": len(layers[0]), "rounds": len(layers) - 1,
                         "prove (ms)": f"{t_prove*1e3:.1f}", f"verify, {nq} queries (ms)": f"{t_ver*1e3:.1f}",
                         "proof size (KB, est.)": f"{size/1024:.1f}", "accepted": ok,
                         "conjectured soundness (bits)": f"{fri.fri_soundness_bits_conjectured(blow, nq):.0f}"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# B3 QAP scaling
# --------------------------------------------------------------------------
def chain_circuit(m, x=3):
    """z = [1, x, v1..vm]; constraints  v_{i-1} * x = v_i  with v0 = x."""
    nv = m + 2
    A = [[0] * nv for _ in range(m)]; B = [[0] * nv for _ in range(m)]; C = [[0] * nv for _ in range(m)]
    for i in range(m):
        A[i][1 if i == 0 else 1 + i] = 1          # v_{i-1}  (v0 = x at column 1)
        B[i][1] = 1                               # x
        C[i][2 + i] = 1                           # v_i
    z = [1, x]
    v = x
    for _ in range(m):
        v = v * x % zk.P; z.append(v)
    return A, B, C, z


def b3_qap():
    rows, xs, ys = [], [], []
    for m in [4, 8, 16, 24, 32, 48]:
        A, B, C, z = chain_circuit(m)
        assert zk.r1cs_satisfied(A, B, C, z)
        t = best_of(lambda: zk.qap_check(A, B, C, z), 2)
        ok = zk.qap_check(A, B, C, z)[0]
        rows.append({"constraints m": m, "R1CS satisfied": True, "QAP divisible": ok, "time (ms)": f"{t*1e3:.1f}"})
        xs.append(math.log(m)); ys.append(math.log(t))
    slope = np.polyfit(xs, ys, 1)[0]
    df = pd.DataFrame(rows)
    df.loc[len(df)] = ["fitted exponent (time ~ m^e)", "", "", f"e = {slope:.2f}"]
    return df


# --------------------------------------------------------------------------
# B4 / B5 accountant and lattice
# --------------------------------------------------------------------------
def b4_dp():
    rows = []
    for T in [8, 30, 100, 300]:
        row = {"rounds T": T}
        for s in [5, 10, 20, 40]:
            row[f"eps @ sigma={s}"] = f"{eps_from_rdp(T, s, 1e-5)[0]:.2f}"
        row["sigma for eps=1.2"] = f"{sigma_for_target(T, 1.2, 1e-5):.1f}"
        rows.append(row)
    return pd.DataFrame(rows)


def b5_lattice():
    rows = []
    for n, q, s, lab in [(256, 4096, 3.2, "thesis Table 17a pair"), (512, 4096, 3.2, "n doubled"),
                         (512, 3329, lat.sigma_from_cbd(3), "ML-KEM-512-like (calibration)"),
                         (768, 3329, 1.0, "ML-KEM-768-like"), (1024, 12289, 3.2, "NewHope-1024-like")]:
        r = lat.report(n, q, s)
        rows.append({"parameter set": lab, "n": n, "q": q, "sigma": round(s, 2), "BKZ beta": r["beta"],
                     "core-SVP classical (bits)": r["classical_bits"], "core-SVP quantum (bits)": r["quantum_bits"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# B6 federated throughput, B7 C kernel
# --------------------------------------------------------------------------
def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def b6_throughput(tmp):
    fed = load_module(ROOT / "experiments" / "pafa_federated_experiments_v2.py", "fedx")
    data = fed.build_data(tmp / "thr_data.npz", n_train=6400, n_test=500, n_val=200, n_public=200)
    sc = fed.Scaler(data["pub_X"])
    rows = []
    for K in [100, 400, 1600]:
        sim = fed.FedSim(data, sc, K, 0)
        for agg, eps in [("mean", None), ("clipmean", None), ("geomed", None), ("dpfl", 1.2), ("geomed_xcheck", None)]:
            t = time.perf_counter()
            fed.run_fl(sim, 0, rounds=5, aggregator=agg, eps=eps, track=False)
            rows.append({"clients K": K, "aggregator": agg, "ms per round": f"{(time.perf_counter()-t)/5*1e3:.1f}"})
    df = pd.DataFrame(rows).pivot(index="aggregator", columns="clients K", values="ms per round").reset_index()
    return df


def b7_c_kernel(tmp):
    cc = shutil.which("cc") or shutil.which("gcc")
    if not cc:
        return pd.DataFrame([{"note": "no C compiler found"}])
    exe = tmp / "lif_kernel_v2"
    r = subprocess.run([cc, "-O2", "-std=c99", "-o", str(exe), str(ROOT / "c" / "lif_kernel_v2.c"), "-lm"],
                       capture_output=True, text=True)
    if r.returncode:
        return pd.DataFrame([{"note": "compile failed: " + r.stderr[:200]}])
    rows = []
    for steps in [200]:
        o = subprocess.run([str(exe), "--bench", "128", "1024", str(steps)], capture_output=True, text=True).stdout
        rows.append({"layout": "128 cores x 1024 neurons", "steps": steps, "result": o.strip()})
    st = subprocess.run([str(exe), "--selftest"], capture_output=True, text=True).stdout.strip().splitlines()
    rows.append({"layout": "selftest", "steps": "-", "result": " | ".join(st)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# claim comparison
# --------------------------------------------------------------------------
def first_num(s):
    import re
    m = re.search(r"-?\d+\.?\d*", str(s))
    return float(m.group()) if m else float("nan")


def claims(fed_dir, atk_dir, b5):
    rows = []
    def add(item, thesis, here, note):
        rows.append({"Chapter 4 item": item, "thesis value": thesis, "this simulation": here, "reading": note})
    try:
        e2 = pd.read_csv(fed_dir / "e2_privacy_utility.csv")
        r = e2[e2.eps_target.astype(str) == "1.2"].iloc[0]
        add("Accuracy at eps=1.2 (Table 16)", ">= 99.2 % target, 99.4 % achieved",
            f"known families {r.acc_known} (K=400)", "depends on client population; see E6")
        e6 = pd.read_csv(fed_dir / "e6_client_scale.csv")
        ks = "; ".join(f"K={int(a)}: {b}" for a, b in zip(e6.clients_K, e6["acc_known_at_eps_1.2"]))
        add("Client population needed at eps=1.2", "number of clients not stated", ks,
            "noise on the average falls as 1/K; K is a first-order parameter")
        e1 = pd.read_csv(fed_dir / "e1_convergence.csv")
        r = e1[e1.method.str.contains("FedAvg")].iloc[0]
        add("Convergence epochs (Table 16)", "8 epochs, target <= 10",
            f"FedAvg reaches 95 % (known families) in rounds {r['rounds_to_95%(known)']}", "easy synthetic task; not comparable in absolute terms")
        e3 = pd.read_csv(fed_dir / "e3_dropout.csv")
        r = e3[e3.dropout == "20%"].iloc[0]
        add("Dropout resilience (Table 16)", "97.3 % recovery at up to 20 % dropout",
            f"{r.recovery_vs_0_percent_dropout if hasattr(r,'recovery_vs_0_percent_dropout') else r.iloc[-1]} of 0 %-dropout accuracy at 20 %",
            "FedAvg is insensitive to random absence here")
        e4 = pd.read_csv(fed_dir / "e4_byzantine.csv")
        r = e4[(e4.attack == "signflip") & (e4.byz_frac == "20%")].iloc[0]
        add("Byzantine robustness (Sec 4.6.3)", "geometric median + cross-check, fraction in 4.28.1",
            f"20 % sign-flip: mean {r['mean']}, geomed {r['geomed']}, +cross-check {r['geomed_xcheck']}",
            "plain mean collapses; geometric median holds")
        e5 = pd.read_csv(fed_dir / "e5_adversarial.csv")
        a, b = e5.iloc[0], e5.iloc[1]
        add("Adversarial robustness (Table 16)", "96.8 % under GAN-crafted queries",
            f"recall@0.5 FGSM: standard {a['recall@0.5']}, augmented {b['recall@0.5']}",
            "FGSM augmentation, not the thesis GAN; same direction of effect")
    except Exception as ex:                                     # noqa: BLE001
        add("federated suite", "-", f"not available ({ex})", "run the suite first")
    try:
        a2 = pd.read_csv(atk_dir / "a2_detector_comparison.csv")
        r = a2[a2.detector.str.startswith("PAFA")].iloc[0]
        add("Tunnelling detection (Table 14: 99.5 %)", "99.5 %", f"PAFA fusion {r['TPR tunnel']}, FPR {r['FPR (benign test)']}",
            "simulated dnscat2-like traffic")
        add("C2 heartbeat (Table 14: 98.7 %)", "98.7 %", f"PAFA fusion {r['TPR beacon']}", "simulated periodic beacon")
        add("Zero-day / evasive family", "'identifying zero-day attacks' (Sec 4.30.3)",
            f"mimic: PAFA fusion {r['TPR mimic']}",
            f"hardest family; TDA alone {a2[a2.detector.str.startswith('TDA')].iloc[0]['TPR mimic']}, "
            f"statistical {a2[a2.detector.str.startswith('Statistical')].iloc[0]['TPR mimic']}")
        t4 = pd.read_csv(atk_dir / "a4_topology.csv")
        r = t4[(t4.scenario == "mimic") & (t4.feature == "betti1")].iloc[0]
        add("Betti-1 separation (Table 17: 2.1 vs 5.8)", "2.1 ± 0.3 vs 5.8 ± 1.2",
            f"mimic: {r['benign mean ± sd']} vs {r['attack mean ± sd']} (delay-embedded IAT)", "different representation, different scale")
        a6 = pd.read_csv(atk_dir / "a6_sidechannel_cpa.csv")
        r = a6[a6.countermeasure.str.contains("x50") & ~a6.countermeasure.str.contains("adaptive|SNN")].iloc[0]
        add("SNR reduction 98 % (Sec 4.14)", "98 % (ChipWhisperer)",
            f"x50 injection: {r['ratio reduction']} by construction; CPA success at n=800: {r['n=800']}", "model; hardware result not reproduced")
    except Exception as ex:                                     # noqa: BLE001
        add("attack suite", "-", f"not available ({ex})", "run the suite first")
    r = b5.iloc[0]
    add("128-bit PQ security of Ring-LWE layer (Sec 4.21)", "n=256, q=4096, 128-bit target",
        f"core-SVP estimate {r['core-SVP classical (bits)']} classical / {r['core-SVP quantum (bits)']} quantum bits",
        "parameter pair does not reach the target under this estimate")
    add("Loihi 0.8 ms / 0.12 mJ vs A100 (Table 13)", "2.6x / 15x", "not reproducible without the hardware", "CPU numbers in B7 are a baseline only")
    add("Groth16 432 ms / 78 ms / 1.8 KB (Table 15)", "432 / 78 / 1.8", "not reproducible: toy QAP and FRI only", "B2/B3 give scaling shapes, not these times")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# report assembly
# --------------------------------------------------------------------------
CSS = """body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:1150px;margin:2rem auto;padding:0 1rem;color:#1b1f24;line-height:1.45}
h1{font-size:1.7rem}h2{margin-top:2.2rem;border-bottom:1px solid #d0d7de;padding-bottom:.2rem}h3{margin-top:1.4rem}
table{border-collapse:collapse;font-size:.82rem;margin:.6rem 0 1.2rem;width:100%}th,td{border:1px solid #d0d7de;padding:.28rem .5rem;text-align:left}
th{background:#f3f5f7}tr:nth-child(even) td{background:#fafbfc}.note{background:#fff8e6;border-left:4px solid #d4a72c;padding:.5rem .8rem;margin:1rem 0}
img{max-width:100%;border:1px solid #d0d7de;margin:.4rem 0}code{background:#f3f5f7;padding:0 .2rem}"""


class Report:
    def __init__(self):
        self.md, self.html = [], []

    def h(self, level, text):
        self.md.append(f"\n{'#'*level} {text}\n"); self.html.append(f"<h{level}>{html.escape(text)}</h{level}>")

    def p(self, text, note=False):
        self.md.append(text + "\n")
        self.html.append(f"<div class='note'>{html.escape(text)}</div>" if note else f"<p>{html.escape(text)}</p>")

    def table(self, df, title=None):
        if title:
            self.h(3, title)
        self.md.append(df.to_markdown(index=False) + "\n")
        self.html.append(df.to_html(index=False, escape=True, border=0))

    def image(self, path, alt):
        path = Path(path)
        if not path.exists():
            return
        self.md.append(f"![{alt}]({path.name})\n")
        b = base64.b64encode(path.read_bytes()).decode()
        self.html.append(f"<img alt='{html.escape(alt)}' src='data:image/png;base64,{b}'>")

    def write(self, out, stem):
        (out / f"{stem}.md").write_text("\n".join(self.md))
        (out / f"{stem}.html").write_text(
            f"<!doctype html><meta charset='utf-8'><title>{stem}</title><style>{CSS}</style>\n" + "\n".join(self.html))


def csv_table(rep, path, title):
    p = Path(path)
    if p.exists():
        rep.table(pd.read_csv(p), title)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="build/report")
    ap.add_argument("--fed", default="build/fed")
    ap.add_argument("--attack", default="build/attack")
    ap.add_argument("--run", action="store_true", help="run both experiment suites first")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fed_dir, atk_dir = Path(a.fed), Path(a.attack)
    if a.run:
        q = ["--quick"] if a.quick else []
        subprocess.run([sys.executable, str(ROOT / "experiments" / "pafa_federated_experiments_v2.py"), "--out", str(fed_dir)] + q, check=True)
        subprocess.run([sys.executable, str(ROOT / "experiments" / "pafa_attack_scenarios_v2.py"), "--out", str(atk_dir)] + q, check=True)

    t0 = time.time()
    print("benchmarks ...")
    b1, b2, b3, b4, b5 = b1_rlwe(), b2_fri(), b3_qap(), b4_dp(), b5_lattice()
    b6, b7 = b6_throughput(out), b7_c_kernel(out)
    for n, d in [("b1_rlwe", b1), ("b2_fri", b2), ("b3_qap", b3), ("b4_dp", b4), ("b5_lattice", b5), ("b6_fed_throughput", b6), ("b7_c_kernel", b7)]:
        d.to_csv(out / f"{n}_v2.csv", index=False)
    cl = claims(fed_dir, atk_dir, b5)
    cl.to_csv(out / "claims_vs_simulation_v2.csv", index=False)

    R = Report()
    R.h(1, "PAFA-DoH Chapter 4: experiments, attack scenarios and performance (v2 artifact)")
    R.p("All numbers below were produced by running the scripts in this repository on the machine described at the end. "
        "Traffic is synthetic (pafa_v2/windows_v2.py), the model is a small MLP over 17 window features, and cryptographic "
        "components are didactic software references. These are not the values of Tables 13-17 of the thesis; they show how "
        "each mechanism behaves under controlled conditions, with uncertainty, and where the thesis figures depend on "
        "hardware or testbed data that this repository does not contain.", note=True)
    R.h(2, "1. Claim-by-claim comparison")
    R.table(cl)
    R.h(2, "2. Federated training (E1-E6)")
    R.p("Setup: label-skewed clients (Dirichlet 0.7), 2 local epochs per round, MLP 17-24-1, client-level clipping C=3.5, "
        "Gaussian noise inside the average, delta=1e-5, accountant of Eq. 60. Training families: benign, tunnel, beacon, "
        "slow_exfil; 'mimic' is held out and never seen in training.")
    for f, t in [("e1_convergence", "E1 Convergence and final quality"), ("e2_privacy_utility", "E2 Privacy-utility trade-off"),
                 ("e3_dropout", "E3 Client dropout"), ("e4_byzantine", "E4 Byzantine clients (accuracy on known families)"),
                 ("e5_adversarial", "E5 Evasion robustness"), ("e6_client_scale", "E6 Client population at eps=1.2")]:
        csv_table(R, fed_dir / f"{f}.csv", t)
    R.image(fed_dir / "e1_convergence.png", "E1 convergence")
    R.h(2, "3. Attack scenarios and detection (A1-A7)")
    R.p("Every detector is thresholded at the same 1 % false-positive budget on a separate benign calibration set; "
        "TPR and FPR carry Wilson 95 % intervals. The supervised baseline sees only tunnel and beacon during training.")
    for f, t in [("a1_catalogue", "A1 Scenario catalogue"), ("a2_detector_comparison", "A2 Detector comparison"),
                 ("a3_ablation", "A3 Fusion ablation"), ("a4_topology", "A4 Topological features (permutation test)"),
                 ("a5_lifecycle", "A5 Multi-phase lifecycle"), ("a6_sidechannel_cpa", "A6 Simulated CPA key recovery"),
                 ("a6_noise_cost", "A6b Injected noise power"), ("a7_cost", "A7 CPU cost per window")]:
        csv_table(R, atk_dir / f"{f}.csv", t)
    R.image(atk_dir / "a2_roc.png", "ROC curves"); R.image(atk_dir / "a5_lifecycle.png", "lifecycle")
    R.h(2, "4. Component benchmarks (B1-B7)")
    R.table(b1, "B1 Ring-LWE reference implementation")
    R.table(b2, "B2 Merkle + FRI (educational, Goldilocks field)")
    R.table(b3, "B3 R1CS to QAP divisibility check (chain circuit)")
    R.table(b4, "B4 Moments-accountant epsilon (delta=1e-5)")
    R.table(b5, "B5 Ring-LWE parameter estimate (core-SVP, primal uSVP)")
    R.table(b6, "B6 Federated training: ms per round vs client population")
    R.table(b7, "B7 C LIF kernel (CPU)")
    R.h(2, "5. Environment")
    R.p(f"Python {platform.python_version()}, numpy {np.__version__}, {platform.platform()}, processor: {platform.processor() or 'n/a'}; "
        f"report built in {time.time()-t0:.0f}s (benchmarks only).")
    R.write(out, "PAFA_Chapter4_Experiment_Report_v2")
    print("wrote", out / "PAFA_Chapter4_Experiment_Report_v2.html")


if __name__ == "__main__":
    main()
