# Chapter 4 reported values (generated)

## table13: Neuromorphic computing efficiency

| metric | loihi | gpu_a100 | gain | evidence | sw_repro | section |
|---|---|---|---|---|---|---|
| Inference latency | 0.8 ms | 2.1 ms | 2.6x | hw | no | 4.12, 4.25 |
| Energy per inference | 0.12 mJ | 1.8 mJ | 15x | hw | no | 4.25 |
| Spike encoding efficiency | 0.45 bits/spike | n/a | n/a | tb | partial | 4.27.4, 4.29.1 |

## table14: Attack simulation metrics

| attack | rate_pct | tool | evidence | sw_repro | section |
|---|---|---|---|---|---|
| DNS tunnelling | 99.5 | dnscat2 | tb | partial | 4.26.1 |
| C2 heartbeat | 98.7 | custom beacon generator | tb | partial | 4.26.4 |
| Side-channel (classical timing/power) | 100.0 | ChipWhisperer | hw | no | 4.26.3 |

## table15: Zero-knowledge attestation performance (Groth16 path)

| parameter | target | measured | evidence | sw_repro | section |
|---|---|---|---|---|---|
| Proof generation time | < 500 ms | 432 ms | tb | partial | 4.16, 4.27.2 |
| Proof verification time | < 100 ms | 78 ms | tb | partial | 4.27.2 |
| Soundness error | <= 2^-128 | 2^-132 | cfg | no | 4.23 |
| Proof size | <= 2 KB | 1.8 KB | tb | yes | 4.27.2 |

## table16: Federated learning performance and robustness (eps = 1.2)

| metric | target | achieved | test | evidence | sw_repro |
|---|---|---|---|---|---|
| Global model accuracy | >= 99.2% | 99.4% (+-0.12%, 95% CI) | Welch t-test, p < 0.01 | tb | partial |
| Privacy-utility trade-off | eps <= 1.5 | eps = 1.2 (utility > 98.7%) | KL divergence, p < 0.05 | tb | partial |
| Convergence efficiency | <= 10 epochs | 8 epochs (delta = 1e-3) | KS test, p < 0.05 | tb | partial |
| Client dropout resilience | >= 95% | 97.3% (up to 20% dropout) | chi-squared GoF, p < 0.01 | tb | partial |
| Adversarial robustness | >= 95% | 96.8% (GAN-crafted queries) | Mann-Whitney U, p < 0.01 | tb | partial |

## table17: Topological data analysis validation

| feature | benign | attack | p | section |
|---|---|---|---|---|
| Betti number beta_1 | 2.1 +- 0.3 | 5.8 +- 1.2 | p < 0.001 | 4.7-4.8 |
| Persistence entropy | 0.85 +- 0.1 | 1.92 +- 0.4 | p = 0.002 | 4.7.1 |
| Wasserstein distance | 1.3 +- 0.2 | 3.7 +- 0.9 | p < 0.001 | 4.7.2 |

## table17a: Closest published method per component and configuration used

| component | published | configuration | where |
|---|---|---|---|
| Federated aggregation | FedAvg + DP-SGD (Abadi et al., CCS 2016) | eps=1.2, C=3.5, sigma via moments accountant | 4.6.2, Eq. 36; 4.28.1, Table 16 |
| Byzantine-robust aggregation | Geometric median, IRLS | weighted geometric median + detector cross-check | 4.6.3, Eq. 38 |
| Homomorphic inference | Ring-LWE encryption | n=256, q=4096, 128-bit PQ target | 4.21, Eq. 58 |
| SNN detector | LIF neuron + pair-based STDP | log burst encoding, 3-layer LIF, 20 ms STDP window, Loihi | 4.9-4.11, Eqs. 43-46; 4.25, Table 13 |
| Topological anomaly detection | Persistent homology, VR filtration, p-Wasserstein | beta_1, entropy, Wasserstein; Bonferroni | 4.7-4.8, Eqs. 40-42; 4.28.2, Table 17 |
| ZK attestation | Groth16 zk-SNARK | ~1e5 constraints; SNARK and STARK paths | 4.15-4.18, Eqs. 50-52; 4.29, Table 15 |
| Side-channel noise injection | Adaptive noise injection | SNN-set amplitude/spectrum; adaptive threshold Eq. 48 | 4.13-4.14, Eqs. 47-49; 4.26.3 |
| Adversarial training data | GAN-generated synthetic queries | TV regulariser on generator | 4.6.1, Eq. 35; 4.26.2 |

## baselines_sec430: baselines_sec430

| name | accuracy_pct | fpr_pct | section | provenance | snr_reduction_pct |
|---|---|---|---|---|---|
| Centralized ML | 92.1 | 1.8 | 4.30.1 | retrained, same data/preproc/tuning budget |  |
| PAFA-DoH | 99.6 | 0.01 | 4.30.1, 4.30.3 | this work |  |
| Rule-based detector | 78.0 | >5 | 4.30.3 | reproduced |  |
| Static ML-KEM |  |  | 4.30.2 | same measurement setup | 60 |
| SNN-guided noise |  |  | 4.30.2 | this work | 98 |
