# Threat model to component to test (Chapter 4, Secs 4.2-4.3)

| Adversary capability (thesis) | Component | Evidence in thesis | Code here | Stated limit |
|---|---|---|---|---|
| Tunnelling that matches benign statistics (4.2.1) | VR persistence + Wasserstein score | Table 17, Table 14 | `tda_v2.py`, `flows_v2.py` | Eqs. 32-33 are descriptive; no detection guarantee (4.20.1). |
| Timing/size/pattern evasion at network layer (4.3) | spike encoding + SNN | Table 13 | `snn_v2.py`, `c/lif_kernel_v2.c` | Hardware numbers need Loihi. |
| TLS-handshake side channel (4.3, 4.26.3) | adaptive noise injection | 98 % SNR reduction | `sidechannel_v2.py` | Statement about measured traces only (4.13.1). |
| Poisoned / Byzantine clients (4.6.3) | geometric median + detector cross-check | Table 16 | `byzantine_v2.py` | Tolerance bound not established (4.6.3). |
| Model-update inference by the coordinator (4.5) | clipping + Gaussian noise, Ring-LWE aggregation | eps = 1.2 | `dp_aggregation_v2.py`, `dp_accountant_v2.py`, `ring_lwe_v2.py` | Bounds one leakage channel (4.6.2). |
| Rogue / compromised resolver (4.1.2, 4.19) | ZK attestation, trust fabric | Table 15 | `zk_r1cs_v2.py`, `merkle_fri_v2.py` | Conditional on assumptions of Sec 4.23. |
| GAN-generated evasive queries (4.2.2, 4.26.2) | GAN-hardened training | 96.8 % robustness | generator objective Eq. 35 (training code in original repo) | Single privacy budget evaluated. |
