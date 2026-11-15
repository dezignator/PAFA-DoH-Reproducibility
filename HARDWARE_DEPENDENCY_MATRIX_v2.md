# What can be regenerated without the original hardware

| Result (Chapter 4) | Needs | Regenerable from this repository | Notes |
|---|---|---|---|
| Latency 0.8 ms vs 2.1 ms, Table 13 | Intel Loihi, NVIDIA A100, NxSDK | no | C/NumPy kernels give a CPU baseline only (`c/lif_kernel_v2.c --bench`). |
| Energy 0.12 mJ vs 1.8 mJ | power instrumentation on both | no | E = P_avg * t_inf (Eq. 67); P_avg must be logged per run. |
| Spike encoding efficiency 0.45 bits/spike | encoded traces | partial | `snn_v2.log_burst_encode` reproduces the encoding; the bits/spike estimator is not in the thesis and must be specified. |
| SNR reduction 98 % / 60 %, Table 14 side-channel row | ChipWhisperer, target board, trace sets | no | `sidechannel_v2.py` scores any trace set with the Eq. 47 definition. |
| Betti-1, entropy, Wasserstein, Table 17 | DoH flow data (Zenodo DOI in Sec 4.25.1), ns-3, dnscat2 | partial | Feature code runs; the numbers depend on the point-cloud definition and window size recorded in the manifest. |
| FL accuracy, convergence, dropout, robustness, Table 16 | FL training code + data | partial | Aggregation, clipping, accountant provided; model and training loop are in the original repository. |
| Proof gen/verify time, size, Table 15 | Groth16 prover (e.g. snarkjs/arkworks) on the circuit | partial | `zk_r1cs_v2.py` is a toy of Eqs. 50-51, not the prover. Times are machine dependent: log CPU model per run. |
| Soundness error 2^-132 | none (analytic) | no | Soundness is a property of the construction and its assumptions, not a measured quantity; quote the assumption (Sec 4.23) and the curve. |
| Baseline numbers, Sec 4.30 | baseline code | partial | Provenance label (reproduced / quoted / retrained) per number is in `tables/reported_values_v2.yaml`. |
