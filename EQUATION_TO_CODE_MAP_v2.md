# Equation / table to code map (Chapter 4)

| Thesis item | Where | Implementation | Test |
|---|---|---|---|
| Eq. 36 clipped, noised aggregation | 4.6.2 | `pafa_v2/dp_aggregation_v2.py::dp_aggregate` | `tests/test_dp_byz_v2.py::TestAggregation` |
| Eq. 37 / 59 (eps, delta)-DP definition | 4.6.2, 4.22 | statement only; accountant in next row | - |
| Eq. 38 weighted geometric median | 4.6.3 | `pafa_v2/byzantine_v2.py::geometric_median` | `TestByzantine` |
| detector cross-check | 4.6.3 | `byzantine_v2.py::cross_check` | `TestByzantine::test_cross_check` |
| Eq. 39 convergence bound | 4.6.4 | quoted from literature, not implemented | - |
| Eq. 40 persistence diagram | 4.7.1 | `pafa_v2/tda_v2.py::vr_persistence` | `tests/test_tda_v2.py::TestPH` |
| Betti number at scale | 4.7.1 | `tda_v2.py::betti_at` | `TestPH::test_betti_at` |
| Eq. 41 p-Wasserstein | 4.7.2 | `tda_v2.py::wasserstein` | `TestWasserstein` |
| Eq. 42 permutation p-value, Bonferroni | 4.8.1 | `pafa_v2/stats_v2.py` | `test_flows_stats_metrics_v2.py::TestStats` |
| Table 17 features | 4.28.2 | `tda_v2.py::summarize_window`, `scripts/tda_pipeline_demo_v2.py` | pipeline demo |
| point cloud of Sec 4.7.1 | 4.7.1 | `pafa_v2/flows_v2.py::flow_to_cloud` | `TestFlows` |
| Eq. 43 log burst spike encoding | 4.9.1 | `pafa_v2/snn_v2.py::log_burst_encode` | `tests/test_snn_sidechannel_v2.py::TestEncoding` |
| Eqs. 44-45 LIF + reset | 4.10 | `snn_v2.py::lif_simulate`; C: `c/lif_kernel_v2.c` | `TestLIF`; `make -C c test` |
| Eq. 46 pair STDP | 4.11 | `snn_v2.py::stdp_delta_w`, `stdp_delta_w_nearest` | `TestSTDP` |
| Eq. 47 SNR objective | 4.13.1 | `pafa_v2/sidechannel_v2.py::snr_ratio`, `meets_objective` | `TestSideChannel` |
| Eq. 48 adaptive threshold | 4.13.2 | `sidechannel_v2.py::adaptive_threshold` | `TestSideChannel` |
| Eq. 49 STDP-modulated error | 4.13.3 | `sidechannel_v2.py::stdp_error_trace` | `TestSideChannel::test_error_trace_shape` |
| Eqs. 50-51 R1CS / QAP | 4.15-4.16 | `pafa_v2/zk_r1cs_v2.py` (toy circuit) | `tests/test_zk_fri_v2.py::TestR1CS` |
| Eq. 52 Merkle commitment | 4.17 | `pafa_v2/merkle_fri_v2.py::Merkle` | `TestMerkle` |
| FRI folding | 4.18 | `merkle_fri_v2.py::fold`, `fri_prove`, `fri_verify` | `TestFRI` |
| Eq. 58 Ring-LWE encryption | 4.21 | `pafa_v2/ring_lwe_v2.py::RLWE` | `tests/test_rlwe_v2.py::TestRLWE` |
| 128-bit PQ parameter claim | 4.21, Table 17a | `pafa_v2/lattice_estimate_v2.py` | `TestEstimator` (calibrated on ML-KEM-512) |
| Eq. 60 moments accountant | 4.22 | `pafa_v2/dp_accountant_v2.py::eps_from_rdp` | `TestAccountant` |
| Eqs. 61-62 completeness / soundness | 4.23 | statements only | - |
| Eq. 63 ROC, gamma* | 4.26.1 | `pafa_v2/metrics_v2.py::roc`, `best_threshold` | `TestMetrics` |
| Eqs. 64-65 accuracy, FPR | 4.27.1 | `metrics_v2.py::accuracy`, `fpr` | `TestMetrics` |
| Eqs. 66-67 latency, energy | 4.27.4 | definitions; measured on hardware | - |
| Tables 13-17a | 4.24-4.30 | `tables/reported_values_v2.yaml` -> `scripts/generate_tables_v2.py` | `make tables`, `make tex` |
