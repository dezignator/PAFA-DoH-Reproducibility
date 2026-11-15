# Attack scenarios and detection (synthetic traffic)

seed=20261116; every detector calibrated to 1% FPR on a separate benign calibration set (900 windows); evaluation on 1500 benign and 250 windows per attack. Wilson 95 % intervals in brackets.

### A1 Scenario catalogue

| scenario   | behaviour (simulated)                                                                | thesis section   |
|:-----------|:-------------------------------------------------------------------------------------|:-----------------|
| benign     | browsing bursts (73 %), periodic telemetry polling (15 %), CDN hash labels (12 %)    | -                |
| tunnel     | dnscat2-like: 28-52 char base32 labels, TXT/CNAME/MX heavy, ~0.15-1 s spacing        | 4.26.1           |
| beacon     | C2 heartbeat: 2-10 s period, 4-15 % jitter, short hex labels, constant small payload | 4.26.4           |
| slow_exfil | low-and-slow: 12-45 s mean spacing, 18-30 char labels, TXT heavy                     | 4.26.1           |
| mimic      | benign-looking names/sizes/types, hidden cyclic rate modulation (zero-day style)     | 4.2.1, 4.26.2    |
| enum       | subdomain enumeration host0..hostN, sequential, fast (lifecycle phase 4 only)        | 4.30.4           |

### A2 Detector comparison at a common 1 % false-positive budget

| detector                                 | FPR (benign test)    | TPR tunnel        | TPR beacon        | TPR slow_exfil    | TPR mimic       | mean TPR   |   mean AUC |
|:-----------------------------------------|:---------------------|:------------------|:------------------|:------------------|:----------------|:-----------|-----------:|
| Rule-based (static thresholds)           | 0.867% [0.51%,1.48%] | 100.0% [98%,100%] | 0.0% [0%,2%]      | 100.0% [98%,100%] | 0.0% [0%,2%]    | 50.0%      |      0.862 |
| Statistical (Mahalanobis, benign only)   | 0.933% [0.56%,1.56%] | 100.0% [98%,100%] | 100.0% [98%,100%] | 100.0% [98%,100%] | 4.8% [3%,8%]    | 76.2%      |      0.911 |
| Supervised ML (trained on tunnel+beacon) | 0.667% [0.36%,1.22%] | 100.0% [98%,100%] | 100.0% [98%,100%] | 100.0% [98%,100%] | 0.0% [0%,2%]    | 75.0%      |      0.868 |
| TDA (persistence, delay-embedded IAT)    | 1.000% [0.61%,1.64%] | 4.0% [2%,7%]      | 1.6% [1%,4%]      | 6.4% [4%,10%]     | 26.8% [22%,33%] | 9.7%       |      0.616 |
| SNN (LIF + STDP, benign-trained)         | 1.000% [0.61%,1.64%] | 99.6% [98%,100%]  | 1.2% [0%,3%]      | 80.0% [75%,84%]   | 0.4% [0%,2%]    | 45.3%      |      0.757 |
| PAFA fusion (stat + TDA + SNN, max)      | 1.067% [0.66%,1.73%] | 100.0% [98%,100%] | 100.0% [98%,100%] | 100.0% [98%,100%] | 18.4% [14%,24%] | 79.6%      |      0.945 |

### A3 Ablation of the PAFA fusion (one component removed)

| detector                   | FPR (benign test)    | TPR tunnel        | TPR beacon        | TPR slow_exfil    | TPR mimic       | mean TPR   |   mean AUC |
|:---------------------------|:---------------------|:------------------|:------------------|:------------------|:----------------|:-----------|-----------:|
| fusion without Statistical | 1.333% [0.86%,2.05%] | 99.6% [98%,100%]  | 2.8% [1%,6%]      | 80.0% [75%,84%]   | 25.2% [20%,31%] | 51.9%      |      0.843 |
| fusion without TDA         | 1.267% [0.81%,1.97%] | 100.0% [98%,100%] | 100.0% [98%,100%] | 100.0% [98%,100%] | 4.0% [2%,7%]    | 76.0%      |      0.891 |
| fusion without SNN         | 1.333% [0.86%,2.05%] | 100.0% [98%,100%] | 100.0% [98%,100%] | 100.0% [98%,100%] | 21.6% [17%,27%] | 80.4%      |      0.957 |

### A4 Topological features, benign vs attack (permutation test, Bonferroni alpha/3)

| scenario   | feature             | benign mean ± sd   | attack mean ± sd   |   cohen_d |   p (perm) | reject @ alpha/3   |
|:-----------|:--------------------|:-------------------|:-------------------|----------:|-----------:|:-------------------|
| tunnel     | betti1              | 0.24 ± 0.50        | 0.23 ± 0.49        |     -0.03 |     0.6798 | no                 |
| tunnel     | persistence_entropy | 1.92 ± 0.27        | 1.81 ± 0.30        |     -0.39 |     0.0003 | yes                |
| tunnel     | wasserstein_to_ref  | 0.28 ± 0.06        | 0.30 ± 0.07        |      0.27 |     0.0007 | yes                |
| beacon     | betti1              | 0.24 ± 0.50        | 0.24 ± 0.51        |      0    |     1      | no                 |
| beacon     | persistence_entropy | 1.92 ± 0.27        | 1.93 ± 0.29        |      0.06 |     0.3675 | no                 |
| beacon     | wasserstein_to_ref  | 0.28 ± 0.06        | 0.28 ± 0.07        |      0.05 |     0.4652 | no                 |
| slow_exfil | betti1              | 0.24 ± 0.50        | 0.22 ± 0.46        |     -0.04 |     0.5785 | no                 |
| slow_exfil | persistence_entropy | 1.92 ± 0.27        | 1.81 ± 0.35        |     -0.39 |     0.0003 | yes                |
| slow_exfil | wasserstein_to_ref  | 0.28 ± 0.06        | 0.30 ± 0.08        |      0.25 |     0.0007 | yes                |
| mimic      | betti1              | 0.24 ± 0.50        | 0.44 ± 0.62        |      0.38 |     0.0003 | yes                |
| mimic      | persistence_entropy | 1.92 ± 0.27        | 1.51 ± 0.43        |     -1.37 |     0.0003 | yes                |
| mimic      | wasserstein_to_ref  | 0.28 ± 0.06        | 0.40 ± 0.09        |      1.82 |     0.0003 | yes                |

### A5 Multi-phase lifecycle: per-phase detection rate of attack windows

| phase                                    | Rule-based (static thresholds)   | Statistical (Mahalanobis, benign only)   | Supervised ML (trained on tunnel+beacon)   | TDA (persistence, delay-embedded IAT)   | SNN (LIF + STDP, benign-trained)   | PAFA fusion (stat + TDA + SNN, max)   |
|:-----------------------------------------|:---------------------------------|:-----------------------------------------|:-------------------------------------------|:----------------------------------------|:-----------------------------------|:--------------------------------------|
| stealth set-up (h1-4, mimic)             | 0.0%                             | 1.6%                                     | 0.0%                                       | 24.0%                                   | 0.0%                               | 10.6%                                 |
| C2 beaconing (h5-7, beacon)              | 0.0%                             | 100.0%                                   | 100.0%                                     | 0.0%                                    | 0.0%                               | 100.0%                                |
| query-flood exfiltration (h8-11, tunnel) | 100.0%                           | 100.0%                                   | 100.0%                                     | 5.0%                                    | 100.0%                             | 100.0%                                |
| subdomain enumeration (h12-14, enum)     | 0.0%                             | 100.0%                                   | 0.0%                                       | 6.1%                                    | 1.8%                               | 100.0%                                |
| benign false-alarm rate (all hours)      | 1.7%                             | 0.0%                                     | 0.9%                                       | 1.7%                                    | 2.4%                               | 0.8%                                  |

### A6 Simulated CPA key recovery: success rate vs number of traces (injected amplitude is a multiple of the measurement-noise sd; the multiples were chosen to span the 60-98 % Eq. 47 reduction range, so the CPA columns are the informative result)

| countermeasure                                 | total noise sd              | ||s||/||n|| (Eq. 47)   | ratio reduction   | n=25         | n=50         | n=100        | n=200        | n=400        | n=800        | n=1600       | n=3200       |
|:-----------------------------------------------|:----------------------------|:-----------------------|:------------------|:-------------|:-------------|:-------------|:-------------|:-------------|:-------------|:-------------|:-------------|
| none (measurement noise only)                  | 1.00                        | 1.414                  | 0.0%              | 0.98         | 1.00         | 1.00         | 1.00         | 1.00         | 1.00         | 1.00         | 1.00         |
| static injection, amplitude x2.5               | 2.69                        | 0.525                  | 62.9%             | 0.25         | 0.53         | 0.95         | 1.00         | 1.00         | 1.00         | 1.00         | 1.00         |
| static injection, amplitude x8                 | 8.06                        | 0.175                  | 87.6%             | 0.00         | 0.05         | 0.10         | 0.33         | 0.57         | 0.93         | 1.00         | 1.00         |
| static injection, amplitude x25                | 25.02                       | 0.057                  | 96.0%             | 0.00         | 0.00         | 0.02         | 0.00         | 0.03         | 0.12         | 0.17         | 0.68         |
| static injection, amplitude x50                | 50.01                       | 0.028                  | 98.0%             | 0.00         | 0.00         | 0.00         | 0.00         | 0.00         | 0.02         | 0.02         | 0.12         |
| SNN-adaptive: x50 while T(t)~1, x2.5 otherwise | 50.01 (alert) / 2.69 (idle) | 0.028 (alert)          | 98.0% (alert)     | 0.00 (alert) | 0.00 (alert) | 0.02 (alert) | 0.00 (alert) | 0.00 (alert) | 0.00 (alert) | 0.00 (alert) | 0.10 (alert) |

### A6b Injected noise power: static vs threat-adaptive

| attack duty cycle   |   static x50 injected power |   adaptive injected power | saving   | V_th idle/alert (Eq. 48)   |
|:--------------------|----------------------------:|--------------------------:|:---------|:---------------------------|
| 1%                  |                        2500 |                        31 | 98.8%    | 1.00/1.80                  |
| 5%                  |                        2500 |                       131 | 94.8%    | 1.00/1.80                  |
| 20%                 |                        2500 |                       505 | 79.8%    | 1.00/1.80                  |

### A7 Measured CPU cost per window

| stage                                   |   us per window (CPU, 1 core) |
|:----------------------------------------|------------------------------:|
| Window feature extraction (17 feats)    |                         364.5 |
| Rule-based                              |                           0.1 |
| Mahalanobis                             |                           0.7 |
| Supervised logistic                     |                           0.1 |
| TDA: embed + H0/H1 + Wasserstein        |                        4482   |
| SNN: encode + 520-step LIF sim (python) |                        6746.4 |


Wall time 61s.
