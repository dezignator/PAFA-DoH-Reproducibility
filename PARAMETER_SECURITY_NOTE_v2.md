# Ring-LWE parameter check (Sec 4.21, Table 17a)

`pafa_v2/lattice_estimate_v2.py` implements the primal uSVP core-SVP estimate
(Alkim et al., 2016). Calibration: ML-KEM-512-like parameters (n=512, q=3329,
CBD eta=3) give beta=406, 2^118.6 classical, matching the published figure.

| LWE dimension n | q | sigma | BKZ beta | classical bits | quantum bits |
|---|---|---|---|---|---|
| 256 | 4096 | 3.20 | 225 | 65.7 | 59.6 |
| 512 | 4096 | 3.20 | 529 | 154.5 | 140.2 |
| 1024 | 4096 | 3.20 | 1171 | 341.9 | 310.3 |
| 512 | 3329 | 1.22 | 406 | 118.6 | 107.6 |

Reading: the pair stated in Table 17a (n=256, q=4096) is a single-ring-element
LWE instance of dimension 256 and lands far below 128 bits under this
estimate. The standardised schemes reach 128-bit class by using module rank
k >= 2 (effective dimension >= 512). Before quoting "128-bit post-quantum
security" for a Ring-LWE layer, either use parameters from a standardised
scheme (ML-KEM-512/768) or run the lattice-estimator and quote its output
together with the exact (n, q, error distribution, secret distribution).

Limits of this estimate: ring structure ignored, no dual or hybrid attacks,
core-SVP cost model (0.292 beta classical, 0.265 beta quantum).
