<!--
README_hardware.md — PAFA-DoH Reproducibility Artifact
Documentation for the hardware-simulation subpackage.
═══════════════════════════════════════════════════════════════════════════

Companion to:
  README.md (File 1), REPRODUCIBILITY_CHECKLIST.md (File 2),
  LICENSE (File 3), configs/hardware_emulation_config.yaml (File 9h),
  src/hardware_simulation/loihi_mock.py (File 16b),
  src/hardware_simulation/a100_emulation.py (File 16c).

Thesis chapter: Chapter 4 (PAFA-DoH)
Citation DOI:   10.1016/j.neunet.2025.108343
License:        MIT
Artifact version: 1.0.0
═══════════════════════════════════════════════════════════════════════════
-->

# Hardware Simulation

This subpackage contains the calibrated software simulators for the
two proprietary hardware platforms referenced in Chapter 4: Intel
Loihi 2 and NVIDIA A100. It also contains the explicit non-use
statement for IBM TrueNorth.

**None of these three platforms is redistributed by this artifact.**
See `LICENSE` §3 for the full statement.

---

## 1. Why simulate?

Chapter 4's Table 13 reports six measured values on two proprietary
platforms:

| Platform | Values reported in Table 13 |
|---|---|
| Intel Loihi 2 | 0.8 ms latency, 0.12 mJ energy, 0.45 bits/spike |
| NVIDIA A100 | 2.1 ms latency, 1.8 mJ energy |

Shipping the hardware is impossible (Loihi 2 is Intel proprietary;
A100 is NVIDIA proprietary). Shipping a *bit-exact* software model of
the hardware is also impossible (both are proprietary architectures).
What this artifact does ship is a **calibrated simulator**: a
pure-NumPy implementation that reproduces the *reported numeric
behavior* of the hardware, within the tolerances below.

The simulator's purpose is narrow and explicit:

1. Confirm that the artifact's Table 13 reproduction logic is
   self-consistent.
2. Give a reviewer who does not have access to Loihi 2 or A100 a way
   to run the full pipeline end to end.
3. Make the distinction between "measured on real hardware" and
   "produced by a simulator" impossible to confuse, by tagging every
   simulator output with `"simulated": true` and a
   `simulator_version` string.

It is **not** a claim that the simulator is a substitute for the real
hardware. Any downstream use that quotes Table 13 must preserve the
`"simulated": true` tag; quoting a simulated number as a hardware
measurement is an integrity violation and is flagged in the
anti-confusion checklist at the bottom of `README.md`.

---

## 2. Tolerances

The simulator's outputs are compared against the Table 13 targets with
the following tolerances. The tolerances are the same as those in
`REPRODUCIBILITY_CHECKLIST.md` §5; they are duplicated here so a
reviewer who opens only this file sees them without a cross-reference.

| Platform | Metric | Target | Tolerance |
|---|---|---|---|
| Loihi 2 | Inference latency | 0.8 ms | ±20 % |
| Loihi 2 | Energy per inference | 0.12 mJ | ±25 % |
| Loihi 2 | Spike encoding efficiency | 0.45 bits/spike | ±10 % |
| A100 | Inference latency | 2.1 ms | ±20 % |
| A100 | Energy per inference | 1.8 mJ | ±25 % |

The simulator draws its observed value from a log-normal distribution
centered on the target with the stated relative jitter. It is
deterministic given a seed (default 42, matching
`configs/hardware_emulation_config.yaml`).

---

## 3. IBM TrueNorth — explicit non-use

**IBM TrueNorth was not used in any experiment reported in Chapter 4.**

§4.24.1 of the thesis mentions TrueNorth only as a *comparison
platform* (4,096 cores × 256 neurons ≈ 1 M neurons), for context
when describing the Loihi 2 configuration. No result in Table 13,
Table 14, Table 15, Table 16, Table 17, or any other table or figure
in Chapter 4 depends on TrueNorth.

This artifact therefore:

- does **not** simulate TrueNorth,
- does **not** require TrueNorth for any check,
- does **not** claim any measurement on TrueNorth.

The `configs/hardware_emulation_config.yaml` file (File 9h) contains
a `truenorth:` block with `used: false` and an explanatory note, so
that a reviewer who greps for "truenorth" sees the explicit flag
rather than a silent absence.

---

## 4. How the simulator tags its output

Every dict returned by `LoihiMock.simulate_inference()` or
`A100Emulation.simulate_inference()` contains the following keys,
in addition to the observed values:

| Key | Value (Loihi) | Value (A100) |
|---|---|---|
| `simulated` | `True` | `True` |
| `hardware` | `"Loihi 2 (simulated)"` | `"A100 (simulated)"` |
| `simulator_version` | `"pafa-hardware-sim/1.0.0"` | `"pafa-hardware-sim/1.0.0"` |
| `target_latency_ms` | `0.8` | `2.1` |
| `target_energy_mj` | `0.12` | `1.8` |
| `tolerance_latency_pct` | `20.0` | `20.0` |
| `tolerance_energy_pct` | `25.0` | `25.0` |

`verification_script.py` (File 20i) refuses to compare a simulated
output against a hardware output, and vice versa. If a reviewer runs
the artifact on real hardware with `mode.hardware: "real"` in the
master config (File 9), they must flip the
`honesty_tags.simulated` field in
`configs/hardware_emulation_config.yaml` to `false` and update
`simulator_version` to `"hardware-direct"`, or the comparison will
fail with a clear message.

---

## 5. What the simulator does NOT claim

- It does not claim to be bit-exact with the hardware. It is
  calibrated to within the tolerances in §2.
- It does not claim to emulate the hardware architecture.
- It does not claim that running on the simulator is equivalent to
  running on real hardware for any purpose other than the
  self-consistency check in §1.
- It does not claim that the Table 13 numbers were "measured on the
  simulator"; they were measured on the real hardware, and the
  simulator reproduces them within tolerance.

The anti-confusion checklist in `README.md` §15 restates this: any
downstream use of Table 13 must preserve the `"simulated": true` tag
or, if quoting the thesis's reported values, state clearly that
those values were measured on real hardware (as §4.24.1 of the thesis
does).

---

*End of `src/hardware_simulation/README_hardware.md`.*
