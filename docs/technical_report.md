# MLX-Native Causal Discovery: From Single-Machine Drift Detection
# to Federated Learning with Differential Privacy

**Technical Report**

---

## Abstract

We present two complementary systems for causal discovery on Apple Silicon,
both implemented natively in MLX. The first detects structural drift in
time-series data using an incremental DirectLiNGAM engine with a multi-signal
CUSUM detector. The second extends the same causal engine to the federated
setting, exploiting the additivity of higher-order cumulant tensors to enable
single-round aggregation with differential privacy. Across both projects we
prioritize depth over breadth: each system is validated with multiple
ablation studies, 10 seeds per condition, and a fully reproducible
experimental pipeline. Four findings stand out: (1) multi-signal drift
detection dominated by coefficient change outperforms residual and
structure-only signals; (2) sample-weighted cumulant aggregation is
mathematically lossless, matching centralized computation to three decimal
places even under non-IID clients; (3) differential privacy creates a sharp
two-phase privacy–accuracy curve with a quantifiable threshold
ε_min ≈ 0.01, below which causal direction recovery degrades below chance;
and (4) DP and structure heterogeneity have different noise geometries,
requiring different threshold-recalibration strategies.

---

## 1. Motivation

Causal discovery — recovering directed relationships from observational data —
has matured rapidly in theory but remains difficult to deploy in production.
Three practical challenges dominate:

- **Non-stationarity.** Industrial processes drift over time; retraining a
  causal model from scratch on every window is computationally prohibitive on
  edge devices.
- **Data fragmentation.** Multi-site healthcare and industrial settings cannot
  centralize data due to regulations (GDPR, HIPAA); standard federated
  averaging is incompatible with LiNGAM's pairwise independence tests.
- **Privacy.** Even federated models can leak information through shared
  parameters; differential privacy is required for formal guarantees.

This report addresses the first two challenges directly, and quantifies the
cost of the third.

---

## 2. Project 1: Single-Machine Incremental Causal Discovery

### 2.1 Problem

Detect changes in the causal structure and coefficients of a time series, and
update the causal graph incrementally without full retraining. Target: edge
deployment on Apple Silicon.

### 2.2 Method

**Engine.** A DirectLiNGAM variant in MLX with delay embedding for lagged and
instantaneous causality (max_lag = 2). Edge selection via backward elimination
with BIC; coefficient estimation via target-prior Ridge regression.

**Drift score.** A fused signal combining:

- Residual energy distance between consecutive windows
- Relative coefficient change on the union edge set
- Normalized structural Hamming distance between consecutive graphs

Each signal is standardized with robust (median/MAD) statistics from a warmup
period, then monitored by a double-sided CUSUM with adaptive threshold and
permutation-test confirmation.

**Incremental update.** Upon confirmed drift, seeds are expanded to their
directed Markov blanket; DirectLiNGAM reruns locally with warm start;
consistency checks (acyclicity, SHD jump, residual variance, coefficient
explosion) gate a full-retrain fallback.

### 2.3 Key Results

5 drift scenarios × 10 seeds × 4 baselines.

| Method | SHD pre | SHD post | F1 post | Det F1 | Update/retrain |
|--------|---------|----------|---------|--------|----------------|
| Full retrain every window | 9.57 | 12.13 | 0.742 | — | 1.0 |
| Detect + full retrain | 9.57 | 12.13 | 0.742 | 0.632 | 1.0 |
| Sliding window, no update | 9.48 | 12.46 | 0.729 | — | — |
| **Incremental (ours)** | **9.58** | **12.06** | **0.743** | **0.634** | **71.7×** |

**Findings.**

1. **Coefficient change dominates drift detection.** The ΔB-only signal
   achieves the highest detection F1 (0.76); adding residual energy distance
   *reduces* F1 (0.63) because ED's heavy right tail inflates the CUSUM
   threshold. SHD post-drift is invariant (12.1–12.3) across all six signal
   modes.
2. **Update scope has minimal impact at 8 variables.** Seeds, seeds+parents,
   and seeds+Markov-blanket are statistically indistinguishable in SHD
   (within 0.01). The consistency-checked fallback absorbs the difference.
3. **The incremental path matches full retraining** on graph quality (SHD
   gap ≤ 0.20) while running 60–71× faster on drift windows. Fallback rate
   is 1.2%.

---

## 3. Project 2: Federated Causal Discovery with Differential Privacy

### 3.1 Problem

K clients hold horizontally partitioned data that cannot be shared. Each
client computes local statistics; a server aggregates and recovers the global
causal graph. Differential privacy is applied to the shared statistics.

### 3.2 Method

**Cumulant additivity.** The central insight is that higher-order cumulant
tensors are additive across independent sample groups:

```
C_global = Σ_k (n_k / N) · C_k
```

This holds *regardless* of whether clients are i.i.d. — as long as each
client's samples are internally independent.

**Protocol.**

1. Each client computes local C2 (covariance) and C3 (third-order cumulant
   tensor).
2. Each client adds Gaussian noise (DP mechanism) with sensitivity 1/n_k.
3. The server performs sample-weighted averaging.
4. The server runs HC-LiNGAM: causal order from third-order
   cross-multiplication (`C3[i,j,j]·C2[j,j] = C2[i,j]·C3[j,j,j]`), adjacency
   via Ridge regression.

Single-round communication suffices because cumulants are sufficient
statistics for the linear non-Gaussian SEM.

### 3.3 Validation

**Synthetic data (i.i.d., ε=∞).** Federated matches centralized to
ΔAA = 0.000, ΔSHD = 0 across 5 seeds. This is not an experimental
coincidence: it is an algebraic identity.

### 3.4 Main Result: Privacy–Accuracy Trade-off

10 seeds; ε ∈ {0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0, ∞}.

| ε | SHD | F1 | Ancestral Acc. | Direction Acc. |
|---|-----|-----|---------------|----------------|
| 0.001 | 10.9 | 0.20 | 0.53 | 0.50 |
| 0.01  | 9.2  | 0.40 | 0.94 | 0.92 |
| 0.1   | 1.2  | 0.90 | 0.99 | 0.98 |
| 0.5   | 0.2  | 0.95 | 0.99 | 0.99 |
| ∞     | 0.3  | 0.95 | 0.99 | 0.99 |

**Two-phase structure.** Below ε ≈ 0.01, DP noise overwhelms the non-Gaussian
signal required by LiNGAM, and causal direction recovery degrades to chance
(AA ≈ 0.5). Above ε = 0.1, the federated model matches the non-private
baseline within SHD ≤ 1. **ε_min ≈ 0.01** is the privacy threshold below which
causal identifiability fails.

### 3.5 Baseline Comparison

| Method | SHD | F1 | AA |
|--------|-----|-----|-----|
| Centralized DirectLiNGAM | 0.00 | 1.000 | 1.000 |
| Local HC-LiNGAM + voting | 0.40 | 0.952 | — |
| Federated HC-LiNGAM (ε=∞) | 0.00 | 1.000 | 1.000 |
| Federated + DP (ε=0.1, τ=0.1) | 4.90 | 0.623 | 0.974 |
| Federated + DP (ε=0.1, τ=0.5) | **1.30** | **0.867** | 0.974 |

**Finding.** Federated matches centralized exactly. DP's damage is
structural, not directional: B4's high SHD coexists with near-perfect AA.
Recalibrating the edge threshold from τ=0.1 to τ=0.5 reduces SHD by 73%.

### 3.6 Six Ablations

#### 3.6.1 Standardization

| ε | Raw AA | Standardized AA |
|---|--------|-----------------|
| ∞ | 1.000 | 0.919 |
| 0.1 | 0.974 | 0.784 |
| 0.01 | 0.917 | **0.350** |

Standardization amplifies DP sensitivity by ~7×. At ε = 0.01, standardized AA
falls *below chance* (0.350 < 0.5) — a qualitatively different failure mode
from randomization: the model produces systematically wrong-direction answers.

#### 3.6.2 Cumulant Order

| Mode | AA |
|------|-----|
| C2 only | 0.418 |
| C2 + C3 | 1.000 |

Third-order cumulants are necessary for direction identification. C2-only
falls below chance as expected (covariance is symmetric).

#### 3.6.3 Client Count

| ε | K=1 | K=8 |
|---|-----|-----|
| ∞ SHD | 0.00 | 0.00 |
| 0.1 SHD | 4.20 | 6.60 |
| 0.1 AA | 0.967 | 0.942 |

Fragmentation penalizes structure (SHD +57%) more than direction (AA −2.5%).

#### 3.6.4 Coefficient Heterogeneity

Clients share the same edge set but with heterogeneous edge coefficients.
Perturbation: `factor ~ N(1, 0.5h)`.

| Method | h=0.0 | h=1.0 |
|--------|-------|-------|
| Pooled / Fed ε=∞ (SHD) | 0.30 | 1.60 |
| Pooled / Fed ε=∞ (AA) | 0.983 | 0.930 |
| Fed ε=0.1 (SHD) | 5.50 | 5.30 |

Federated equals pooled across all h to three decimal places (algebraic
identity, not experimental coincidence). Heterogeneity damages structure
(SHD 5.3×) but not direction (AA −5%). DP and heterogeneity effects are
approximately additive on SHD.

#### 3.6.5 Structure Heterogeneity

Clients share a base DAG but have client-specific edges, with base edges
dropped and new edges added at probability `perturb_prob` per client.

| perturb_prob | R_shared | R_specific | F1_union |
|--------------|----------|------------|----------|
| 0.0 | 1.000 | — | 0.975 |
| 0.2 | 0.978 | 0.884 | 0.772 |
| 0.5 | 1.000 | 0.957 | 0.818 |
| 0.8 | 1.000 | 0.826 | 0.750 |

Federated aggregation does *not* preferentially lose client-specific edges
(recall ≥ 0.83). The failure mode is **over-densification**: extra edges
beyond the union drop precision, not recall.

**Threshold recalibration for structure heterogeneity.**

| p | τ=0.1 | τ=0.2 | τ=0.3 | τ=0.5 | Best τ |
|---|-------|-------|-------|-------|--------|
| 0.2 | 0.772 | 0.825 | 0.823 | 0.680 | 0.2 |
| 0.5 | 0.818 | 0.852 | 0.836 | 0.649 | 0.2 |
| 0.8 | 0.750 | 0.744 | 0.747 | 0.612 | 0.1–0.3 |

Threshold recalibration recovers only 4–7% F1 at moderate perturbation — 
*weaker* than the DP-aware recalibration (39% F1 gain). This indicates
different noise geometries: DP adds threshold-separable uniform noise;
structure heterogeneity produces half-strength signals that overlap the
true edge distribution. At p = 0.8, no threshold recovers the graph.

---

## 4. Synthesis

**Common thread: cumulants as the right abstraction.** Both projects use
third-order cumulants as the primary signal. In Project 1, cross-cumulants
between residuals and predictors drive direction identification. In Project 2,
joint cumulants enable lossless federated aggregation. This is not
coincidental: cumulants are the minimal statistic that distinguishes
direction in linear non-Gaussian systems, and their tensor structure is
compatible with distributed computation.

**Where the two projects diverge.** Project 1 is time-axis oriented — its
scalability axis is *variables per window* and *drift frequency*. Project 2
is space-axis oriented — its scalability axis is *clients* and *privacy
budget*. Both use the same underlying causal engine; only the deployment
context differs.

**Noise geometry as a unifying lens.** The federation results reveal that
different failure sources leave different fingerprints on cumulant tensors.
DP noise is uniformly additive and can be almost completely thresholded away.
Structure heterogeneity produces systematic half-strength signals that
partially overlap the true edge distribution, so thresholding only helps
marginally. Coefficient heterogeneity lies in between: it biases strength
estimates but preserves direction, and is not recoverable by thresholding.

**Practical recommendations.**

1. Use coefficient-change signal (ΔB), not residual energy distance, for
   drift detection under linear non-Gaussian assumptions.
2. Do not standardize before cumulant-based federated causal discovery. The
   cost is 8 percentage points of AA without DP, 57 percentage points under
   strong DP.
3. Recalibrate the edge threshold when DP is active: τ = 0.5 vs default 0.1.
4. Recalibrate to a lesser degree under structure heterogeneity: τ = 0.2.
5. Plan for ε ≥ 0.1 for i.i.d. federation, ε ≥ 0.5 for heterogeneous
   federation with coefficient drift.
6. Recognize that single-round cumulant aggregation cannot recover
   client-specific structure at high divergence — multi-round communication
   or explicit disentanglement is needed.

---

## 5. Limitations

Both projects assume:

- Linear mechanisms
- Non-Gaussian noise (asymmetric distributions specifically; uniform and
  Laplace carry no third-order signal)
- No latent confounders
- No feedback loops

Project 1 is further limited to horizontal partitioning on a single machine.
Project 2 is limited to horizontal partitioning; vertical federation requires
secure cross-client cumulants and is deferred to future work.

Heterogeneity coverage is limited to coefficient perturbations (same edge
set, different strengths) and structure perturbations (different edge sets).
Distribution shift in noise, or feedback loops across clients, would require
a different aggregation scheme.

The Sachs dataset is included as a qualitative case study only. It violates
multiple LiNGAM assumptions and yields AA ≈ 0.52, consistent with published
benchmarks.

---

## 6. Reproducibility

All experiments are deterministic given a seed and can be reproduced from
scripts in under 20 minutes on M3 Pro 18GB. See individual project READMEs
for command lines.

Full artifact list:

| Artifact | Project | Results |
|----------|---------|---------|
| Main experiment | B | 5 scenarios × 10 seeds × 4 baselines |
| Signal ablation | B | A1–A6 |
| Update scope ablation | B | U1–U4 |
| CUSUM sweep | B | κ, warmup, permutations |
| ε scan | C | 10 ε × 10 seeds |
| Baseline comparison | C | B1–B4 |
| Standardization ablation | C | A1–A2 |
| Cumulant order ablation | C | O1–O2 |
| Client count ablation | C | K=1/2/4/8 |
| Coefficient heterogeneity ablation | C | h=0/0.2/0.5/1.0 |
| Structure heterogeneity ablation | C | p=0/0.2/0.5/0.8 |
| DP-aware threshold recalibration | C | τ sweep at ε=0.1 |
| Structure-aware threshold recalibration | C | τ sweep at p=0.2/0.5/0.8 |

---

## 7. References

1. Shimizu et al., "A Linear Non-Gaussian Acyclic Model for Causal
   Discovery", JMLR 2006.
2. Page, "Continuous Inspection Schemes", Biometrika 1954.
3. Dwork & Roth, "The Algorithmic Foundations of Differential Privacy", 2014.
4. Sachs et al., "Causal Protein-Signaling Networks Derived from
   Multiparameter Single-Cell Data", Science 2005.
5. FedRCD: "Federated Causal Discovery" — cumulant-based federated LiNGAM
   framework; source of the standardization warning.
6. MLX: https://github.com/ml-explore/mlx
7. lingam package: https://github.com/cdt15/lingam
```
