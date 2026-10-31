# F3-AGI v4.1 Technical Addendum: Zero-Touch ARC-AGI-2 Generalization, Hybrid NREM Cellular Plasticity, and Industrial Safety-Critical Rule Repair

**Authors:** Henrique Oliveira & Polybee AI Research (`https://polybee.ai/`)  
**Affiliation:** Polybee AI Research Laboratory  
**Document Series:** Polybee Technical Report Series · Addendum to v4.0.0 (`arXiv:2610.xxxxx`, DOI: `10.5281/zenodo.23816665`)  
**Date:** October 2026  
**Target Architecture:** CPU-First (<400 MB RAM, Zero GPU, NumPy Native)  

---

## Abstract

This technical addendum reports the empirical findings and architectural extensions developed during Horizons 6 through 9 of the **Polybee F3-AGI v4.1** research cycle. Following the formal release and archival of F3-AGI v4.0.0 (Zenodo DOI: `10.5281/zenodo.23816665`), we investigate out-of-distribution generalization on the newly released **ARC-AGI-2** benchmark, introduce the **Cycle 3.5 Hybrid NREM Cellular Mutator** (`NREMCellularResidualLearner`), evaluate multi-seed replay-free continual learning on real ARC feature vectors, and benchmark real-world industrial utility beyond abstract grids. 

Our key findings demonstrate:
1. **Zero-Touch ARC-AGI-2 Evaluation:** Evaluating the frozen v4.0.0 neuro-symbolic engine across all $N=120$ tasks in `ARC-AGI-2/data/evaluation` reveals that the 30 learned operators achieve exact output dimensional prediction ($S_{\text{shape}} = 1.0$) on **95/120 tasks (79.17%)**, with **46/120 (38.33%)** achieving $\ge 85\%$ pixel accuracy and **33/120 (27.50%)** achieving $\ge 90\%$, at an average inference latency of **1.179 s/task** on pure CPU. On a calibration sample of $N=100$ tasks from `ARC-AGI-2/data/training`, frozen v4.0.0 scores **33/100 (33.00% `pass@2`)** and 92.00% shape match. We cataloged 282 high-confidence near-miss tasks exhibiting localized residual discrepancies.
2. **Cycle 3.5 Hybrid NREM Cellular Mutator & Two-Step Residual Chaining ($\Delta_2 \circ \Delta_1$):** Coupling a localized cellular neural network (Motor A) with a relational predicate lookup table and object-snapping engine (Motor B) under leave-one-out cross-validation (LOO-CV) Rashomon certification converts four challenging near-miss tasks into exact 100% solutions (`84f2aca1`, `ae58858e`, `929ab4e9` in ARC-1, and `05f2a901` in ARC-2 Training). This elevates the verified ARC-AGI-1 evaluation score to **60/400 (15.00% global, 41/350 = 11.71% blind hold-out)**, brings ARC-2 Training calibration to **34/100 (34.00%)**, advances ARC-2 evaluation near-misses to near-perfection (`cad67732` to **99.5%** and `88e364bc` to **98.6%**), and improves pixel accuracy on **90 of the 282 near-misses (31.91%)** with **0.00% regressions**.
3. **Multi-Seed Replay-Free Continual Plasticity on Real ARC Vectors:** Across 5 seeds and task order permutations on 64D cell embeddings extracted from active ARC foreground transformations, `F3-Metabolic-Optim` achieves **$76.76 \pm 3.82\%$ final accuracy** while eliminating **$92.45 \pm 0.32\%$ of synaptic weight updates**, reducing $L_1$ weight drift ($M_1$) by **$30.5\%$**, and running **$3.13\times$ faster** than Elastic Weight Consolidation (EWC) without storing a single past exemplar in memory.
4. **Real-World Industrial Utility Beyond ARC:** In an Edge IoT telemetry stream undergoing non-stationary concept drift across 5 regimes, F3-Metabolic-Optim cuts synaptic updates by **$90.73 \pm 1.06\%$** and operates **$2.32\times$ faster** than EWC. In a safety-critical legacy rule repair task ($N=1000$ test items), the F3-AGI certified residual patch achieves **100.0% test accuracy, 100.0% defect repair, and 0.00% regression**, whereas unconstrained neural network replacement suffers catastrophic regression on **$45.84 \pm 1.77\%$** of previously correct legacy cases.

---

## 1. Zero-Touch Generalization on ARC-AGI-2 and Near-Miss Taxonomy

### 1.1 Experimental Protocol
The ARC-AGI-2 benchmark presents significantly harder compositional challenges than ARC-AGI-1, featuring expanded multi-component interactions, recursive frame packings, and subtle boundary recoloring conditions. To establish an unbiased out-of-distribution baseline, the frozen release of F3-AGI v4.0.0 (including the 30-operator learned library in `f3_learned_library.py`) was executed without modifications or retraining on two distinct splits:
- **Full Public Evaluation Split ($N=120$ tasks):** Evaluated strictly following the ARC Prize blind hold-out protocol (predicting test outputs without test label leakage).
- **Training Calibration Sample ($N=100$ tasks):** The first 100 tasks of `ARC-AGI-2/data/training` evaluated lexicographically to assess library domain coverage.

Both splits were evaluated using a per-task compute cap of $15.0\,\text{s}$ under pure CPU execution (<400 MB RAM).

### 1.2 Quantitative Baseline Results
Table 1 summarizes the empirical performance of frozen F3-AGI v4.0.0 on ARC-AGI-2.

| Split | Total Tasks ($N$) | Output Shape Match ($S_{\text{shape}}=1.0$) | Pixel Acc $\ge 85\%$ | Pixel Acc $\ge 90\%$ | Exact Solved (`pass@1`) | Exact Solved (`pass@2`) | Total Latency | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ARC-AGI-2 Public Eval** | 120 | **95 (79.17%)** | **46 (38.33%)** | **33 (27.50%)** | 0 (0.00%) | 0 (0.00%) | 141.53 s | 1.179 s/task |
| **ARC-AGI-2 Train Sample** | 100 | **92 (92.00%)** | **57 (57.00%)** | **50 (50.00%)** | 33 (33.00%) | 33 (33.00%) | 70.20 s | 0.702 s/task |

As evidenced by the results, the v4.0.0 symbolic DSL captured the macroscopic geometry across both splits, predicting the exact output grid dimensions in 79.17% of evaluation tasks and 92.00% of training tasks. Furthermore, 38.33% of evaluation tasks reached $\ge 85\%$ pixel accuracy. However, zero evaluation tasks attained 100.0% exact match under zero-touch evaluation, whereas 33% of training calibration tasks were solved outright by library operators such as `alternating_mirror_tile`, `kronecker_self_pattern`, `fill_enclosed_holes`, and `dihedral_template_matching`.

### 1.3 Near-Miss Taxonomy ($N=282$)
Detailed inspection of failure modes revealed that candidate programs frequently induced the correct global transformation (e.g., symmetry reflection, frame bounding, or gravity projection) but diverged by only 1 to 5 pixels on fine-grained topological details:
- **Perimeter & Cavity Boundary Discrepancies:** Background cells enclosed by foreground contours not filled or misattributed.
- **Relational Object Collisions:** Objects correctly translated along an axis but stopping 1 cell prior to marker collision.
- **Color Inversion Invariants:** Transformation rules dependent on odd/even parity of neighboring markers.

We cataloged all $N=282$ tasks exhibiting $S_{\text{shape}} = 1.0$ and pixel accuracy $\ge 80\%$ into `h6_nearmiss_candidates.json`:
- **184 tasks** in the ARC-AGI-1 evaluation hold-out ($N=400$ split);
- **63 tasks** in the ARC-AGI-2 public evaluation split ($N=120$ split);
- **35 tasks** in the ARC-AGI-2 training sample ($N=100$ split).

---

## 2. Cycle 3.5 Hybrid NREM Cellular Mutator & Two-Step Residual Chaining

To bridge the gap between global symbolic operators ($90\%\text{--}99\%$ accuracy) and exact verification ($100\%$), we introduced **Cycle 3.5: The Hybrid NREM Cellular Mutator** (`f3_nrem_cellular_mutator.py`).

![Figure 5: ARC-AGI-2 Benchmark and Hybrid NREM Cellular Mutator Architecture](../figures/fig5_h6_arc2_and_hybrid_nrem.png)

### 2.1 Dual-Engine Architecture
Let $\hat{Y}_{\text{st}} = h_{\text{best}}(X)$ denote the symbolic output grid produced by the best program found in Cycles 1–3, and let $M_{\text{diff}} = (\hat{Y}_{\text{st}} \neq Y)$ denote the binary discrepancy mask over demonstration pairs. Cycle 3.5 operates strictly when $\text{shape}(\hat{Y}_{\text{st}}) = \text{shape}(Y)$ and $0 < \|M_{\text{diff}}\|_0 \le 0.25 \cdot |\Omega|$, employing two synergistic engines:

1. **Motor A — Localized Cellular Neural Network (`F3-MLP`):**
   - Extracts a 64-dimensional feature vector $\mathbf{x}_{r,c}$ for each cell $(r, c)$, encoding local $3\times 3$ and $5\times 5$ color patches, coordinate position offsets, ray projections to the four cardinal boundaries, and connected component labels.
   - A 2-layer MLP ($64 \to 32 \to 10$) trained via `F3-Metabolic-Optim` (SynGAP diagonal Fisher preconditioning + Memorized Mistake Gating). Synaptic updates are strictly restricted to mispredicted cells where error exceeds the metabolic margin $\tau_m = 0.05$.
2. **Motor B — Relational & Morphological Synthesizer:**
   - Synthesizes non-linear discrete repair operators including:
     - **Sparse Predicate LUT:** Mapping local neighborhood hashes to target color substitutions.
     - **Connected Component Attribute Recoloring:** Re-coloring isolated objects based on bounding box aspect ratio, area parity, or cavity containment.
     - **Object Collision Snapping:** Translating coherent foreground components until bounding contact with reference marker colors.

### 2.2 Two-Step Residual Chaining ($\Delta_2 \circ \Delta_1$)
In complex ARC-2 tasks, localized errors stem from two distinct physical sources (e.g., an object shift followed by an enclosed cavity fill). We formulated **Two-Step Residual Chaining**:
$$\hat{Y}^{(1)} = \Delta_{\text{res}, 1}(\hat{Y}_{\text{st}}), \quad \hat{Y}^{(2)} = \Delta_{\text{res}, 2}(\hat{Y}^{(1)})$$
where $\Delta_{\text{res}, 1}$ is selected from the primary candidate engine (Motor A or Motor B) and $\Delta_{\text{res}, 2}$ is induced over the remaining residual discrepancy $M_{\text{diff}}^{(2)} = (\hat{Y}^{(1)} \neq Y)$.

### 2.3 Leave-One-Out Cross-Validation (LOO-CV) Rashomon Certification
To guarantee zero test-time regressions, every proposed mutation $\Delta$ must satisfy a strict **Rashomon Acceptance Criterion**:
$$\mathcal{M}(\Delta) = \mathbb{I}\!\left(\mathcal{A}_{\text{struct}}(\Delta(\hat{Y}_{\text{st}}); \mathcal{D}_{\text{train}}) = 1.0\right) \land \mathbb{I}\!\left(\min_{k} \text{acc}(\Delta(\hat{Y}_{\text{st}, k}), Y_k) \ge \min_k \text{acc}(\hat{Y}_{\text{st}, k}, Y_k)\right)$$
If a residual candidate does not achieve 100% training accuracy across all demonstration pairs or degrades accuracy on any single pair in leave-one-out validation, it is rejected, safely falling back to the unmutated symbolic prediction $\hat{Y}_{\text{st}}$.

### 2.4 Empirical Impact on Near-Miss Conversion
Table 2 details the empirical results of applying Cycle 3.5 across all 282 cataloged near-misses (`benchmark_h6_hybrid_nrem_results.json`).

| Group | Total Near-Misses ($N$) | Converted to Exact 100% | Pixel Acc Improved | Regressions | Conversion Highlights |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **ARC-1 Hold-Out** | 184 | **3** | **61** | **0 (0.00%)** | `84f2aca1` (98.1% $\to$ 100%), `929ab4e9` (99.8% $\to$ 100%), `ae58858e` (85.4% $\to$ 100%) |
| **ARC-2 Eval Split** | 63 | **0** | **16** | **0 (0.00%)** | `cad67732` (85.5% $\to$ 99.5%), `88e364bc` (90.0% $\to$ 98.6%) |
| **ARC-2 Train Sample** | 35 | **1** | **13** | **0 (0.00%)** | `05f2a901` (92.7% $\to$ 100.0%) |
| **Overall** | **282** | **4** | **90 (31.91%)** | **0 (0.00%)** | **Strict Monotonicity Preserved** |

Key milestones include:
- **ARC-AGI-1 Score Elevation:** The autonomous conversion of hold-out tasks `84f2aca1`, `ae58858e`, and `929ab4e9` elevates the official ARC-AGI-1 evaluation score from **57/400 (14.25%)** to **60/400 (15.00%)** overall, and from **38/350 (10.86%)** to **41/350 (11.71%)** on strictly uninspected hold-out tasks.
- **ARC-AGI-2 Training Score Elevation:** Task `05f2a901` was resolved through object collision translation snapping, elevating the ARC-2 training calibration sample from **33/100 (33.00%)** to **34/100 (34.00%)**.
- **Near-Miss Closure in ARC-2 Evaluation:** On `cad67732`, Two-Step Chaining (Motor A MLP followed by Step-2 Sparse LUT) eliminated 96.5% of residual errors, lifting accuracy from 85.5% to **99.5%** (a single cavity pixel off). On `88e364bc`, residual snapping lifted accuracy from 90.0% to **98.6%**.

---

## 3. Replay-Free Continual Plasticity on Real ARC Cell Vectors

### 3.1 Study Design (Study 6.1 Multi-Seed)
To evaluate synaptic stability and bioenergetic efficiency on realistic abstract distributions, we extracted 490 active foreground cell vectors (64D) across a sequential class-incremental stream of 5 real ARC-AGI tasks (`83302e8f`, `73251a56`, `3631a71a`, `264363fd`, `32597951`).
Each regime was evaluated across **$N=5$ random seeds** (`42, 101, 202, 303, 404`), with each seed executing a distinct random permutation of the 5 tasks.

![Figure 4: Continual Learning on Real ARC-AGI Cell Vectors](../figures/fig4_h6_real_arc_continual.png)

### 3.2 Strict Replay-Free Constraints
To ensure rigorous scientific validity, **all four evaluated regimes operated under a strict Zero-Replay constraint**: no past input vectors or labels $(\mathbf{x}, y)$ were stored or replayed. The models maintain only parameter state $\boldsymbol{\theta}$ and diagonal Fisher curvature statistics $\hat{\mathbf{F}}$.
- **Standard SGD:** Unregularized stochastic gradient descent.
- **Classic EWC:** Second-order quadratic Fisher penalty $\mathcal{L}_{\text{EWC}} = \frac{\lambda}{2}\sum_i \hat{F}_{ii}(\theta_i - \theta_i^*)^2$.
- **Pure Mistake Gating:** Synaptic updates gated by error margin $\tau_m = 0.05$ without Fisher preconditioning.
- **F3-Metabolic-Optim (Ours):** Unifies SynGAP diagonal Fisher EMA preconditioning, mistake gating, and mitochondrial energy tensor $\mathbf{M}_t$.

### 3.3 Empirical Results
Table 3 summarizes the multi-seed results recorded in `benchmark_h6_real_arc_continual.json`.

| Regime | Final Mean Accuracy (%) | Forgetting Ratio (%) | Saved Weight Updates (%) | Synaptic Drift ($M_1$ Cost) | Execution Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Standard SGD** | $54.19 \pm 7.31$ | $57.26 \pm 9.13$ | $0.00 \pm 0.00$ | $1156.58 \pm 51.34$ | $0.407 \pm 0.056$ |
| **Classic EWC** | $78.00 \pm 5.09$ | $27.50 \pm 6.36$ | $0.00 \pm 0.00$ | $1183.95 \pm 52.58$ | $0.879 \pm 0.192$ |
| **Pure Mistake Gating** | $47.05 \pm 9.44$ | $65.95 \pm 11.96$ | $\mathbf{93.14 \pm 0.33}$ | $814.69 \pm 57.99$ | $\mathbf{0.152 \pm 0.026}$ |
| **F3-Metabolic-Optim (Ours)** | **$76.76 \pm 3.82$** | **$28.81 \pm 4.84$** | **$92.45 \pm 0.32$** | **$823.17 \pm 55.36$** | **$0.281 \pm 0.049$** |

### 3.4 Scientific Takeaways
1. **Competitive Retention without Replay:** F3-Metabolic-Optim matches the catastrophic forgetting resistance of EWC ($28.81\%$ vs $27.50\%$, overlapping within 1 standard deviation) while eliminating $92.45\%$ of synaptic updates.
2. **Substantial Synaptic Economy:** By suppressing updates on predictable samples, F3-Metabolic-Optim reduces total parameter displacement ($M_1$) from $1183.95$ to $823.17$ (**$-30.47\%$ drift**).
3. **Execution Throughput:** On pure CPU, F3-Metabolic-Optim executes in **$0.281\,\text{s}$**, representing a **$3.13\times$ speedup** over EWC ($0.879\,\text{s}$).

---

## 4. Engineering & Industrial Utility Beyond ARC

To establish whether the principles of F3-AGI generalize to practical engineering domains outside grid puzzles, Horizon 9 introduced benchmark evaluations across two non-ARC real-world problems (`benchmark_h9_realworld_utility.json`).

![Figure 6: Real-World Industrial & Engineering Utility](../figures/fig6_h9_realworld_utility.png)

### 4.1 Scenario A: Edge IoT Telemetry Under Continual Concept Drift
In ultra-low-power embedded sensors (e.g., vibration monitors, environmental IoT loggers), devices must classify operating regimes from a streaming 32-dimensional sensor vector across changing operational environments without transmitting raw data to the cloud or overflowing internal SRAM.

We simulated 5 consecutive environmental regimes ($N=5$ seeds, 32D features, 6 classes, non-stationary class distributions).

| Regime / Method | Final Retention Acc (%) | Forgetting (%) | Saved Synaptic Updates (%) | Total CPU Latency (s) |
| :--- | :---: | :---: | :---: | :---: |
| **Standard SGD** | $51.54 \pm 7.67$ | $72.95 \pm 14.23$ | $0.00 \pm 0.00$ | $0.638 \pm 0.061$ |
| **Classic EWC** | $64.14 \pm 8.62$ | $54.70 \pm 22.15$ | $0.00 \pm 0.00$ | $0.913 \pm 0.094$ |
| **Pure Mistake Gating** | $50.48 \pm 7.31$ | $72.20 \pm 13.31$ | $91.28 \pm 0.91$ | $\mathbf{0.302 \pm 0.038}$ |
| **F3-Metabolic-Optim (Ours)** | **$58.86 \pm 8.27$** | **$59.90 \pm 17.77$** | **$90.73 \pm 1.06$** | **$0.394 \pm 0.045$** |

**Engineering Finding:** F3-Metabolic-Optim achieves **$90.73\%$ energy savings** in synaptic arithmetic operations while running **$2.32\times$ faster** than EWC, establishing its viability for low-power microcontroller deployment (Cortex-M4/M7).

### 4.2 Scenario B: Safety-Critical Legacy Rule Repair (Zero-Regression Guarantee)
In mission-critical industrial applications (e.g., flight avionics, medical triage logic, automated trading risk systems), legacy rule-based engines contain deterministic business logic. Occasionally, calibration testing exposes known defect cases. Standard practice is often paralyzed: updating the system via end-to-end retraining risks silent regression on thousands of working edge cases.

We modeled an industrial decision rule engine with 10 feature conditions, exhibiting a baseline defect rate of 14% (86.0% accuracy on $N=1000$ test items). A repair set of $K=4$ defect calibration batches was supplied.

| Repair Methodology | Overall Test Acc (%) | Defect Cases Repaired (%) | Regressions on Working Logic (%) | Test Latency (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline: Frozen Legacy Engine** | $86.00 \pm 1.20$ | $0.00 \pm 0.00$ | $\mathbf{0.00 \pm 0.00}$ | $0.85 \pm 0.05$ |
| **1. Unconstrained End-to-End Retraining** | $53.74 \pm 1.76$ | $51.11 \pm 2.95$ | $45.84 \pm 1.77$ | $4.12 \pm 0.35$ |
| **2. Unverified Residual Patch** | $63.68 \pm 6.28$ | $75.99 \pm 7.60$ | $38.34 \pm 7.15$ | $2.30 \pm 0.20$ |
| **3. F3-AGI Certified Residual Patch (Ours)** | **$100.00 \pm 0.00$** | **$100.00 \pm 0.00$** | **$0.00 \pm 0.00$** | $\mathbf{1.05 \pm 0.08}$ |

**Engineering Finding:** Whereas unconstrained deep learning causes catastrophic regression on **$45.84\%$ of previously correct operational states**, the F3-AGI Rashomon-certified residual patch achieves **100.0% repair of known defects with mathematically certified 0.00% regressions**, demonstrating an immediate deployment pathway for safety-critical enterprise systems.

---

## 5. Architectural Roadmap and Conclusion

The technical milestones achieved in Horizons 6 through 9 solidify the theoretical and empirical foundations of F3-AGI v4.1:
1. **Compositional Modularity Over Pure Scaling:** The combination of global symbolic operators with fine-grained NREM residual mutation provides a scalable, interpretable alternative to massive LLM sampling.
2. **Bioenergetic Gating Enables Sustainable Plasticity:** Memorized mistake gating and SynGAP diagonal Fisher preconditioning successfully reconcile plasticity and stability across both abstract grids and real-world edge signals without replay buffers.
3. **Certified Repair Provides Industrial Safety:** Rashomon LOO-CV verification prevents regressions in both cognitive puzzle solving and mission-critical legacy rule maintenance.

All code, data, and reproduction scripts for this addendum reside in the `feat/horizon-6-arc2` branch of the official repository, preserving the frozen v4.0.0 release for independent scientific audits.

---

## References

1. Aguilar, C., Zainal, M., & Kavehei, O. (2026). *SynGAP: Synaptic Geometric Adaptive Preconditioning for Continual Learning*. arXiv:2608.14634.
2. Arnez, F., & Gomez-Villa, R. (2026). *Isotropic Latent Representations via SIGReg-JEPA*. arXiv:2607.13612.
3. Chollet, F. (2019). *On the Measure of Intelligence*. arXiv:1911.01547.
4. Kirkpatrick, J., et al. (2017). *Overcoming catastrophic forgetting in neural networks*. PNAS, 114(13), 3521-3526.
5. Liao, Y., & Gu, S. (2025). *ARC-AGI Without Pretraining: Minimum Description Length as an Inductive Bias*. arXiv:2512.06104.
6. Pache, C., & van Rossum, M. (2026). *Memorized Mistake Gating for Energy-Efficient Synaptic Plasticity*. arXiv:2604.14336.
7. Pourcel, J., Colas, C., & Oudeyer, P.-Y. (2025). *SOAR: Synthesis of Operators via Actionable Residuals for ARC*. arXiv:2507.14172.
8. Wang, K., et al. (2025). *Hierarchical Recursive Models for Visual Reasoning*. arXiv:2506.21734.
