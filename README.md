# Polybee F3-AGI v4.0

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Hardware: CPU-First](https://img.shields.io/badge/Hardware-100%25%20CPU%20%7C%20%3C400MB%20RAM-success.svg)]()
[![Preprint](https://img.shields.io/badge/Preprint-cs.AI%20%7C%20cs.LG-red.svg)](paper/polybee_f3_agi_v4_arxiv.pdf)

> **Official Implementation of F3-AGI v4.0** by **[Polybee AI Research](https://polybee.ai/)**  
> *Bioenergetically-Gated Active Inference, Metaplastic Preconditioning, and Residual Program Synthesis for Sample-Efficient Abstract Reasoning.*

---

## Executive Summary

Current frontier approaches to abstract reasoning benchmarks (such as ARC-AGI-1, ARC-AGI-2, and interactive ARC-AGI-3) rely heavily on brute-force test-time sampling over massive Large Language Models (LLMs), consuming thousands of watt-hours per task while suffering from catastrophic forgetting, unconstrained exploration of dead-end states, and fragile pixel-level heuristics.

**F3-AGI v4.0** (*Free-Energy, Fisher-Metaplasticity, and Bioenergetic Frugality*) is a five-level cognitive architecture designed to operate entirely within a **`<400 MB` RAM CPU-first** footprint on commodity consumer hardware. Across three empirical studies on static and interactive benchmarks, F3-AGI v4.0 establishes:

1. **Continual Bioenergetic Plasticity (`F3-Metabolic-Optim`):** Evaluated on a **Synthetic 5-Task Class-Incremental Feature Stream (64D Prototype-Latent Benchmark)** ($6,000$ online steps, zero replay buffer), coupling online diagonal Fisher EMA preconditioning (**SynGAP**, Aguilar et al., 2026), **Memorized Mistake Gating** (Pache & van Rossum, 2026), and a virtual **Mitochondrial ATP Tensor** ($\mathbf{M}$) across orthogonal class prototypes ($\sigma_{\text{proto}} = 0.35$) and isotropic observation noise ($\sigma_{\text{obs}} = 0.42$). It achieves **91.67% final accuracy**, **9.17% forgetting ratio**, **93.47% saved synaptic updates**, and a **65.7% reduction in $M_1$ metabolic cost** with a **2.19x wall-clock speedup** over Elastic Weight Consolidation (EWC) ($1.060\,\text{s}$ vs. $2.321\,\text{s}$).
2. **Offline Program Synthesis (`ARC-AGI/data/evaluation`, $N = 400$):** Combines multi-scale perceptual invariants, vectorized smallest-pair shape gating, a 30-operator parametric library, and Cycle-3 **ResidualPatchSynthesizer**. Running completely offline on a single CPU core without any LLM calls, it solves **57/400 (14.25% `pass@2`; 56/400 `pass@1`)** with **128/400 (32.00%)** of tasks achieving $\ge 90\%$ test pixel accuracy, in an average of **1.62 s/task** (**647.5 s = 10.8 min total**, with a **42.5x controlled paired speedup** on the $N=50$ calibration slice: $50.61\,\text{s} \to 1.19\,\text{s/task}$). This comprises:
   - **Calibration Slice ($N=50$):** **19/50 (38.00%)** exact zero-shot solve rate ($100\%$ train-to-test transfer, `1.19 s/task`, 42.5x paired speedup).
   - **Strictly Uninspected Hold-Out ($N=350$, Tasks 51–400):** **38/350 (10.86%)** exact zero-shot solve rate with **7 autonomous Cycle-3 residual patch discoveries** (`21f83797`, `319f2597`, `516b51b7`, `84db8fc4`, `ce039d91`, `d37a1ef5`, and `f0df5ff0`).
3. **Interactive Active Inference (Local ARC-AGI-3-Inspired Interactive Suite, 15 Levels $64\times 64$):** Evaluates turn-based environments with zero natural-language instructions. Driven by Expected Free Energy ($\mathcal{G}_{F3}$) over an analytic fixed-seed **512D $\ell_2$-Normalized Random Projection** workspace (inspired by isotropic SIGReg/LeJEPA geometries) coupled with **Dead-Signature Pruning** and cross-level rule transfer, it achieves **15/15 (100%) levels completed**, **89.44% Level-Weighted RHAE** ($w=(1,2,3)$), **90.72% Unweighted RHAE**, and **0 dead actions** (down from 32).

---

## Architecture Overview

F3-AGI v4.0 is structured into five integrated cognitive levels:

```
+-----------------------------------------------------------------------------------+
| Level 4: Continuous Global Workspace                                              |
| - AlgebraicWorldModelSLS (Analytic 512D L2-Normalized Random Projection)          |
| - Dead-Signature Tracker & Cross-Level Rule Transfer                              |
| - NREM Consolidation Buffer (soar_nrem_traces.json & f3_learned_library.py)       |
+-----------------------------------------------------------------------------------+
                                        |
+-----------------------------------------------------------------------------------+
| Level 3: 512D L2-Random Proj. (SIGReg/LeJEPA-Inspired) + Active Inference         |
| - VisualCortexV40 (Invariant extraction over ALL demonstration pairs)             |
| - CompositionalProgramSynthesizer (30-operator parametric library)                |
| - Vectorized Smallest-Pair Shape Gate (Lemma 1: zero false negatives)             |
| - ResidualPatchSynthesizer (Cycle-3 local cellular & ray repair)                  |
+-----------------------------------------------------------------------------------+
                                        |
+-----------------------------------------------------------------------------------+
| Level 2: SynGAP Metaplastic Memory                                                |
| - Online Diagonal Fisher EMA Preconditioning (gamma_syn = 12.0)                   |
| - Plastic-Anchor Elastic Regularization (lambda_anc = 4.5)                         |
+-----------------------------------------------------------------------------------+
                                        |
+-----------------------------------------------------------------------------------+
| Level 1.5: Metabolic & Minimum Description Length (MDL) Controller                |
| - Memorized Mistake Gating (margin tau_m = 0.18, sparse error memory)             |
| - Mitochondrial ATP Tensor (M_l,j) with Attwell-Laughlin bioenergetic depletion   |
| - Kolmogorov/MDL AST penalty (beta_MDL = 0.05, Prop 1 certified margin >= 0.4498) |
+-----------------------------------------------------------------------------------+
                                        |
+-----------------------------------------------------------------------------------+
| Level 1: Multi-Scale Recursive Core (MSRC)                                        |
| - Fast Perceptual Recurrent Module (256D)                                         |
| - Slow Abstract Reasoning Recurrent Module (512D)                                 |
+-----------------------------------------------------------------------------------+
```

---

## Experimental Benchmark Results

### 1. Continual Plasticity Ablation (Study 1)

Comparison of continual learning regimes on the **Synthetic 5-Task Class-Incremental Feature Stream (64D Prototype-Latent Benchmark)** (orthogonal prototypes with $\sigma_{\text{proto}} = 0.35$, sample observations with $\sigma_{\text{obs}} = 0.42$, $6,000$ online forward steps, zero replay buffer):

| Plasticity Regime (`Synthetic 5-Task Benchmark`, 6000 Steps) | Final Avg. Accuracy (%) $\uparrow$ | Forgetting Ratio (%) $\downarrow$ | Saved Updates (%) $\uparrow$ | Synaptic Updates $\downarrow$ | $M_1$ Metabolic Cost ($L_1$) $\downarrow$ | Wall Time (s) $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Standard SGD / Adam Baseline | 44.33% | 69.58% | 0.00% | 6,000 / 6,000 | 3,433.87 | 0.915s |
| Classic EWC (Fisher Penalty) | 90.67% | 11.25% | 0.00% | 6,000 / 6,000 | 7,986.85 | 2.321s |
| Pure Memorized Mistake Gating | 50.67% | 61.25% | **95.97%** | **242 / 6,000** | 3,040.31 | **0.293s** |
| **F3-Metabolic-Optim (Ours)** | **91.67%** | **9.17%** | 93.47% | 392 / 6,000 | **2,736.81** | 1.060s |

*F3-Metabolic-Optim achieves a **2.19x wall-clock speedup** ($1.060\,\text{s}$ vs. $2.321\,\text{s}$) over Classic EWC.*

<p align="center">
  <img src="figures/fig1_metabolic_ablation.png" width="90%" alt="Continual Plasticity Ablation">
</p>

---

### 2. Full-Scale Official Evaluation on `ARC-AGI/data/evaluation` ($N = 400$) (Study 2)

Evaluated 100% offline on a single CPU core without external API or LLM calls:

| Official Full-Benchmark Metric (`ARC-AGI/data/evaluation`, $N = 400$) | F3-AGI v4.0 Result | Verification & Scientific Notes |
| :--- | :---: | :--- |
| **Kaggle ARC Prize 2026 Schema Validation** | **100% Valid (400 / 400)** | Rectangular grids $1 \le H, W \le 30$, integer colors $0\dots 9$ |
| **Exact Blind Test Solved — `pass@1` (`attempt_1`)** | **56 / 400 (14.00%)** | Exact pixel-for-pixel match across all test outputs |
| **Exact Blind Test Solved — `pass@2` (`attempt_1` $\lor$ `attempt_2`)** | **57 / 400 (14.25%)** | Official ARC Prize `pass@2` metric |
| **Exclusive Second-Attempt Rescues (`attempt_2_rescue_count`)** | **+1 task (`9a4bb226`)** | `largest_object` ($0\%$) $\to$ `smallest_object` ($100\%$) |
| **Exact Demonstration Fit ($\mathcal{A}_{\text{struct}} = 1.0$ on Train)** | 61 / 400 (15.25%) | 86.89% zero-shot Train $\to$ Test generalization |
| **High-Precision Train Stratum ($\mathcal{A}_{\text{struct}} \ge 90.0\%$)** | 119 / 400 (29.75%) | Targeted candidates for Cycle-3 residual repair |
| **High-Precision Blind Test Pixel Accuracy ($\ge 90.0\%$)** | **128 / 400 (32.00%)** | ~32% of all evaluation tasks at $\ge 90\%$ test pixel accuracy |
| **Total Wall-Clock Execution Time ($N=400$, Single CPU)** | **647.51 s (10.79 min)** | **1.619 s/task** mean latency across all 400 tasks (**42.40x controlled paired speedup** on $N=50$) |

#### Split Decomposition:
- **Calibration Slice ($N = 50$, Tasks 1–50):** **19 / 50 (38.00%)** exact blind test solve rate (**`1.19 s/task`**, **42.5x paired speedup** over Horizon 2, 100% train-to-test transfer).
- **Strictly Uninspected Hold-Out ($N = 350$, Tasks 51–400):** **38 / 350 (10.86%)** exact blind test solve rate with **7 autonomous discoveries** by Cycle-3 `ResidualPatchSynthesizer`.

<p align="center">
  <img src="figures/fig2_arc_blind_eval.png" width="90%" alt="ARC Blind Evaluation">
</p>

---

### 3. Interactive ARC-AGI-3-Inspired Benchmark (Study 3)

Evaluated across 5 complex mechanics (15 levels total, $64\times 64$ turn-based grids):

| Interactive Game ($64\times 64$, 3 Levels/Game) | Levels Completed | Agent Actions (L1/L2/L3 $\to$ Total) | Human Baseline (L1/L2/L3 $\to$ Total) | Level-Weighted RHAE (%) $\uparrow$ | Unweighted RHAE (%) $\uparrow$ | Dead Actions $\downarrow$ | ATP Consumed $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. `SwitchAndDoorMaze` | **3 / 3** | $5 / 7 / 11$ (**23**) | $5 / 7 / 9$ ($21$) | **83.47%** | **88.98%** | **0** | 0.460 |
| 2. `SokobanPushBlocks` | **3 / 3** | $6 / 9 / 10$ (**25**) | $5 / 7 / 8$ ($20$) | **63.74%** | **64.64%** | **0** | 0.500 |
| 3. `ColorInfectionToggle` | **3 / 3** | $3 / 4 / 5$ (**12**) | $3 / 4 / 5$ ($12$) | **100.00%** | **100.00%** | **0** | 0.240 |
| 4. `IceSlideInertia` | **3 / 3** | $2 / 3 / 4$ (**9**) | $2 / 3 / 4$ ($9$) | **100.00%** | **100.00%** | **0** | 0.180 |
| 5. `TeleporterKeyPortal` | **3 / 3** | $5 / 6 / 7$ (**18**) | $5 / 6 / 7$ ($18$) | **100.00%** | **100.00%** | **0** | 0.360 |
| **Suite Total / Mean (15 Levels)** | **15 / 15 (100.0%)** | **87 total actions** | **80 human actions** | **89.44%** | **90.72%** | **0** | **1.740 ATP** |

#### Interactive Agent Ablation:

| Agent Variant | Levels Completed ($/15$) $\uparrow$ | Level-Weighted RHAE (%) $\uparrow$ | Unweighted RHAE (%) $\uparrow$ | Wasted Dead Actions $\downarrow$ | Metabolic ATP $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random / Greedy Heuristic | 5 / 15 (33.33%) | 6.19% | 10.11% | 82 | 4.000 |
| Pragmatic-Only (No Epistemic / No Dead-Signature) | 3 / 15 (20.00%) | 9.68% | 15.33% | 119 | 4.000 |
| Active Inference w/o Dead-Signature Pruning | **15 / 15 (100.00%)** | 46.24% | 47.17% | 32 | 2.380 |
| **F3-AGI v4.0 Complete ($\mathcal{G}_{F3}$ + 512D Random Proj. + Dead-Signature)** | **15 / 15 (100.00%)** | **89.44%** | **90.72%** | **0** | **1.740** |

<p align="center">
  <img src="figures/fig3_arc3_interactive_ablation.png" width="90%" alt="Interactive Suite Ablation">
</p>

---

## Repository Structure

```
polybee-f3-agi/
├── f3_agi_v4_0_core.py          # Core 5-level neuro-symbolic engine (<400MB RAM, pure NumPy)
├── f3_learned_library.py        # 30 parameterized operators (Tier 1 & Tier 2) + 22 solved_* routines
├── f3_agi_v4_arc3_runner.py     # Interactive ARC-AGI-3 runner, DeadSignatureTracker & 4-agent ablation
├── f3_metabolic_optim.py        # Continual learning benchmark (SynGAP + Mistake Gating + Tensor M)
├── generate_paper_figures.py    # Figure generation script producing 300 DPI PNG and vector SVG figures
├── kaggle_submission/           # Standalone Kaggle ARC Prize submission bundle & schemas
│   ├── f3_agi_v4_kaggle_submission.py
│   ├── submission_full400.json
│   └── verification_report_full400.json
├── figures/                     # High-resolution academic figures (Figures 1, 2, and 3)
│   ├── fig1_metabolic_ablation.{png,svg}
│   ├── fig2_arc_blind_eval.{png,svg}
│   └── fig3_arc3_interactive_ablation.{png,svg}
├── paper/                       # Complete arXiv preprint manuscript and compiled PDF
│   ├── polybee_f3_agi_v4_arxiv.pdf # Canonical compiled preprint PDF
│   ├── polybee_f3_agi_v4_arxiv.tex # LaTeX manuscript
│   └── polybee_f3_agi_v4_arxiv.md  # Markdown manuscript
├── benchmark_h3_results.json    # Verified telemetry logs for Horizon 3
├── benchmark_h3_arc3_ablation.json # Verified per-game telemetry for the 4 interactive agents
├── benchmark_h5_blind_eval.json # Verified longitudinal telemetry for N=50 and N=400
├── benchmark_h5_full400_eval.json # Full 400-task official verification report
├── f3_metabolic_optim_results.json # Verified telemetry for the continual plasticity study
├── requirements.txt             # Minimal dependencies (numpy, matplotlib)
└── README.md                    # Project documentation and reproduction guide
```

---

## Reproduction Instructions

### 1. Installation

Clone the repository and install the lightweight dependencies (no GPU required):

```bash
git clone https://github.com/henrique-polybee/f3-agi-v4.git
cd f3-agi-v4
pip install -r requirements.txt
```

### 2. Reproduce Study 1 (Continual Plasticity Ablation)

Runs the 5-task continual classification benchmark across all 4 plasticity regimes:

```bash
python f3_metabolic_optim.py
```
*Outputs: `f3_metabolic_optim_results.json`.*

### 3. Reproduce Study 3 (Interactive ARC-AGI-3-Inspired Suite & 4-Agent Ablation)

Runs the 5 games (15 levels, $64\times 64$) comparing Greedy, Pragmatic-Only, Active Inference w/o Dead-Signature, and F3-AGI v4.0 Complete:

```bash
python f3_agi_v4_arc3_runner.py --ablation
```
*Outputs: `benchmark_h3_arc3_ablation.json` (reporting both Level-Weighted and Unweighted RHAE).*

### 4. Regenerate All Paper Figures

Regenerates Figures 1, 2, and 3 in high-resolution PNG (`300 DPI`) and scalable SVG:

```bash
python generate_paper_figures.py
```
*Outputs saved to `figures/`.*

### 5. Reproduce Study 2 (Full ARC-AGI Evaluation Split)

Download the official ARC-AGI dataset into `ARC-AGI/`:

```bash
git clone https://github.com/fchollet/ARC-AGI.git
python kaggle_submission/f3_agi_v4_kaggle_submission.py
```
*Generates `kaggle_submission/submission_full400.json` and `kaggle_submission/verification_report_full400.json` in ~10.79 minutes.*

---

## Citation

```bibtex
@article{oliveira2026f3agi,
  title={{F3-AGI v4.0: Bioenergetically-Gated Active Inference, Metaplastic Preconditioning, and Residual Program Synthesis for Sample-Efficient Abstract Reasoning}},
  author={Oliveira, Henrique and {Polybee AI Research}},
  journal={Preprint [cs.AI, cs.LG]},
  year={2026},
  url={https://polybee.ai/}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
