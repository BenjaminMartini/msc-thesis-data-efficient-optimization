# Data-Efficient Optimization Benchmarks

This repository contains the code and notebooks used in the Master's thesis

> **“Data-Efficient Optimization in Materials Informatics:  
> Benchmarking Surrogate Modeling and Experimental Design Algorithms”**

The Python modules (`.py` files) provide a small, self-contained benchmarking
framework. The actual experiments are driven through **Jupyter notebooks**
(`SB*.ipynb`), which import and orchestrate the underlying modules.

You typically **do not run the `.py` files directly**. Instead, you open the
sub-benchmark notebooks "SBX - name " which are corresponding to the sub-benhcmarks defined in Chapter 3.6 and execute them cell by cell.

## 1. Repository structure

### 1.1 Core modules (used as a library by the notebooks)

These modules are imported by the SB-notebooks and contain the reusable logic:

- `problems.py`  
  Definitions of benchmark objective functions and a `Problem` dataclass with
  affine mappings between original coordinate space and the canonical cube
  \[0, 1\]^d (Branin, Rosenbrock, Rastrigin, Sphere, Hartmann-6, Goldstein-Price).

- `methods.py`  
  Implementations of optimization methods with a unified interface:

  - Fixed designs (e.g. Latin Hypercube DOE, full-factorial DOE, CCD, D-optimal)
  - Random baseline
  - Gaussian process based Bayesian optimization with Expected Improvement (EI)
  - Bayesian optimization with Knowledge Gradient (KG, Monte Carlo approximation)

- `runner.py`  
  Core routine for a **single** experiment configuration. It

  - builds the problem,
  - instantiates the method,
  - runs for a fixed budget,
  - logs one CSV row per iteration.

- `main.py`  
  Orchestrates **full benchmark sweeps** over
  problems × methods × budgets × seeds × noise levels and writes CSV logs and
  (optionally) plots. In the thesis workflow, this is typically called from
  within the notebooks, not from the command line.

- `evaluator.py`  
  Evaluation and visualization utilities, including:

  - regret and best-so-far curves,
  - boxplots of (normalized) regret at a fixed budget T,
  - rank and winrate heatmaps,
  - best-at-T tables (true and observed regret, normalized regret).

- `evaluate_latest.py`  
  Convenience helpers that are used from the notebooks to:

  - find the most recent `results/<timestamp>_*` folder,
  - rebuild aggregate plots and tables from logs,
  - optionally generate an HTML summary report.

- `viz_samples.py`  
  Visualization routines for 2D problems:

  - contour and surface plots of the true objective and RSM surrogate,
  - sample locations and trajectories,
  - predicted optima overlays.

- `tb_profiles.py`  
  Pre-defined “test-bench profiles” (TB1–TB6, RSM profiles) that specify
  combinations of problems, budgets, seeds, and noise levels used in the thesis.

- `tb_style.py`  
  Helpers for configuring Matplotlib styles to produce consistent,
  publication-quality figures.

### 1.2 Sub-benchmark notebooks (main entry points)

The experiment logic for the thesis is structured into several sub-benchmarks:

- `SB1_DOE_head_to_head.ipynb`  
  DOE baselines head-to-head, using the shared logging and evaluation pipeline.

- `SB2_RSM_variants_on_shared_DOE.ipynb`  
  RSM variants evaluated on a fixed DOE design, including RSM2/RSM3 modes.

- `SB4_BO_EI_vs_KG_noise.ipynb`  
  Comparison of BO with Expected Improvement vs. Knowledge Gradient under noise.

- `SB5_Noise_sensitivity.ipynb`  
  Noise ladder experiments to study robustness across noise levels.

- `SB6_Budget_sensitivity.ipynb`  
  Budget scaling experiments to study data efficiency and convergence speed.

- `SB7_Problem_class.ipynb`  
  Cross-problem analysis and problem-class comparison.

Each notebook imports the modules listed in section 1.1, configures a specific
sub-benchmark, runs the underlying code, and produces the figures and tables
reported in the thesis.

## 2. Installation

### 2.1. Python version

The code is intended for **Python 3.10+**.

### 2.2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate      # on Linux / macOS
# .venv\Scripts\activate       # on Windows
```
