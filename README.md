# Active Inference as a Normative Framework for Adaptive CT Acquisition

Simulation code accompanying:

Active Inference as a Normative Framework for Adaptive CT Acquisition. *Physics in Medicine and Biology* (under review).

---

## Overview

This repository contains the Python simulation code used to generate all results and figures in the paper. The simulation implements five controllers in a compressed CT benchmark and evaluates them under nonlinear (Poisson) and linearised measurement models.

The five controllers are:

- **Bayesian Design T=1** — epistemic limit (information gain only)
- **Bayesian Decision T=1** — pragmatic limit (task utility and dose only)
- **AIF T=1, T=2, T=3** — Active Inference at planning horizons 1, 2, and 3

---

## Files

| File | Description |
|------|-------------|
| `bayes_aif_v1.py` | Main simulation — nonlinear and linearised benchmark, n=400 trials |
| `pmb_three_figures.py` | Generates paper figures from summary CSVs |
| `bayes_aif_cross_condition.py` | Cross-condition experiment — decomposes the model gap into measurement and inference effects |
| `bayes_aif_sensitivity.py` | Hyperparameter sensitivity screen — 3×3 grid at n=20 trials |
| `bayes_aif_sensitivity_n100.py` | Sensitivity confirmation — lambda=0.06 column at n=100 trials |

---

## Requirements

```
Python >= 3.10
numpy >= 1.24
scipy
matplotlib
```

Install with:

```bash
pip install numpy scipy matplotlib
```

---

## Usage

### Main benchmark

```bash
python bayes_aif_v1.py
```

Runs all five controllers under both measurement models at n=400 trials. Output is printed to the console and written as CSV files to `figures/`.

Configure the run by editing the `CONFIGURE` block near the top of the script:

```python
RATE_MODEL = "nonlinear"   # "nonlinear" or "linear"
STRUCTURE  = "clustered"   # "clustered" or "random"
N_TRIALS   = 400
SEED       = 2
```

**Runtime:** approximately 8--10 hours per model at T=3 on a laptop.

### Generate figures

```bash
python pmb_three_figures.py
```

Reads the CSV files produced by the main benchmark and writes two figures:

- `figures/fig1_planning_horizon.png` — task success and dose vs planning horizon
- `figures/fig3_benchmark_bars.png` — task success by controller and model

### Cross-condition experiment

```bash
python bayes_aif_cross_condition.py
```

Runs three conditions to decompose the performance gap between the nonlinear and linearised models:

| Condition | Data | Controller | Purpose |
|-----------|------|------------|---------|
| A | Nonlinear (Poisson) | Nonlinear | Matched — main result |
| B | Linearised | Linearised | Matched — linear baseline |
| C | Nonlinear (Poisson) | Linearised | Diagnostic — misspecified controller |

Comparing A, B, and C separates the contribution of Poisson measurement statistics (C minus B) from the contribution of correct likelihood inference (A minus C). Results are reported in Table 3 of the paper.

**Runtime:** approximately 3--4 hours at n=100 trials.

### Hyperparameter sensitivity

**Quick screen across the full 3x3 grid (n=20 per condition):**

```bash
python bayes_aif_sensitivity.py
```

Evaluates all nine combinations of beta in {0.05, 0.20, 0.80} and lambda in {0.02, 0.06, 0.25}. Runtime approximately 13 hours for the full grid. Individual rows can be commented out in the `GRID` list to run subsets.

**Targeted confirmation at n=100 (lambda=0.06 column only):**

```bash
python bayes_aif_sensitivity_n100.py
```

Runs beta=0.05 and beta=0.80 at lambda=0.06 and n=100 trials. The paper values (beta=0.20, lambda=0.06) are available from the main n=400 run and are not repeated. Runtime approximately 8--10 hours.

---

## Paper values

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Epistemic weight | beta | 0.20 |
| Dose penalty | lambda | 0.06 |
| Detectability threshold | d0 | 1.7 |
| Maximum dose budget | — | 15.0 |
| State space dimension | D | 9 |
| Random seed (main runs) | — | 2 |

---

## Notation

The code variable names differ from the paper's mathematical notation in two places:

| Code | Paper | Reason |
|------|-------|--------|
| `z` | `s` | Hidden attenuation state |
| `lam` | `lambda` | `lambda` is a reserved keyword in Python |

---

## Citation

```bibtex
@article{Coleman2025AIFCTAcquisition,
  author  = {},
  title   = {Active Inference as a Normative Framework for
             Adaptive {CT} Acquisition},
  journal = {Physics in Medicine and Biology},
  year    = {2025}
}
```

---

## Licence

MIT Licence.
MIT Licence.
