# Active Inference as a Normative Framework for Adaptive CT Acquisition

Simulation code for:

> **Active Inference as a Normative Framework for Adaptive CT Acquisition**  
> [Authors], *Physics in Medicine and Biology*, 2025  
> DOI: [to be added on acceptance]

---

## Overview

This repository contains the simulation code and figure-generation scripts
for the controlled Poisson CT benchmark described in the paper. The benchmark
evaluates five acquisition controllers under nonlinear (Poisson) and
linearised measurement models, demonstrating that:

1. Bayesian experimental design and Bayesian decision theory are recoverable
   as limiting cases of the Active Inference expected free energy objective.

2. Multi-step planning confers a measurable task-success advantage under
   Poisson measurement statistics, where information gain is belief-dependent.

The simulation uses a low-dimensional phantom (D=9: one lesion component,
eight nuisance components) that permits exact finite-horizon policy
enumeration, isolating the effect of the controller objective from search
approximation.

---

## Repository structure

```
aif-ct-acquisition/
├── bayes_aif_v1.py               # Main simulation — runs all five controllers
├── bayes_aif_cross_condition.py  # Cross-condition experiment (Section 5.4)
├── pmb_three_figures.py          # Generates the three paper figures
├── README.md
└── figures/                      # Created automatically on first run
    ├── nonlinear_clustered_b0p20_l0p06_summary.csv
    ├── linear_clustered_b0p20_l0p06_summary.csv
    ├── fig1_planning_horizon.pdf
    ├── fig3_benchmark_bars.pdf
    └── ...
```

---

## Requirements

```
Python >= 3.10
numpy >= 1.24
matplotlib >= 3.7
```

Install with:

```bash
pip install numpy matplotlib
```

---

## Reproducing the paper results

### Step 1 — Run the nonlinear (main) condition

```bash
cd aif-ct-acquisition
python bayes_aif_v1.py
```

In `main()` the defaults are:

```python
seed       = 2
rate_model = "nonlinear"
structure  = "clustered"
```

Runtime: approximately 8–10 hours on a laptop (100 trials, T=3
requires enumerating up to 90,720 candidate policies per step).
Output: `figures/nonlinear_clustered_b0p20_l0p06_summary.csv`

### Step 2 — Run the linearised condition

Edit the two lines in `main()`:

```python
rate_model = "linear"
structure  = "clustered"
```

Runtime: approximately 3–5 hours (linear rate function is faster).
Output: `figures/linear_clustered_b0p20_l0p06_summary.csv`

### Step 3 — Generate the paper figures

```bash
python pmb_three_figures.py
```

Edit the four path variables at the top of `pmb_three_figures.py`
to point at your `figures/` directory, then run. Produces:

| File | Figure in paper |
|------|----------------|
| `fig1_planning_horizon.pdf` | Fig. 1 — Reach and dose vs planning horizon |
| `fig3_benchmark_bars.pdf`   | Fig. 2 — Benchmark bar chart               |

### Step 4 — Cross-condition experiment (optional)

To reproduce the model-misspecification diagnostic (Section 5.4):

```bash
python bayes_aif_cross_condition.py
```

This runs three conditions automatically (nonlinear/nonlinear,
linear/linear, nonlinear data with linear controller) and saves
a summary CSV for each.

### Step 5 — Sensitivity to view geometry (optional)

Edit `main()` in `bayes_aif_v1.py`:

```python
rate_model = "nonlinear"
structure  = "random"
```

This uses randomly generated view sensitivity vectors rather than
the clustered geometry used in the main experiments.

---

## Controllers

| Name | Mode | Planning horizon | Epistemic term | Pragmatic term | Dose term |
|------|------|-----------------|----------------|----------------|-----------|
| Design T=1   | design   | T=1 | ✓ | ✗ | ✗ (budget cap) |
| Decision T=1 | decision | T=1 | ✗ | ✓ | ✓ |
| AIF T=1      | aif      | T=1 | ✓ | ✓ | ✓ |
| AIF T=2      | aif      | T=2 | ✓ | ✓ | ✓ |
| AIF T=3      | aif      | T=3 | ✓ | ✓ | ✓ |

Design and Decision are limiting cases of the AIF expected free
energy objective, obtained by suppressing the pragmatic or epistemic
component respectively (see paper Section 3).

---

## Key parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d0` | 1.7 | Detectability threshold (task success criterion) |
| `beta_aif` | 0.20 | Epistemic weight in AIF objective |
| `lam` | 0.06 | Dose penalty weight |
| `n_trials` | 100 | Trials per controller |
| `seed` | 2 | Random seed (main result) |
| `D` | 9 | Hidden state dimension (1 lesion + 8 nuisance) |
| `n_views` | 16 | Number of projection views |
| `E_grid` | [0.3, 1.0, 3.0] | Discrete exposure levels |
| Max dose budget | 15.0 | Hard cap for Design controller only |

---

## Output CSV format

`{stem}_summary.csv` — one row per controller:

| Column | Description |
|--------|-------------|
| `Controller` | Controller name |
| `dprime_ge_d0_pct` | Task success rate (%) |
| `dprime_ge_d0_n` | Raw count of successful trials |
| `N` | Total trials |
| `Dose_med/q1/q3` | Median and IQR of total dose |
| `dprime_med/q1/q3` | Median and IQR of final d' |
| `S11_med/q1/q3` | Median and IQR of lesion posterior variance |
| `logdet_med/q1/q3` | Median and IQR of log-determinant of posterior |

`{stem}_policy_map.csv` — one row per acquisition step:

| Column | Description |
|--------|-------------|
| `controller` | Controller name |
| `trial` | Trial index |
| `step` | Acquisition step index |
| `m1` | Posterior mean of lesion component |
| `s11` | Posterior variance of lesion component |
| `exposure` | Chosen exposure level |

---

## Policy surface computation

The functions `compute_policy_surface`, `plot_policy_surface_grid`,
and `save_policy_surface_csv` are included in `bayes_aif_v1.py` but
are not called in `main()` by default. The T=3 surface requires
enumerating ~90,720 policies at each of ~1,350 grid points and takes
approximately 6 hours on a laptop. To generate surfaces for
supplementary material, uncomment the block at the end of `main()`.

---

## Reproducibility

All randomness is controlled by a single integer `seed` passed to
`numpy.random.default_rng`. With the same seed, every run produces
identical results. The seed controls:

- View sensitivity matrix G (model geometry)
- Trial hidden states z (lesion and nuisance values)
- Poisson measurement noise (per controller, per trial, per step)

The mapping is:
- Model geometry: `default_rng(seed)`
- Trial t: `default_rng(seed + 5000 + t)`
- Controller k, trial t: `default_rng(seed + 100000*(k+1) + t)`

---

## Citation

```bibtex
@article{[key]2025aif_ct,
  author  = {[Authors]},
  title   = {Active Inference as a Normative Framework for
             Adaptive {CT} Acquisition},
  journal = {Physics in Medicine and Biology},
  year    = {2025},
  doi     = {[to be added]}
}
```

---

## Licence

[MIT / CC-BY / institutional licence — add as appropriate]
