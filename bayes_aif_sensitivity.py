"""
bayes_aif_sensitivity_n100.py
==============================
Targeted n=100 runs for the lambda=0.06 column of the
hyperparameter sensitivity grid, excluding the paper values
(beta=0.20, lambda=0.06) which are already run at n=400.

Conditions:
  beta=0.05, lambda=0.06   -- low epistemic weight
  beta=0.80, lambda=0.06   -- high epistemic weight

These two conditions bracket the paper values and allow the
sentence: "at lambda=0.06, the planning advantage was evaluated
at n=100 across the full beta range [0.05, 0.80]."

Runtime: ~8-10 hours total (T=3, nonlinear, n=100 each).

OUTPUT
------
  sensitivity_b0p05_l0p06_n100_summary.csv
  sensitivity_b0p80_l0p06_n100_summary.csv

USAGE
-----
Place alongside bayes_aif_v1.py and run:

    python bayes_aif_sensitivity_n100.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bayes_aif_v1 import (
    POLICY_CACHE,
    CtrlSpec,
    make_benchmark_model,
    run_one,
    sample_trial,
    save_summary_csv,
    summarise_results,
)

# =============================================================================
# Configuration
# =============================================================================

SEED       = 2            # same seed as all other runs
N_TRIALS   = 100
RATE_MODEL = "nonlinear"
STRUCTURE  = "clustered"
D0         = 1.7
LAM        = 0.06         # fixed -- paper value

# Only the two bracketing beta values; centre (0.20) already done at n=400
BETAS = [0.05, 0.80]


def main() -> None:
    print("\nHYPERPARAMETER SENSITIVITY -- n=100, lambda=0.06")
    print("  rate_model =", RATE_MODEL)
    print("  structure  =", STRUCTURE)
    print(f"  lambda     = {LAM}  (fixed at paper value)")
    print(f"  n_trials   = {N_TRIALS} per condition")
    print(f"  beta values: {BETAS}")
    print(f"  (beta=0.20 already done at n=400 -- not repeated)")
    print()

    model = make_benchmark_model(
        seed=SEED, rate_model=RATE_MODEL, structure=STRUCTURE
    )

    trials = [
        sample_trial(np.random.default_rng(SEED + 5000 + t), model)
        for t in range(N_TRIALS)
    ]

    for beta_aif in BETAS:
        print(f"\n{'='*50}")
        print(f"beta={beta_aif:.2f}   lambda={LAM:.2f}")
        print(f"{'='*50}")

        POLICY_CACHE.clear()

        ctrls = [
            CtrlSpec("Design T=1",   mode="design",   T=1, beta=0.0,
                     lam=LAM, d0=D0, dose_budget=15.0),
            CtrlSpec("Decision T=1", mode="decision", T=1, beta=0.0,
                     lam=LAM, d0=D0),
            CtrlSpec("AIF T=1",      mode="aif",      T=1, beta=beta_aif,
                     lam=LAM, d0=D0),
            CtrlSpec("AIF T=2",      mode="aif",      T=2, beta=beta_aif,
                     lam=LAM, d0=D0),
            CtrlSpec("AIF T=3",      mode="aif",      T=3, beta=beta_aif,
                     lam=LAM, d0=D0),
        ]

        results: dict[str, list] = {c.name: [] for c in ctrls}

        for k, spec in enumerate(ctrls):
            for t, trial in enumerate(trials):
                print(f"\r  {spec.name}: trial {t+1}/{N_TRIALS}",
                      end="", flush=True)
                rng_run = np.random.default_rng(
                    SEED + 100000 * (k + 1) + t
                )
                results[spec.name].append(
                    run_one(rng_run, model, trial, spec)
                )
            print()

        summarise_results(results, d0=D0)

        stem = (
            f"sensitivity_b{beta_aif:.2f}_l{LAM:.2f}_n100"
        ).replace(".", "p")
        save_summary_csv(results, d0=D0, stem=stem)
        print(f"  Saved: {stem}_summary.csv")

    print("\n\nDONE.")
    print("  Combine with n=400 paper values (beta=0.20, lambda=0.06)")
    print("  to report the full lambda=0.06 column at n>=100.")


if __name__ == "__main__":
    main()