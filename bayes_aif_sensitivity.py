"""
bayes_aif_sensitivity.py
========================
Hyperparameter sensitivity analysis for:

  "Active Inference as a Normative Framework for Adaptive CT Acquisition"
  Physics in Medicine and Biology, 2025.

Runs the nonlinear clustered benchmark across a 3x3 grid of
(beta, lambda) values to assess sensitivity of the key result
(AIF T=2 vs Design task success gap) to hyperparameter choice.

GRID
----
  beta   in {0.05, 0.20, 0.80}   -- epistemic weight
  lambda in {0.02, 0.06, 0.25}   -- dose penalty weight

The paper values (beta=0.20, lambda=0.06) are at the centre of
the grid.  All 9 conditions are run at n=20 trials each to give
a rapid qualitative picture of sensitivity across the grid.

For the lambda=0.06 column only, targeted n=100 runs are
provided separately in bayes_aif_sensitivity_n100.py.

RUNTIME
-------
Approximately 90 minutes per condition at T=3 on a laptop.
Full grid: ~13 hours.  Run overnight or split across sessions
by commenting out rows of GRID.

OUTPUT
------
One CSV per condition written to figures/:
  sensitivity_b{beta}_l{lambda}_summary.csv

USAGE
-----
Place alongside bayes_aif_v1.py and run:

    python bayes_aif_sensitivity.py

REQUIREMENTS
------------
  bayes_aif_v1.py in the same directory
  Python >= 3.10, numpy >= 1.24
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

SEED       = 2            # same seed as main result for comparability
N_TRIALS   = 20           # reduced for sensitivity screen
RATE_MODEL = "nonlinear"
STRUCTURE  = "clustered"
D0         = 1.7

# Full 3x3 grid.
# To run a subset, comment out rows as needed.
GRID = [
    (0.05, 0.02), (0.05, 0.06), (0.05, 0.25),
    (0.20, 0.02), (0.20, 0.06), (0.20, 0.25),   # (0.20, 0.06) = paper values
    (0.80, 0.02), (0.80, 0.06), (0.80, 0.25),
]


def main() -> None:
    print("\nHYPERPARAMETER SENSITIVITY ANALYSIS  (n=20 screen)")
    print("  rate_model =", RATE_MODEL)
    print("  structure  =", STRUCTURE)
    print(f"  n_trials   = {N_TRIALS} per condition")
    print(f"  grid       = {len(GRID)} conditions (3x3 beta x lambda)")
    print()
    print("  Note: SE on a proportion at n=20 is ~11pp.")
    print("  Results indicate qualitative trends only.")
    print("  See bayes_aif_sensitivity_n100.py for n=100 confirmation")
    print("  of the lambda=0.06 column.")
    print()

    model = make_benchmark_model(
        seed=SEED, rate_model=RATE_MODEL, structure=STRUCTURE
    )

    # Same trials shared across all conditions for a fair comparison
    trials = [
        sample_trial(np.random.default_rng(SEED + 5000 + t), model)
        for t in range(N_TRIALS)
    ]

    for beta_aif, lam in GRID:
        marker = " <-- PAPER VALUES" if (beta_aif == 0.20 and lam == 0.06) else ""
        print(f"\n{'='*50}")
        print(f"beta={beta_aif:.2f}   lambda={lam:.2f}{marker}")
        print(f"{'='*50}")

        POLICY_CACHE.clear()

        ctrls = [
            CtrlSpec("Design T=1",   mode="design",   T=1, beta=0.0,
                     lam=lam, d0=D0, dose_budget=15.0),
            CtrlSpec("Decision T=1", mode="decision", T=1, beta=0.0,
                     lam=lam, d0=D0),
            CtrlSpec("AIF T=1",      mode="aif",      T=1, beta=beta_aif,
                     lam=lam, d0=D0),
            CtrlSpec("AIF T=2",      mode="aif",      T=2, beta=beta_aif,
                     lam=lam, d0=D0),
            CtrlSpec("AIF T=3",      mode="aif",      T=3, beta=beta_aif,
                     lam=lam, d0=D0),
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
            f"sensitivity_b{beta_aif:.2f}_l{lam:.2f}"
        ).replace(".", "p")
        save_summary_csv(results, d0=D0, stem=stem)
        print(f"  Saved: {stem}_summary.csv")

    print("\n\nDONE.")
    print("  Review AIF T=2 task success across the grid.")
    print("  The planning advantage (AIF T=2 vs Design) should hold")
    print("  across the interior of the grid if results are robust.")
    print("  Degradation at extreme beta or lambda values is expected")
    print("  and is discussed in the paper.")


if __name__ == "__main__":
    main()
