"""
bayes_aif_cross_condition.py
============================
Cross-condition experiment for:

  "Active Inference as a Normative Framework for Adaptive CT Acquisition"
  Physics in Medicine and Biology, 2025.

PURPOSE
-------
The main simulation (bayes_aif_v1.py) compares two matched conditions:

  Condition A: nonlinear (Poisson) data  +  nonlinear controller
  Condition B: linearised data           +  linearised controller

These conditions differ in two ways simultaneously: the measurement
statistics of the data, and the likelihood model used by the controller.
The ~19 percentage-point performance gap between A and B could therefore
reflect either (1) richer Poisson measurement statistics enabling more
informative observations, or (2) the nonlinear controller having the
correct likelihood model, or both.

This script adds a diagnostic third condition:

  Condition C: nonlinear (Poisson) data  +  linearised controller

Condition C holds the data statistics fixed at Poisson while using a
misspecified controller. By comparing A, B, and C, the total gap can
be decomposed into:

  Measurement model effect  = C - B  (Poisson data, same misspecified ctrl)
  Inference quality effect  = A - C  (correct ctrl, same Poisson data)

Results are reported in Section 5.4 of the paper.

IMPLEMENTATION
--------------
The key change is separating the data-generating model from the belief
model. A new function run_one_cross() accepts both:

  true_model   -- used for measurement generation (true physics)
  belief_model -- used for Laplace updates and policy evaluation

Setting true_model == belief_model recovers the original run_one().
Setting true_model=nonlinear, belief_model=linear gives Condition C.

All other machinery (trial sampling, d-prime computation, CSV output)
is imported unchanged from bayes_aif_v1.py.

USAGE
-----
Place this file in the same directory as bayes_aif_v1.py and run:

    python bayes_aif_cross_condition.py

All three conditions run sequentially. Conditions A and B should match
the results from bayes_aif_v1.py (same seed, same trials) -- this
serves as a validation check before trusting Condition C.

Runtime: approximately 12--18 hours total on a laptop (three full runs
of 100 trials each).

Output CSVs (written to figures/):
  cross_A_nonlinear_data_nonlinear_ctrl_b0p20_l0p06_summary.csv
  cross_B_linear_data_linear_ctrl_b0p20_l0p06_summary.csv
  cross_C_nonlinear_data_linear_ctrl_b0p20_l0p06_summary.csv

REQUIREMENTS
------------
  Python >= 3.10
  numpy >= 1.24
  bayes_aif_v1.py (must be in the same directory)
"""

from __future__ import annotations

import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Import all machinery from bayes_aif_v1.py.
# This script adds only the cross-condition logic; no duplication.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from bayes_aif_v1 import (
    Belief,
    CtrlSpec,
    Model,
    RunResult,
    Trial,
    POLICY_CACHE,
    candidate_policies_exact,
    dprime_from_belief,
    dprime_from_cov,
    laplace_update_one,
    make_benchmark_model,
    nominal_laplace_update,
    safe_logdet,
    sample_trial,
    save_summary_csv,
    simulate_measurement,
    summarise_results,
)


# =============================================================================
# Cross-condition policy evaluation
# =============================================================================

def evaluate_policy_cross(
    belief_model: Model,
    bel: Belief,
    policy: tuple[tuple[int, float], ...],
    spec: CtrlSpec,
) -> float:
    """
    Score a candidate policy using belief_model for nominal rollouts.

    Identical to evaluate_policy() in bayes_aif_v1.py except that
    belief_model (not the true data-generating model) is used for
    the Laplace update and rate computation during rollout.  This
    allows the controller to plan under a misspecified likelihood.

    Parameters
    ----------
    belief_model : Model
        The model whose rate function is used for belief updates and
        policy evaluation.  May differ from the true data-generating
        model (cross-condition C).
    bel : Belief
        Current posterior belief state.
    policy : sequence of (view, exposure) pairs
        Candidate action sequence of length T_eff.
    spec : CtrlSpec
        Controller specification (mode, beta, lam, d0).

    Returns
    -------
    float
        Objective value (lower is better for all modes).
    """
    m_roll, S_roll = bel.m.copy(), bel.S.copy()
    total_info_gain = total_risk = total_cost = 0.0

    for view, exposure in policy:
        g = belief_model.g(view)
        m_next, S_next = nominal_laplace_update(
            model=belief_model, m=m_roll, S=S_roll,
            g=g, exposure=exposure,
        )
        # Information gain: 0.5 * (log|S_prev| - log|S_next|)
        total_info_gain += 0.5 * (
            float(np.linalg.slogdet(S_roll)[1])
            - float(np.linalg.slogdet(S_next)[1])
        )
        total_risk += max(0.0, spec.d0 - dprime_from_cov(m_next, S_next))
        total_cost += float(exposure)
        m_roll, S_roll = m_next, S_next

    if spec.mode == "design":
        return -total_info_gain
    if spec.mode == "decision":
        return total_risk + spec.lam * total_cost
    if spec.mode == "aif":
        return total_risk - spec.beta * total_info_gain + spec.lam * total_cost
    raise ValueError(f"Unknown mode: {spec.mode!r}")


def select_action_cross(
    true_model: Model,
    belief_model: Model,
    bel: Belief,
    remaining_views: list[int],
    spec: CtrlSpec,
    dose_used: float = 0.0,
) -> tuple[int, float] | None:
    """
    Select the first action of the best-scoring policy using belief_model
    for evaluation over the action space defined by true_model.

    The candidate action space (views and exposures) is always drawn from
    true_model so that all conditions share the same discrete action set.
    Policy scoring uses belief_model, which may be misspecified relative
    to the true data-generating process.

    Parameters
    ----------
    true_model : Model
        Data-generating model; defines the candidate action space.
    belief_model : Model
        Model used for belief updates and policy scoring.
    """
    policies = candidate_policies_exact(
        model=true_model,
        remaining_views=remaining_views,
        spec=spec,
    )

    if spec.mode == "design" and spec.dose_budget is not None:
        policies = [
            p for p in policies
            if dose_used + sum(e for _, e in p) <= spec.dose_budget + 1e-12
        ]

    if not policies:
        return None

    best_score = np.inf
    best_first: tuple[int, float] | None = None

    for policy in policies:
        score = evaluate_policy_cross(belief_model, bel, policy, spec)
        if score < best_score:
            best_score = score
            best_first = policy[0]

    return best_first


def run_one_cross(
    rng: np.random.Generator,
    true_model: Model,
    belief_model: Model,
    trial: Trial,
    spec: CtrlSpec,
    max_steps: int | None = None,
) -> RunResult:
    """
    Execute one closed-loop trial with separate data and belief models.

    At each step:
      1. Evaluate d' from current belief; stop if d' >= d0.
      2. Select action via exact policy enumeration using belief_model.
      3. Draw a Poisson observation from true_model (true physics).
      4. Update belief using belief_model's rate function (Laplace).

    Setting belief_model == true_model reproduces run_one() exactly.
    Setting true_model=nonlinear, belief_model=linear gives Condition C:
    the controller plans under the linearised likelihood while receiving
    genuine Poisson observations.

    Parameters
    ----------
    true_model : Model
        Data-generating model. Measurements are always Poisson draws
        from this model regardless of the controller's internal model.
    belief_model : Model
        Model used for belief updates and policy evaluation. For matched
        conditions (A, B) this equals true_model. For Condition C it is
        the linearised model.
    """
    bel = Belief(m=true_model.prior_mean.copy(), S=true_model.prior_cov.copy())
    remaining_views = list(range(true_model.n_views))
    used_views: list[int] = []
    used_exposures: list[float] = []
    d_hist: list[float] = []
    m1_hist: list[float] = []
    s11_hist: list[float] = []
    total_dose = 0.0
    max_steps_eff = (true_model.n_views if max_steps is None
                     else min(max_steps, true_model.n_views))

    for _ in range(max_steps_eff):
        d_now = dprime_from_belief(bel)
        if d_now >= spec.d0 or not remaining_views:
            break

        d_hist.append(float(d_now))
        m1_hist.append(float(bel.m[0]))
        s11_hist.append(float(bel.S[0, 0]))

        action = select_action_cross(
            true_model=true_model,
            belief_model=belief_model,
            bel=bel,
            remaining_views=remaining_views,
            spec=spec,
            dose_used=total_dose,
        )
        if action is None:
            break

        view, exposure = action

        # Observation from true physics
        y = simulate_measurement(rng, true_model, trial,
                                  view=view, exposure=exposure)

        # Belief update under (possibly misspecified) belief_model
        m_post, S_post = laplace_update_one(
            model=belief_model,
            m=bel.m, S=bel.S,
            g=belief_model.g(view),
            exposure=exposure, y=y,
        )
        bel = Belief(m=m_post, S=S_post)

        used_views.append(view)
        used_exposures.append(float(exposure))
        total_dose += float(exposure)
        remaining_views.remove(view)

    return RunResult(
        dose=total_dose, n_used=len(used_views),
        d_final=dprime_from_belief(bel),
        var_task_final=float(bel.S[0, 0]),
        logdet_final=safe_logdet(bel.S),
        views=used_views, exposures=used_exposures,
        d_hist=d_hist, m1_hist=m1_hist, s11_hist=s11_hist,
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    # ------------------------------------------------------------------
    # Parameters — keep identical to bayes_aif_v1.py main condition
    # so that Conditions A and B reproduce those results exactly.
    # ------------------------------------------------------------------
    seed      = 2
    d0        = 1.7
    n_trials  = 100
    beta_aif  = 0.20
    lam       = 0.06
    structure = "clustered"

    print("\nCROSS-CONDITION EXPERIMENT")
    print("=" * 50)
    print("  A: nonlinear data  +  nonlinear controller  (matched)")
    print("  B: linearised data +  linearised controller (matched)")
    print("  C: nonlinear data  +  linearised controller (diagnostic)")
    print()
    print("Decomposition of the A-B performance gap:")
    print("  Measurement model effect = C - B")
    print("  Inference quality effect = A - C")
    print("=" * 50)

    # Build both models with the same seed so they share the same
    # view geometry (G matrix). Only the rate function differs.
    nl_model  = make_benchmark_model(seed=seed, rate_model="nonlinear",
                                      structure=structure)
    lin_model = make_benchmark_model(seed=seed, rate_model="linear",
                                      structure=structure)

    # Trials drawn from nl_model's prior — identical across conditions
    trials = [
        sample_trial(np.random.default_rng(seed + 5000 + t), nl_model)
        for t in range(n_trials)
    ]

    ctrls = [
        CtrlSpec("Design T=1",   mode="design",   T=1, beta=0.0,
                 lam=lam, d0=d0, dose_budget=15.0),
        CtrlSpec("Decision T=1", mode="decision", T=1, beta=0.0,
                 lam=lam, d0=d0),
        CtrlSpec("AIF T=1",      mode="aif",      T=1, beta=beta_aif,
                 lam=lam, d0=d0),
        CtrlSpec("AIF T=2",      mode="aif",      T=2, beta=beta_aif,
                 lam=lam, d0=d0),
        CtrlSpec("AIF T=3",      mode="aif",      T=3, beta=beta_aif,
                 lam=lam, d0=d0),
    ]

    conditions = [
        # (label, true_model, belief_model)
        ("A_nonlinear_data_nonlinear_ctrl", nl_model,  nl_model),
        ("B_linear_data_linear_ctrl",       lin_model, lin_model),
        ("C_nonlinear_data_linear_ctrl",    nl_model,  lin_model),
    ]

    for cond_name, true_model, belief_model in conditions:
        print(f"\n{'=' * 50}")
        print(f"Condition {cond_name}")
        print(f"  Data from   : {true_model.rate_model} model")
        print(f"  Controller  : {belief_model.rate_model} model")
        print(f"{'=' * 50}")

        POLICY_CACHE.clear()
        results: dict[str, list[RunResult]] = {s.name: [] for s in ctrls}

        for k, spec in enumerate(ctrls):
            for t, trial in enumerate(trials):
                print(f"\r  {spec.name}: trial {t + 1}/{n_trials}",
                      end="", flush=True)
                rng_run = np.random.default_rng(
                    seed + 100000 * (k + 1) + t
                )
                results[spec.name].append(
                    run_one_cross(rng_run, true_model, belief_model,
                                  trial, spec)
                )
            print()

        summarise_results(results, d0=d0)

        stem = (
            f"cross_{cond_name}_b{beta_aif:.2f}_l{lam:.2f}"
        ).replace(".", "p")
        save_summary_csv(results, d0=d0, stem=stem)
        print(f"  Saved: {stem}_summary.csv")

    print("\n" + "=" * 50)
    print("INTERPRETATION")
    print("=" * 50)
    print("  C - B = measurement model effect")
    print("    (Poisson data vs linearised data, same misspecified ctrl)")
    print("  A - C = inference quality effect")
    print("    (correct ctrl vs misspecified ctrl, same Poisson data)")
    print()
    print("  If C ~ B: gap is mostly inference quality (model spec)")
    print("  If C ~ A: gap is mostly measurement model (Poisson stats)")
    print("  In practice both contribute; C quantifies the split.")


if __name__ == "__main__":
    main()