"""
bayes_aif_v1.py
===============
Simulation code for:

  "Active Inference as a Normative Framework for Adaptive CT Acquisition"
  Physics in Medicine and Biology, 2025.

This script implements the controlled Poisson CT benchmark described in
Section 4 of the paper. It evaluates five acquisition controllers --
Bayesian design, Bayesian decision, and Active Inference at planning
horizons T=1, 2, 3 -- under nonlinear (Poisson) and linearised
measurement models over 100 independent trials.

OUTPUT
------
Running this script produces three CSV files in figures/:

  {stem}_summary.csv          -- per-controller summary statistics
                                 (reach, dose, d-prime, IQR)
                                 consumed by pmb_three_figures.py
  {stem}_policy_map.csv       -- per-step belief state and exposure
                                 for every trial (reproducibility)
  {stem}_mean_exposure_by_step.csv  -- mean exposure per step per
                                       controller (diagnostic)

The paper figures are generated separately by pmb_three_figures.py,
which reads the summary CSV from each condition (nonlinear/linear).

CONFIGURATION
-------------
Edit the three lines in main() marked CONFIGURE:
  seed        -- random seed (2 for main result, 3+ for sensitivity)
  rate_model  -- "nonlinear" or "linear"
  structure   -- "clustered" (main) or "random" (sensitivity)

The output CSV stem encodes these parameters automatically, e.g.:
  nonlinear_clustered_b0p20_l0p06_summary.csv

NOTATION
--------
The paper uses s for the hidden attenuation state and lambda for the
dose penalty weight.  In the code these are named z (z_true, bel.m)
and lam respectively.  z is used throughout to avoid shadowing Python
built-ins; lam is used because lambda is a reserved keyword in Python.
The correspondence is exact: z <-> s, lam <-> lambda (paper notation).

REQUIREMENTS
------------
  Python >= 3.10
  numpy >= 1.24
  matplotlib >= 3.7

RELATED FILES
-------------
  pmb_three_figures.py        -- generates paper figures from CSVs
  bayes_aif_cross_condition.py -- cross-condition experiment (Section 5.4)

REPOSITORY
----------
  https://github.com/[repo]/aif-ct-acquisition
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


# =============================================================================
# Configuration
# =============================================================================

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

GREY3 = ListedColormap(["#F0F0F0", "#A0A0A0", "#404040"])

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "lines.markersize": 5,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

POLICY_CACHE: dict[
    tuple[tuple[int, ...], int],
    list[tuple[tuple[int, float], ...]],
] = {}


# =============================================================================
# Data classes
# =============================================================================

@dataclass(frozen=True)
class Model:
    """Unified Poisson benchmark model with switchable rate map."""
    n_views: int
    D: int
    G: np.ndarray
    prior_mean: np.ndarray
    prior_cov: np.ndarray
    E_grid: np.ndarray
    rate_model: str
    gain: float
    rate_floor: float

    def g(self, view: int) -> np.ndarray:
        return self.G[view]


@dataclass(frozen=True)
class Trial:
    z_true: np.ndarray
    lesion_present: int


@dataclass
class Belief:
    m: np.ndarray
    S: np.ndarray


@dataclass(frozen=True)
class CtrlSpec:
    name: str
    mode: str        # "design" | "decision" | "aif"
    T: int           # planning horizon
    beta: float      # epistemic weight (AIF only)
    lam: float       # dose penalty weight (lambda in paper; lam here because lambda is a Python keyword)
    d0: float        # detectability threshold
    dose_budget: float | None = None  # hard cap (Design only)


@dataclass(frozen=True)
class RunResult:
    dose: float
    n_used: int
    d_final: float
    var_task_final: float
    logdet_final: float
    views: list[int]
    exposures: list[float]
    d_hist: list[float]
    m1_hist: list[float]
    s11_hist: list[float]


# =============================================================================
# Numerical utilities
# =============================================================================

def safe_logdet(S: np.ndarray) -> float:
    sign, ld = np.linalg.slogdet(S)
    return float(ld) if sign > 0 else -np.inf


def symmetrise(S: np.ndarray) -> np.ndarray:
    return 0.5 * (S + S.T)


def median_iqr(x: np.ndarray) -> tuple[float, float, float]:
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    return float(q50), float(q25), float(q75)


def save_figure(fig: plt.Figure, path_stem: Path) -> None:
    """Save PDF and PNG at 600 dpi."""
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")


# =============================================================================
# Rate model
# =============================================================================

def rate_and_derivatives(
    model: Model,
    g: np.ndarray,
    z: np.ndarray,
    x_clip: float = 30.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Compute rate r(z), gradient dr/dz, and Hessian d²r/dz² for
    the view-specific rate function.

    Nonlinear:  r = exp(gain * g @ z)   -- true Poisson CT model
    Linearised: r = 1 + gain * g @ z    -- first-order approximation
    """
    gg = model.gain * g
    x = float(gg @ z)

    if model.rate_model == "linear":
        raw_r = 1.0 + x
        if raw_r <= model.rate_floor:
            r = model.rate_floor
            grad_r = np.zeros_like(g)
            hess_r = np.zeros((g.size, g.size))
        else:
            r = raw_r
            grad_r = gg
            hess_r = np.zeros((g.size, g.size))
        return float(r), grad_r, hess_r

    if model.rate_model == "nonlinear":
        x_safe = np.clip(x, -x_clip, x_clip)
        r = float(np.exp(x_safe))
        grad_r = r * gg
        hess_r = r * np.outer(gg, gg)
        return r, grad_r, hess_r

    raise ValueError(f"Unknown rate_model: {model.rate_model!r}")


def poisson_rate(
    model: Model,
    g: np.ndarray,
    z: np.ndarray,
    exposure: float,
) -> tuple[float, float]:
    r, _, _ = rate_and_derivatives(model, g, z)
    return r, float(exposure) * r


# =============================================================================
# Benchmark construction
# =============================================================================

def make_benchmark_model(
    seed: int = 0,
    rate_model: str = "nonlinear",
    structure: str = "clustered",
) -> Model:
    """
    Construct the benchmark model.

    structure="clustered": sixteen views in four clusters of four,
        each cluster with a distinct sensitivity profile (see paper
        Section 4.3).  The clustered geometry creates conditions where
        early nuisance-resolving measurements increase the value of
        later lesion-sensitive measurements.

    structure="random": sensitivity vectors drawn i.i.d., providing
        a geometry-independent sensitivity check.
    """
    rng = np.random.default_rng(seed)
    n_views = 16
    K = 8
    D = 1 + K   # 1 lesion + 8 nuisance components

    if structure == "clustered":
        G = np.zeros((n_views, D), dtype=float)

        for v in range(n_views):
            cluster = v // 4
            g = rng.normal(0.0, 0.06, size=D)

            if cluster == 0:   # high lesion sensitivity
                g[0] += 1.30 + 0.06 * rng.normal()
                g[1:4] += 0.08 * rng.normal(size=3)
            elif cluster == 1: # low lesion, high nuisance
                g[0] += 0.15 + 0.03 * rng.normal()
                g[1:5] += rng.normal(1.25, 0.05, size=4)
            elif cluster == 2: # intermediate
                g[0] += 0.70 + 0.05 * rng.normal()
                g[1:5] += rng.normal(0.55, 0.05, size=4)
            else:              # moderate lesion, low nuisance
                g[0] += 0.40 + 0.04 * rng.normal()
                g[5:] += rng.normal(0.20, 0.04, size=D - 5)

            G[v] = g

        # Introduce within-cluster correlation
        for c in range(0, n_views, 4):
            base = G[c].copy()
            for j in range(4):
                G[c + j] = base + rng.normal(0.0, 0.015, size=D)

    elif structure == "random":
        G = rng.normal(0.0, 0.35, size=(n_views, D))
        G[:, 0] += rng.normal(0.8, 0.15, size=n_views)
    else:
        raise ValueError(f"Unknown structure: {structure!r}")

    return Model(
        n_views=n_views,
        D=D,
        G=G,
        prior_mean=np.zeros(D, dtype=float),
        prior_cov=np.diag([1.0] * D),
        E_grid=np.array([0.3, 1.0, 3.0], dtype=float),
        rate_model=rate_model,
        gain=0.5,
        rate_floor=1e-3,
    )


def sample_trial(rng: np.random.Generator, model: Model) -> Trial:
    """
    Draw a single trial from the generative model.

    Lesion-present (50% prevalence): z[0] ~ N(1.0, 0.18^2)
    Lesion-absent:                   z[0] ~ N(0.0, 0.05^2)
    Nuisance:                        z[1:] ~ N(0, 0.55^2 I)
    """
    z = np.zeros(model.D, dtype=float)
    lesion_present = int(rng.random() < 0.5)
    z[0] = rng.normal(1.0, 0.18) if lesion_present else rng.normal(0.0, 0.05)
    z[1:] = rng.normal(0.0, 0.55, size=model.D - 1)
    return Trial(z_true=z, lesion_present=lesion_present)


# =============================================================================
# Measurement model and Laplace update
# =============================================================================

def simulate_measurement(
    rng: np.random.Generator,
    model: Model,
    trial: Trial,
    view: int,
    exposure: float,
) -> int:
    """Draw a Poisson photon count from the true generative model."""
    g = model.g(view)
    _, lam = poisson_rate(model, g, trial.z_true, exposure)
    return int(rng.poisson(lam))


def laplace_update_one(
    model: Model,
    m: np.ndarray,
    S: np.ndarray,
    g: np.ndarray,
    exposure: float,
    y: float,
    newton_steps: int = 8,
    damping: float = 1.0,
    ridge: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Laplace (Gaussian) approximation to the Poisson posterior after
    observing y photons from view g at exposure level e.

    Uses Newton-Raphson iteration on the log-posterior.  For the
    linearised model this is equivalent to a Kalman covariance update.
    """
    D = m.size
    I = np.eye(D)
    P0 = np.linalg.inv(S + ridge * I)
    z = m.copy()

    for _ in range(newton_steps):
        r, grad_r, _ = rate_and_derivatives(model, g, z)
        grad = P0 @ (z - m) + (float(exposure) - float(y) / r) * grad_r
        H = P0 + (float(y) / (r * r)) * np.outer(grad_r, grad_r) + ridge * I

        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]

        z = z - damping * step

        # Enforce rate floor for linearised model
        if model.rate_model == "linear":
            gg = model.gain * g
            if 1.0 + float(gg @ z) < model.rate_floor:
                excess = model.rate_floor - (1.0 + float(gg @ z))
                z = z + (excess / max(float(gg @ gg), 1e-12)) * gg

        if np.linalg.norm(step) < 1e-7:
            break

    r, grad_r, _ = rate_and_derivatives(model, g, z)
    H = P0 + (float(y) / (r * r)) * np.outer(grad_r, grad_r) + ridge * I

    try:
        S_post = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        S_post = np.linalg.pinv(H)

    return z, symmetrise(S_post)


def nominal_laplace_update(
    model: Model,
    m: np.ndarray,
    S: np.ndarray,
    g: np.ndarray,
    exposure: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Deterministic (nominal) belief update for policy rollout.
    Replaces the random observation y with its expected value E[y] = e*r(m).
    Used during policy evaluation, not during actual trial execution.
    """
    r, _, _ = rate_and_derivatives(model, g, m)
    y_bar = float(exposure) * r
    return laplace_update_one(model=model, m=m, S=S, g=g,
                               exposure=exposure, y=y_bar, newton_steps=3)


# =============================================================================
# Information and detectability
# =============================================================================

def info_gain_from_cov(S_prev: np.ndarray, S_next: np.ndarray) -> float:
    """
    KL-based information gain: 0.5 * (log|S_prev| - log|S_next|).
    Equals the reduction in log-determinant of the posterior covariance.
    """
    ld_prev = safe_logdet(S_prev)
    ld_next = safe_logdet(S_next)
    if not np.isfinite(ld_prev) or not np.isfinite(ld_next):
        return 0.0
    return 0.5 * (ld_prev - ld_next)


def dprime_from_cov(m: np.ndarray, S: np.ndarray, idx_task: int = 0) -> float:
    """
    Hotelling observer detectability index d' from Gaussian belief.
    d' = |mu_task| / sqrt(sigma_task^2)
    """
    m_task = float(m[idx_task])
    v_task = max(float(S[idx_task, idx_task]), 1e-12)
    return abs(m_task) / np.sqrt(v_task)


def dprime_from_belief(bel: Belief, idx_task: int = 0) -> float:
    return dprime_from_cov(bel.m, bel.S, idx_task=idx_task)


# =============================================================================
# Policy construction and controller objectives
# =============================================================================

def evaluate_policy(
    model: Model,
    bel: Belief,
    policy: tuple[tuple[int, float], ...],
    spec: CtrlSpec,
) -> float:
    """
    Score a candidate policy under the controller objective.

    Returns the scalar objective L(pi) as defined in the paper
    (equation calL in Algorithm 1):

      Design:   L(pi) = -I_pi                        (epistemic only)
      Decision: L(pi) =  R_pi + lam * C_pi           (pragmatic only)
      AIF:      L(pi) =  R_pi - beta * I_pi + lam * C_pi  (joint)

    where I_pi is cumulative information gain, R_pi is cumulative
    detectability shortfall (calR_pi in paper), and C_pi is cumulative
    dose.  Future observations are replaced by nominal expected values
    (deterministic rollout). Only the first action is executed.
    """
    m_roll, S_roll = bel.m.copy(), bel.S.copy()
    total_info_gain = total_risk = total_cost = 0.0

    for view, exposure in policy:
        g = model.g(view)
        m_next, S_next = nominal_laplace_update(model, m_roll, S_roll,
                                                 g, exposure)
        total_info_gain += info_gain_from_cov(S_roll, S_next)
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


def candidate_policies_exact(
    model: Model,
    remaining_views: Sequence[int],
    spec: CtrlSpec,
) -> list[tuple[tuple[int, float], ...]]:
    """
    Enumerate all candidate policies of length T_eff = min(T, |V|)
    over remaining views V and exposure grid E.
    Results are cached by (remaining_views, T_eff).

    For T=3 this gives up to P(16,3) * 3^3 = 90,720 candidates.
    """
    T_eff = min(spec.T, len(remaining_views))
    if T_eff <= 0:
        return []
    key = (tuple(remaining_views), T_eff)
    if key in POLICY_CACHE:
        return POLICY_CACHE[key]

    policies = [
        tuple((int(v), float(e)) for v, e in zip(view_seq, exp_seq))
        for view_seq in permutations(remaining_views, T_eff)
        for exp_seq in product(model.E_grid, repeat=T_eff)
    ]
    POLICY_CACHE[key] = policies
    return policies


def select_action_exact(
    model: Model,
    bel: Belief,
    remaining_views: Sequence[int],
    spec: CtrlSpec,
    dose_used: float = 0.0,
) -> tuple[int, float] | None:
    """Select the first action of the best-scoring candidate policy."""
    policies = candidate_policies_exact(model, remaining_views, spec)

    if spec.mode == "design" and spec.dose_budget is not None:
        policies = [p for p in policies
                    if dose_used + sum(e for _, e in p)
                    <= spec.dose_budget + 1e-12]
    if not policies:
        return None

    best_score = np.inf
    best_first: tuple[int, float] | None = None

    for policy in policies:
        score = evaluate_policy(model, bel, policy, spec)
        if score < best_score:
            best_score = score
            best_first = policy[0]

    return (int(best_first[0]), float(best_first[1])) if best_first else None


# =============================================================================
# Closed-loop trial execution
# =============================================================================

def run_one(
    rng: np.random.Generator,
    model: Model,
    trial: Trial,
    spec: CtrlSpec,
    max_steps: int | None = None,
) -> RunResult:
    """
    Execute one closed-loop trial.

    At each step:
      1. Evaluate d' from current belief; stop if d' >= d0.
      2. Select action via exact policy enumeration.
      3. Draw a Poisson observation from the true model.
      4. Update belief via Laplace iteration.
    """
    bel = Belief(m=model.prior_mean.copy(), S=model.prior_cov.copy())
    remaining_views = list(range(model.n_views))
    used_views: list[int] = []
    used_exposures: list[float] = []
    d_hist: list[float] = []
    m1_hist: list[float] = []
    s11_hist: list[float] = []
    total_dose = 0.0
    max_steps_eff = (model.n_views if max_steps is None
                     else min(max_steps, model.n_views))

    for _ in range(max_steps_eff):
        d_now = dprime_from_belief(bel)
        if d_now >= spec.d0 or not remaining_views:
            break

        d_hist.append(float(d_now))
        m1_hist.append(float(bel.m[0]))
        s11_hist.append(float(bel.S[0, 0]))

        action = select_action_exact(model, bel, remaining_views, spec,
                                      dose_used=total_dose)
        if action is None:
            break

        view, exposure = action
        y = simulate_measurement(rng, model, trial, view=view,
                                  exposure=exposure)
        m_post, S_post = laplace_update_one(
            model=model, m=bel.m, S=bel.S, g=model.g(view),
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
# Summary, CSV output, and console reporting
# =============================================================================

def summarise_results(results: dict[str, list[RunResult]], d0: float) -> None:
    print("\nOverall summary:")
    for name, rr in results.items():
        dose  = np.array([r.dose for r in rr])
        dfin  = np.array([r.d_final for r in rr])
        vtask = np.array([r.var_task_final for r in rr])
        ldet  = np.array([r.logdet_final for r in rr])
        reach_n = int(np.sum(dfin >= d0))
        n = len(rr)
        dm, dq1, dq3   = median_iqr(dose)
        fm, fq1, fq3   = median_iqr(dfin)
        vm, vq1, vq3   = median_iqr(vtask)
        lm, lq1, lq3   = median_iqr(ldet)
        print(
            f"{name:14s} d'≥d0 {100*reach_n/n:5.1f}% ({reach_n:3d}/{n})   "
            f"Dose {dm:6.2f} [{dq1:.2f}, {dq3:.2f}]   "
            f"Final d' {fm:5.2f} [{fq1:.2f}, {fq3:.2f}]   "
            f"S11 {vm:7.4f} [{vq1:.4f}, {vq3:.4f}]   "
            f"log|S| {lm:7.3f} [{lq1:.3f}, {lq3:.3f}]"
        )


def make_summary_rows(
    results: dict[str, list[RunResult]], d0: float
) -> list[dict]:
    order = ["Design T=1", "Decision T=1", "AIF T=1", "AIF T=2", "AIF T=3"]
    rows = []
    for name in order:
        rr = results[name]
        dose  = np.array([r.dose for r in rr])
        dfin  = np.array([r.d_final for r in rr])
        vtask = np.array([r.var_task_final for r in rr])
        ldet  = np.array([r.logdet_final for r in rr])
        reach_n = int(np.sum(dfin >= d0))
        n = len(rr)
        dm, dq1, dq3 = median_iqr(dose)
        fm, fq1, fq3 = median_iqr(dfin)
        vm, vq1, vq3 = median_iqr(vtask)
        lm, lq1, lq3 = median_iqr(ldet)
        rows.append({
            "Controller": name,
            "dprime_ge_d0_pct": 100 * reach_n / n,
            "dprime_ge_d0_n": reach_n, "N": n,
            "Dose_med": dm, "Dose_q1": dq1, "Dose_q3": dq3,
            "dprime_med": fm, "dprime_q1": fq1, "dprime_q3": fq3,
            "S11_med": vm, "S11_q1": vq1, "S11_q3": vq3,
            "logdet_med": lm, "logdet_q1": lq1, "logdet_q3": lq3,
        })
    return rows


def save_summary_csv(
    results: dict[str, list[RunResult]], d0: float, stem: str
) -> None:
    """Write per-controller summary statistics to CSV."""
    rows = make_summary_rows(results, d0)
    out = FIG_DIR / f"{stem}_summary.csv"
    headers = list(rows[0].keys())
    with out.open("w") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(str(row[h]) for h in headers) + "\n")


def save_policy_map_csv(
    results: dict[str, list[RunResult]], stem: str
) -> None:
    """Write per-step belief state and exposure for every trial."""
    out = FIG_DIR / f"{stem}_policy_map.csv"
    order = ["Design T=1", "Decision T=1", "AIF T=1", "AIF T=2", "AIF T=3"]
    with out.open("w") as f:
        f.write("controller,trial,step,m1,s11,exposure\n")
        for name in order:
            for t, r in enumerate(results[name]):
                for step, (m1, s11, e) in enumerate(
                    zip(r.m1_hist, r.s11_hist, r.exposures)
                ):
                    f.write(f"{name},{t},{step},{m1},{s11},{e}\n")


def save_mean_exposure_csv(
    results: dict[str, list[RunResult]], stem: str
) -> None:
    """Write mean exposure per acquisition step per controller."""
    order = ["Design T=1", "Decision T=1", "AIF T=1", "AIF T=2", "AIF T=3"]
    max_len = max(
        (len(r.exposures) for rr in results.values() for r in rr), default=0
    )
    out = FIG_DIR / f"{stem}_mean_exposure_by_step.csv"
    with out.open("w") as f:
        f.write("controller,step,mean_exposure,n_trials_at_step\n")
        for name in order:
            rr = results[name]
            for k in range(max_len):
                vals = [r.exposures[k] for r in rr if len(r.exposures) > k]
                mean_e = float(np.mean(vals)) if vals else float("nan")
                f.write(f"{name},{k + 1},{mean_e},{len(vals)}\n")


def make_latex_table(
    results: dict[str, list[RunResult]], d0: float, label: str
) -> str:
    """Return a LaTeX table string for the console output."""
    order = ["Design T=1", "Decision T=1", "AIF T=1", "AIF T=2", "AIF T=3"]
    rows = []
    for name in order:
        rr = results[name]
        dose = np.array([r.dose for r in rr])
        dfin = np.array([r.d_final for r in rr])
        vtask = np.array([r.var_task_final for r in rr])
        ldet = np.array([r.logdet_final for r in rr])
        reach_n = int(np.sum(dfin >= d0))
        n = len(rr)
        dm, dq1, dq3 = median_iqr(dose)
        fm, fq1, fq3 = median_iqr(dfin)
        vm, vq1, vq3 = median_iqr(vtask)
        lm, lq1, lq3 = median_iqr(ldet)
        rows.append(
            f"{name} & {100*reach_n/n:.1f} ({reach_n}/{n}) & "
            f"{dm:.2f} [{dq1:.2f}, {dq3:.2f}] & "
            f"{fm:.2f} [{fq1:.2f}, {fq3:.2f}] & "
            f"{vm:.4f} [{vq1:.4f}, {vq3:.4f}] & "
            f"{lm:.3f} [{lq1:.3f}, {lq3:.3f}] \\\\"
        )
    return (
        rf"\begin{{table*}}[t]" "\n"
        rf"\centering" "\n"
        rf"\caption{{Poisson CT benchmark: {label}. Median [IQR]; "
        rf"reach is \% (count/total).}}" "\n"
        rf"\begin{{tabular}}{{lccccc}}" "\n"
        r"\toprule" "\n"
        r"Controller & $d' \geq d_0$ (\%) & Dose & Final $d'$"
        r" & $S_{11}$ & $\log|S|$ \\" "\n"
        r"\midrule" "\n"
        + "\n".join(rows) + "\n"
        r"\bottomrule" "\n"
        rf"\end{{tabular}}" "\n"
        rf"\label{{tab:{label}_benchmark}}" "\n"
        rf"\end{{table*}}"
    )


# =============================================================================
# Supplementary plotting functions (not called in main)
# These are retained for exploratory analysis and reproducibility.
# =============================================================================

def compute_policy_surface(
    model: Model,
    spec: CtrlSpec,
    m1_vals: np.ndarray,
    s11_vals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the first-action exposure surface over a (m1, s11) grid.
    Not called in main() -- T=3 surface takes ~6 hours on a laptop.
    Uncomment the call in main() if needed for supplementary figures.
    """
    remaining_views = list(range(model.n_views))
    exp_to_idx = {float(e): i for i, e in enumerate(model.E_grid)}
    total = len(s11_vals) * len(m1_vals)
    done = 0
    Z = np.full((len(s11_vals), len(m1_vals)), np.nan)

    for i, s11 in enumerate(s11_vals):
        for j, m1 in enumerate(m1_vals):
            m = model.prior_mean.copy()
            S = model.prior_cov.copy()
            m[0] = m1
            S[0, 0] = s11
            bel = Belief(m=m, S=S)
            action = select_action_exact(model, bel, remaining_views,
                                          spec, dose_used=0.0)
            if action is not None:
                Z[i, j] = action[1]
            done += 1
            if done % 100 == 0 or done == total:
                print(f"\r    {spec.name}: surface {done}/{total}",
                      end="", flush=True)
    print()

    Z_idx = np.full_like(Z, np.nan)
    for e, idx in exp_to_idx.items():
        Z_idx[np.isclose(Z, e, equal_nan=False)] = idx
    return Z, Z_idx


def plot_policy_surface_grid(
    model: Model, spec: CtrlSpec, stem: str,
    m1_vals: np.ndarray | None = None,
    s11_vals: np.ndarray | None = None,
    Z: np.ndarray | None = None,
    Z_idx: np.ndarray | None = None,
) -> None:
    """Plot first-action exposure surface. Not called in main()."""
    if m1_vals is None:
        m1_vals = np.linspace(-1.2, 1.2, 41)
    if s11_vals is None:
        s11_vals = np.linspace(0.08, 0.60, 33)
    if Z is None or Z_idx is None:
        Z, Z_idx = compute_policy_surface(model, spec, m1_vals, s11_vals)

    exp_levels = list(model.E_grid)
    exp_to_idx = {float(e): i for i, e in enumerate(exp_levels)}
    Z_idx2 = np.full_like(Z, np.nan, dtype=float)
    for e, idx in exp_to_idx.items():
        Z_idx2[np.isclose(Z, e)] = idx

    fig, ax = plt.subplots(figsize=(5.2, 4.4), constrained_layout=True)
    im = ax.imshow(Z_idx2, origin="lower", aspect="auto",
                   extent=[m1_vals[0], m1_vals[-1],
                           s11_vals[0], s11_vals[-1]],
                   cmap=GREY3, vmin=0, vmax=len(exp_levels) - 1,
                   interpolation="nearest")
    ax.set_xlabel(r"$m_1$")
    ax.set_ylabel(r"$\sigma_{11}$")
    ax.set_title(spec.name)
    cbar = fig.colorbar(im, ax=ax, ticks=range(len(exp_levels)))
    cbar.ax.set_yticklabels([f"{e:g}" for e in exp_levels])
    cbar.set_label("First exposure")
    fname = f"{stem}_{spec.name.replace(' ', '_').replace('=', '')}_policy_surface"
    save_figure(fig, FIG_DIR / fname)
    plt.close(fig)


def save_policy_surface_csv(
    model: Model, spec: CtrlSpec, stem: str,
    m1_vals: np.ndarray | None = None,
    s11_vals: np.ndarray | None = None,
    Z: np.ndarray | None = None,
) -> None:
    """Save policy surface to CSV. Not called in main()."""
    if m1_vals is None:
        m1_vals = np.linspace(-1.2, 1.2, 41)
    if s11_vals is None:
        s11_vals = np.linspace(0.08, 0.60, 33)
    if Z is None:
        Z, _ = compute_policy_surface(model, spec, m1_vals, s11_vals)

    exp_to_idx = {float(e): i for i, e in enumerate(model.E_grid)}
    out = FIG_DIR / (
        f"{stem}_{spec.name.replace(' ', '_').replace('=', '')}"
        "_policy_surface.csv"
    )
    with out.open("w") as f:
        f.write("controller,m1,s11,exposure,exposure_index\n")
        for i, s11 in enumerate(s11_vals):
            for j, m1 in enumerate(m1_vals):
                exposure = Z[i, j]
                if np.isnan(exposure):
                    f.write(f"{spec.name},{m1},{s11},,\n")
                else:
                    idx = exp_to_idx.get(float(exposure), -1)
                    f.write(f"{spec.name},{m1},{s11},{exposure},{idx}\n")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    # ------------------------------------------------------------------
    # CONFIGURE — edit these three lines for each condition
    # ------------------------------------------------------------------
    seed       = 2            # 2=main result; 3,4,5=sensitivity seeds
    rate_model = "nonlinear"  # "nonlinear" or "linear"
    structure  = "clustered"  # "clustered" or "random"
    # ------------------------------------------------------------------

    d0       = 1.7
    n_trials = 400
    beta_aif = 0.20   # beta in paper: epistemic weight
    lam      = 0.06   # lambda in paper: dose penalty weight (lam avoids Python keyword clash)

    model = make_benchmark_model(seed=seed, rate_model=rate_model,
                                  structure=structure)

    trials = [
        sample_trial(np.random.default_rng(seed + 5000 + t), model)
        for t in range(n_trials)
    ]

    print("\n" + "=" * 42)
    print(f"rate_model={rate_model}   structure={structure}"
          f"   beta={beta_aif:.2f}   lambda={lam:.2f}")
    print("=" * 42)

    POLICY_CACHE.clear()

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

    results: dict[str, list[RunResult]] = {c.name: [] for c in ctrls}

    for k, spec in enumerate(ctrls):
        for t, trial in enumerate(trials):
            print(f"\r{spec.name}: trial {t + 1}/{n_trials}", end="",
                  flush=True)
            rng_run = np.random.default_rng(seed + 100000 * (k + 1) + t)
            results[spec.name].append(run_one(rng_run, model, trial, spec))
        print()

    summarise_results(results, d0=d0)
    print("\nLaTeX table:\n")
    print(make_latex_table(results, d0=d0, label=rate_model))

    stem = (f"{rate_model}_{structure}"
            f"_b{beta_aif:.2f}_l{lam:.2f}").replace(".", "p")

    save_summary_csv(results, d0=d0, stem=stem)
    save_policy_map_csv(results, stem=stem)
    save_mean_exposure_csv(results, stem=stem)

    print(f"\nSaved CSVs to: {FIG_DIR}")
    print("Run pmb_three_figures.py to generate paper figures.")

    # ------------------------------------------------------------------
    # Policy surface computation (supplementary material only).
    # Uncomment to generate. T=3 surface takes ~6 hours on a laptop.
    # ------------------------------------------------------------------
    # m1_vals  = np.linspace(-1.2, 1.2, 41)
    # s11_vals = np.linspace(0.08, 0.60, 33)
    # for spec in ctrls:
    #     Z, Z_idx = compute_policy_surface(model, spec, m1_vals, s11_vals)
    #     plot_policy_surface_grid(model, spec, stem=stem,
    #                              m1_vals=m1_vals, s11_vals=s11_vals,
    #                              Z=Z, Z_idx=Z_idx)
    #     save_policy_surface_csv(model, spec, stem=stem,
    #                             m1_vals=m1_vals, s11_vals=s11_vals,
    #                             Z=Z)


if __name__ == "__main__":
    main()