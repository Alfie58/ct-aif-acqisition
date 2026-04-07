"""
pmb_figures.py
==============
Standalone figure generation for the Active Inference CT paper (PMB).
Reads CSV files produced by the main simulation script.
Produces four publication figures as PDF + PNG at 600 dpi.

Usage
-----
    python pmb_figures.py

All paths and display options are set in the CONFIG block below.
No other file needs to be modified.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import warnings

# =============================================================================
# CONFIG  —  edit this block to match your file locations and preferences
# =============================================================================

# Directory where CSVs live (produced by the main simulation script)
CSV_DIR = Path("figures")

# Output directory for figures
OUT_DIR = Path("figures")

# --- CSV stems (the part before _summary.csv / _policy_map.csv etc.) ---
# Set STEM_LINEAR = None if you haven't run the linear model yet.
STEM_NONLINEAR = "nonlinear_clustered_b0p20_l0p06"
STEM_LINEAR    = None   # e.g. "linear_clustered_b0p20_l0p06"

# --- Figure dimensions (inches) ---
# PMB single column ≈ 3.39 in, double column ≈ 6.93 in
# These are easy to rescale — change WIDTH_* and heights follow automatically.
WIDTH_SINGLE = 3.39
WIDTH_DOUBLE = 6.93

# --- Font sizes ---
FONT_BASE   = 8    # body / tick labels
FONT_TITLE  = 9    # panel titles
FONT_LEGEND = 7.5  # legend entries

# --- Controllers and display order ---
CTRL_ORDER = ["Design T=1", "Decision T=1", "AIF T=1", "AIF T=2", "AIF T=3"]

# Greyscale fills for bar charts (one per controller, light → dark)
BAR_GREYS = ["0.95", "0.65", "0.45", "0.25", "0.05"]

# Exposure levels used in the simulation
E_LEVELS = [0.3, 1.0, 3.0]

# Greyscale for exposure levels in policy surface and trajectory figures:
#   light = low dose, dark = high dose
E_GREY   = ["0.92", "0.60", "0.20"]   # backgrounds
E_EDGE   = ["0.50", "0.30", "0.00"]   # marker edges for trajectories

# Line styles for mean-exposure-by-step plot (one per controller)
LINE_STYLES = {
    "Design T=1":   dict(color="black",   linestyle="-",  linewidth=1.2, marker="o", markersize=3.5),
    "Decision T=1": dict(color="black",   linestyle="--", linewidth=1.2, marker="s", markersize=3.5),
    "AIF T=1":      dict(color="0.45",    linestyle="-.", linewidth=1.2, marker="^", markersize=3.5),
    "AIF T=2":      dict(color="0.45",    linestyle="-",  linewidth=1.2, marker="D", markersize=3.5),
    "AIF T=3":      dict(color="0.45",    linestyle=":",  linewidth=1.2, marker="v", markersize=3.5),
}

# d0 threshold (must match what was used in the simulation)
D0 = 1.7

# =============================================================================
# Matplotlib global settings
# =============================================================================

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          FONT_BASE,
    "axes.titlesize":     FONT_TITLE,
    "axes.labelsize":     FONT_BASE,
    "legend.fontsize":    FONT_LEGEND,
    "xtick.labelsize":    FONT_BASE,
    "ytick.labelsize":    FONT_BASE,
    "axes.linewidth":     0.6,
    "lines.linewidth":    1.2,
    "lines.markersize":   3.5,
    "savefig.dpi":        600,
    "pdf.fonttype":       42,   # embed fonts for vector PDF
    "ps.fonttype":        42,
})

# =============================================================================
# Helpers
# =============================================================================

def out_path(stem: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR / stem


def save_fig(fig: plt.Figure, stem: str) -> None:
    p = out_path(stem)
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), dpi=600, bbox_inches="tight")
    print(f"  saved  {p.with_suffix('.pdf')}  +  .png")


def load_csv(path: Path) -> list[dict[str, str]] | None:
    """Load a CSV as a list of dicts. Returns None with a warning if missing."""
    if not path.exists():
        warnings.warn(f"CSV not found — skipping: {path}", stacklevel=3)
        return None
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as f:
        headers = f.readline().strip().split(",")
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(dict(zip(headers, line.split(","))))
    return rows


def load_summary(stem: str) -> list[dict] | None:
    return load_csv(CSV_DIR / f"{stem}_summary.csv")


def load_policy_map(stem: str) -> list[dict] | None:
    return load_csv(CSV_DIR / f"{stem}_policy_map.csv")


def load_policy_surface(stem: str, ctrl_name: str) -> list[dict] | None:
    safe = ctrl_name.replace(" ", "_").replace("=", "")
    return load_csv(CSV_DIR / f"{stem}_{safe}_policy_surface.csv")


def load_mean_exposure(stem: str) -> list[dict] | None:
    return load_csv(CSV_DIR / f"{stem}_mean_exposure_by_step.csv")


def summary_to_arrays(rows: list[dict], field: str) -> dict[str, float]:
    """Extract a single numeric field keyed by controller name."""
    return {r["Controller"]: float(r[field]) for r in rows}


def summary_to_iqr(rows: list[dict], med_field: str, q1_field: str, q3_field: str
                   ) -> dict[str, tuple[float, float, float]]:
    return {
        r["Controller"]: (float(r[med_field]), float(r[q1_field]), float(r[q3_field]))
        for r in rows
    }


# =============================================================================
# Figure 1 — Policy surfaces (2×2): Design, AIF T=1, AIF T=2, AIF T=3
# =============================================================================

def fig_policy_surfaces(stem: str, label: str = "") -> None:
    """
    2×2 grid of policy surface heatmaps.
    Panels: Design T=1, AIF T=1, AIF T=2, AIF T=3.
    Single shared greyscale colourbar on the right.
    """
    print("Figure 1 — policy surfaces")

    panel_ctrls = ["Design T=1", "AIF T=1", "AIF T=2", "AIF T=3"]
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    # Load surface data for each controller
    surfaces: dict[str, np.ndarray | None] = {}
    m1_vals_ref: np.ndarray | None = None
    s11_vals_ref: np.ndarray | None = None

    for ctrl in panel_ctrls:
        rows = load_policy_surface(stem, ctrl)
        if rows is None:
            surfaces[ctrl] = None
            continue

        m1_all  = sorted(set(float(r["m1"])  for r in rows))
        s11_all = sorted(set(float(r["s11"]) for r in rows))
        m1_arr  = np.array(m1_all)
        s11_arr = np.array(s11_all)

        if m1_vals_ref is None:
            m1_vals_ref  = m1_arr
            s11_vals_ref = s11_arr

        Z = np.full((len(s11_arr), len(m1_arr)), np.nan)
        m1_idx  = {v: i for i, v in enumerate(m1_all)}
        s11_idx = {v: i for i, v in enumerate(s11_all)}

        for r in rows:
            ei = r["exposure_index"].strip()
            if ei == "":
                continue
            i = s11_idx[float(r["s11"])]
            j = m1_idx[float(r["m1"])]
            Z[i, j] = float(ei)

        surfaces[ctrl] = Z

    # Build figure
    fig, axes = plt.subplots(
        2, 2,
        figsize=(WIDTH_DOUBLE, WIDTH_DOUBLE * 0.82),
        constrained_layout=True,
    )

    # Greyscale colourmap: light = low exposure, dark = high exposure
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(E_GREY)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    im_ref = None
    for ax, ctrl, plabel in zip(axes.ravel(), panel_ctrls, panel_labels):
        Z = surfaces.get(ctrl)
        if Z is None or m1_vals_ref is None or s11_vals_ref is None:
            ax.text(0.5, 0.5, "data missing", transform=ax.transAxes,
                    ha="center", va="center", color="0.5")
            ax.set_title(f"{plabel} {ctrl}")
            continue

        extent = [m1_vals_ref[0], m1_vals_ref[-1],
                  s11_vals_ref[0], s11_vals_ref[-1]]
        im = ax.imshow(
            Z, origin="lower", aspect="auto",
            extent=extent, cmap=cmap, norm=norm,
            interpolation="nearest",
        )
        im_ref = im

        ax.set_title(f"{plabel} {ctrl}")
        ax.set_xlabel(r"$m_1$")
        ax.set_ylabel(r"$\sigma_{11}$")
        ax.tick_params(direction="in", length=2)

    # Single shared colourbar
    if im_ref is not None:
        cbar = fig.colorbar(
            im_ref, ax=axes, ticks=[0, 1, 2],
            shrink=0.6, pad=0.02, aspect=20,
        )
        cbar.ax.set_yticklabels([f"{e:g}" for e in E_LEVELS])
        cbar.set_label("First exposure", labelpad=4)

    tag = f"_{label}" if label else ""
    save_fig(fig, f"fig1_policy_surfaces{tag}")
    plt.close(fig)


# =============================================================================
# Figure 2 — Belief-space trajectory plot (2×2)
# Decision T=1, AIF T=1, AIF T=2, AIF T=3
# Background: exposure decision regions from policy surface
# Overlay: individual trial trajectories as thin lines
# =============================================================================

def fig_trajectories(stem: str, label: str = "") -> None:
    """
    2×2 panel showing how each controller navigates belief space.
    Background shading = exposure the controller would choose at each (m1, σ11).
    Overlaid lines = actual trial trajectories (one line per trial).
    """
    print("Figure 2 — belief-space trajectories")

    panel_ctrls = ["Decision T=1", "AIF T=1", "AIF T=2", "AIF T=3"]
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(E_GREY)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    # Load policy map data — group by (controller, trial)
    map_rows = load_policy_map(stem)
    trajectories: dict[str, dict[int, tuple[list, list]]] = {c: {} for c in panel_ctrls}

    if map_rows is not None:
        for r in map_rows:
            ctrl = r["controller"]
            if ctrl not in trajectories:
                continue
            trial = int(r["trial"])
            m1    = float(r["m1"])
            s11   = float(r["s11"])
            if trial not in trajectories[ctrl]:
                trajectories[ctrl][trial] = ([], [])
            trajectories[ctrl][trial][0].append(m1)
            trajectories[ctrl][trial][1].append(s11)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(WIDTH_DOUBLE, WIDTH_DOUBLE * 0.82),
        constrained_layout=True,
    )

    im_ref = None
    for ax, ctrl, plabel in zip(axes.ravel(), panel_ctrls, panel_labels):

        # --- background: policy surface ---
        surf_rows = load_policy_surface(stem, ctrl)
        if surf_rows is not None:
            m1_all  = sorted(set(float(r["m1"])  for r in surf_rows))
            s11_all = sorted(set(float(r["s11"]) for r in surf_rows))
            Z = np.full((len(s11_all), len(m1_all)), np.nan)
            m1_idx  = {v: i for i, v in enumerate(m1_all)}
            s11_idx = {v: i for i, v in enumerate(s11_all)}
            for r in surf_rows:
                ei = r["exposure_index"].strip()
                if ei == "":
                    continue
                Z[s11_idx[float(r["s11"])], m1_idx[float(r["m1"])]] = float(ei)

            extent = [m1_all[0], m1_all[-1], s11_all[0], s11_all[-1]]
            im = ax.imshow(
                Z, origin="lower", aspect="auto",
                extent=extent, cmap=cmap, norm=norm,
                interpolation="nearest", alpha=0.45,
            )
            im_ref = im

        # --- overlay: trial        # --- overlay: trial trajectories ---
        trial_data = trajectories.get(ctrl, {})
        for trial_id, (m1_seq, s11_seq) in trial_data.items():
            ax.plot(
                m1_seq, s11_seq,
                color="black", linewidth=0.4, alpha=0.45,
                solid_capstyle="round",
            )
            if m1_seq:
                ax.plot(m1_seq[0], s11_seq[0], "o",
                        color="black", markersize=1.8, alpha=0.7, zorder=3)

        ax.set_title(f"{plabel} {ctrl}")
        ax.set_xlabel(r"$m_1$")
        ax.set_ylabel(r"$\sigma_{11}$")
        ax.tick_params(direction="in", length=2)

    if im_ref is not None:
        cbar = fig.colorbar(
            im_ref, ax=axes, ticks=[0, 1, 2],
            shrink=0.6, pad=0.02, aspect=20,
        )
        cbar.ax.set_yticklabels([f"{e:g}" for e in E_LEVELS])
        cbar.set_label("Chosen exposure (background)", labelpad=4)

    traj_handle  = mlines.Line2D([], [], color="black", linewidth=0.8, alpha=0.6,
                                 label="Trial trajectory")
    start_handle = mlines.Line2D([], [], linestyle="", marker="o",
                                 color="black", markersize=3, alpha=0.7,
                                 label="Trial start")
    axes[0, 0].legend(handles=[traj_handle, start_handle],
                      fontsize=FONT_LEGEND, frameon=False, loc="upper right")

    tag = f"_{label}" if label else ""
    save_fig(fig, f"fig2_trajectories{tag}")
    plt.close(fig)


# =============================================================================
# Figure 3 — Mean exposure by acquisition step
# =============================================================================

def fig_mean_exposure(stem_nl, stem_lin, label=""):
    print("Figure 3 — mean exposure by step")

    datasets = []
    if stem_nl  is not None: datasets.append(("Nonlinear (Poisson)", load_mean_exposure(stem_nl)))
    if stem_lin is not None: datasets.append(("Linear",              load_mean_exposure(stem_lin)))
    available = [(lbl, rows) for lbl, rows in datasets if rows is not None]

    if not available:
        warnings.warn("No mean-exposure CSVs found — skipping Figure 3.")
        return

    n_panels = len(available)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(WIDTH_DOUBLE if n_panels == 2 else WIDTH_SINGLE * 1.5,
                 WIDTH_SINGLE * 1.0),
        constrained_layout=True,
        sharey=(n_panels > 1),
    )
    if n_panels == 1:
        axes = [axes]

    for ax, (panel_label, rows) in zip(axes, available):
        by_ctrl = {}
        for r in rows:
            ctrl   = r["controller"]
            step   = int(r["step"])
            mean_e = r["mean_exposure"].strip()
            if mean_e in ("", "nan"):
                continue
            if ctrl not in by_ctrl:
                by_ctrl[ctrl] = {}
            by_ctrl[ctrl][step] = float(mean_e)

        for ctrl in CTRL_ORDER:
            if ctrl not in by_ctrl:
                continue
            steps = sorted(by_ctrl[ctrl])
            means = [by_ctrl[ctrl][s] for s in steps]
            ax.plot(steps, means, label=ctrl, **LINE_STYLES.get(ctrl, {}))

        ax.set_xlabel("Acquisition step")
        if ax is axes[0]:
            ax.set_ylabel("Mean chosen exposure")
        ax.set_title(panel_label)
        ax.tick_params(direction="in", length=2)
        if by_ctrl:
            max_step = max(max(by_ctrl[c]) for c in by_ctrl if by_ctrl[c])
            ax.set_xticks(range(1, max_step + 1, 2))

    axes[0].legend(frameon=False, loc="upper right", fontsize=FONT_LEGEND)

    tag = f"_{label}" if label else ""
    save_fig(fig, f"fig3_mean_exposure{tag}")
    plt.close(fig)


# =============================================================================
# Figure 4 — Benchmark summary: reach & dose, linear vs nonlinear
# =============================================================================

def fig_benchmark_summary(stem_nl, stem_lin, label=""):
    print("Figure 4 — benchmark summary")

    datasets = []
    if stem_nl  is not None: datasets.append(("Nonlinear", load_summary(stem_nl)))
    if stem_lin is not None: datasets.append(("Linear",    load_summary(stem_lin)))
    available = [(lbl, rows) for lbl, rows in datasets if rows is not None]

    if not available:
        warnings.warn("No summary CSVs found — skipping Figure 4.")
        return

    n_models   = len(available)
    ctrl_labels = CTRL_ORDER
    n_ctrl     = len(ctrl_labels)
    x          = np.arange(n_ctrl)

    total_width = 0.65
    bar_w   = total_width / n_models
    offsets = np.linspace(-total_width / 2 + bar_w / 2,
                           total_width / 2 - bar_w / 2, n_models)

    model_style = [
        dict(facecolor="0.20", edgecolor="black", hatch="",    linewidth=0.5),
        dict(facecolor="0.80", edgecolor="black", hatch="///", linewidth=0.5),
    ]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(WIDTH_DOUBLE, WIDTH_SINGLE * 1.05),
        constrained_layout=True,
    )

    # Panel A — reach
    ax = axes[0]
    for mi, (model_lbl, rows) in enumerate(available):
        reach = [float(next(r for r in rows if r["Controller"] == c)["dprime_ge_d0_pct"])
                 for c in ctrl_labels]
        ax.bar(x + offsets[mi], reach, width=bar_w, label=model_lbl, **model_style[mi])

    ax.axhline(50, linestyle=":", color="black", linewidth=0.7, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(ctrl_labels, rotation=30, ha="right")
    ax.set_ylabel(r"Reach: $d' \geq d_0$  (%)")
    ax.set_title("(a) Task success rate")
    ax.set_ylim(0, 100)
    ax.tick_params(direction="in", length=2)
    ax.legend(frameon=False, fontsize=FONT_LEGEND)

    # Panel B — median dose with IQR
    ax = axes[1]
    for mi, (model_lbl, rows) in enumerate(available):
        meds, q1s, q3s = [], [], []
        for c in ctrl_labels:
            row = next(r for r in rows if r["Controller"] == c)
            meds.append(float(row["Dose_med"]))
            q1s.append(float(row["Dose_q1"]))
            q3s.append(float(row["Dose_q3"]))
        meds   = np.array(meds)
        yerr_lo = meds - np.array(q1s)
        yerr_hi = np.array(q3s) - meds
        ax.bar(x + offsets[mi], meds, width=bar_w, label=model_lbl, **model_style[mi])
        ax.errorbar(x + offsets[mi], meds, yerr=[yerr_lo, yerr_hi],
                    fmt="none", color="black", linewidth=0.8, capsize=2)

    ax.set_xticks(x)
    ax.set_xticklabels(ctrl_labels, rotation=30, ha="right")
    ax.set_ylabel("Median dose  [IQR]")
    ax.set_title("(b) Radiation dose")
    ax.tick_params(direction="in", length=2)
    ax.legend(frameon=False, fontsize=FONT_LEGEND)

    tag = f"_{label}" if label else ""
    save_fig(fig, f"fig4_benchmark_summary{tag}")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"\nReading CSVs from : {CSV_DIR.resolve()}")
    print(f"Writing figures to: {OUT_DIR.resolve()}\n")

    have_nl  = STEM_NONLINEAR is not None
    have_lin = STEM_LINEAR    is not None

    if not have_nl and not have_lin:
        raise RuntimeError("Set at least one of STEM_NONLINEAR or STEM_LINEAR in CONFIG.")

    if have_nl:  fig_policy_surfaces(STEM_NONLINEAR, label="nonlinear")
    if have_lin: fig_policy_surfaces(STEM_LINEAR,    label="linear")

    if have_nl:  fig_trajectories(STEM_NONLINEAR, label="nonlinear")
    if have_lin: fig_trajectories(STEM_LINEAR,    label="linear")

    fig_mean_exposure(
        stem_nl  = STEM_NONLINEAR if have_nl  else None,
        stem_lin = STEM_LINEAR    if have_lin else None,
    )

    fig_benchmark_summary(
        stem_nl  = STEM_NONLINEAR if have_nl  else None,
        stem_lin = STEM_LINEAR    if have_lin else None,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()