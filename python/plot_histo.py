#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot posterior histograms from MCMC trace files produced by optimize_kinematic.py.

Usage
-----
    python3 fold/python/plot_histo.py <traces_dir> [output_dir]

    traces_dir  : directory containing the .txt trace files
    output_dir  : where to save the PDF (defaults to traces_dir)

Each .txt file must contain one posterior sample per line (as written by save_traces()).
The HDI (95% highest density interval) is computed with arviz.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import arviz as az


# ======================================================================================================================
# HDI + histogram helper
# ======================================================================================================================

def plot_histo(samples, label, ax, hdi_prob=0.95, bins=30, color='steelblue'):
    """
    Plot a posterior histogram with mean and HDI on ax.

    Parameters
    ----------
    samples  : 1-D array of posterior draws
    label    : parameter name shown in legend and title
    ax       : matplotlib Axes
    hdi_prob : credible interval probability (default 0.95)
    bins     : number of histogram bins
    color    : fill color
    """
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        ax.set_title(f"{label}\n(no data)")
        return

    ax.hist(samples, bins=bins, density=True, histtype='stepfilled',
            alpha=0.5, color=color, label=label)

    mean = samples.mean()
    hdi  = az.hdi(samples, hdi_prob=hdi_prob)

    ax.axvline(mean,    color='crimson',  linewidth=1.5, label=f"Mean: {mean:.3g}")
    ax.axvline(hdi[0],  color='crimson',  linewidth=1.0, linestyle='--', alpha=0.7)
    ax.axvline(hdi[1],  color='crimson',  linewidth=1.0, linestyle='--', alpha=0.7,
               label=f"{int(hdi_prob*100)}% HDI: [{hdi[0]:.3g}, {hdi[1]:.3g}]")

    ax.set_title(f"{label}\n{mean:.3g}  [{hdi[0]:.3g} – {hdi[1]:.3g}]", fontsize=8)
    # ax.legend(fontsize=6, loc='best')
    ax.tick_params(labelsize=7)


# ======================================================================================================================
# Parameter definitions: (filename_stem, display_label, unit)
# ======================================================================================================================

PARAMS = [
    ("beta",  "β — Ramp 1 dip",           "°"),
    ("teta",  "θ — Ramp 2 dip",           "°"),
    ("omega", "ω — Ramp 3 dip",           "°"),
    ("Y_r2",  "Y_r2 — Ramp 1→2 position", "m"),
    ("Y_r3",  "Y_r3 — Ramp 2→3 position", "m"),
    ("W",     "W — Upper hinge width",    "m"),
    ("W2",    "W2 — Lower hinge width",   "m"),
    ("Smax",  "Smax — Max shortening",    "mm/yr"),
]

COLORS = [
    "steelblue", "darkorange", "seagreen", "mediumpurple",
    "crimson",   "teal",       "goldenrod", "slategray",
]


# ======================================================================================================================
# Main
# ======================================================================================================================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    traces_dir = sys.argv[1]
    out_dir    = sys.argv[2] if len(sys.argv) > 2 else traces_dir

    os.makedirs(out_dir, exist_ok=True)

    # ── load all traces ──────────────────────────────────────────────────────
    loaded = {}
    for stem, label, unit in PARAMS:
        fpath = os.path.join(traces_dir, f'{stem}.txt')
        if os.path.isfile(fpath):
            loaded[stem] = (np.loadtxt(fpath), label, unit)
        else:
            print(f"Warning: {fpath} not found — skipping {stem}")

    if not loaded:
        print(f"No trace files found in {traces_dir}")
        sys.exit(1)

    n_params = len(loaded)
    n_cols   = 4
    n_rows   = (n_params + n_cols - 1) // n_cols

    # ── figure 1: individual posteriors ─────────────────────────────────────
    fig1, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4 * n_cols, 3.5 * n_rows),
                              constrained_layout=True)
    axes = np.array(axes).flatten()

    for k, (stem, label, unit) in enumerate(PARAMS):
        if stem not in loaded:
            axes[k].set_visible(False)
            continue
        samples, lbl, unit = loaded[stem]
        plot_histo(samples, f"{lbl}\n({unit})", axes[k], color=COLORS[k % len(COLORS)])

    # hide unused subplots
    for k in range(len(PARAMS), len(axes)):
        axes[k].set_visible(False)

    fig1.suptitle("Posterior distributions — kinematic inversion", fontsize=11, y=1.01)
    out1 = os.path.join(out_dir, 'posterior_histo.pdf')
    fig1.savefig(out1, bbox_inches='tight')
    print(f"Saved: {out1}")

    # ── figure 2: dip angles together ───────────────────────────────────────
    dip_params = [("beta", "β"), ("teta", "θ"), ("omega", "ω")]
    fig2, ax = plt.subplots(figsize=(6, 4))
    for (stem, short), color in zip(dip_params, ["steelblue", "darkorange", "seagreen"]):
        if stem in loaded:
            samples, _, _ = loaded[stem]
            ax.hist(samples, bins=30, density=True, histtype='stepfilled',
                    alpha=0.4, color=color, label=short)
            ax.axvline(samples.mean(), color=color, linewidth=1.5, linestyle='-')
            hdi = az.hdi(samples, hdi_prob=0.95)
            ax.axvline(hdi[0], color=color, linewidth=1.0, linestyle='--', alpha=0.7)
            ax.axvline(hdi[1], color=color, linewidth=1.0, linestyle='--', alpha=0.7)

    ax.set_xlabel("Dip angle (°)")
    ax.set_ylabel("Density")
    ax.set_title("Posterior dip angles β, θ, ω")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    out2 = os.path.join(out_dir, 'posterior_dip_angles.pdf')
    fig2.savefig(out2, bbox_inches='tight')
    print(f"Saved: {out2}")

    plt.show()


if __name__ == "__main__":
    main()
