#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot posterior histograms from MCMC trace files produced by optimize_kinematic.py.

Usage
-----
    python3 fold/python/plot_histo.py <traces_dir> [output_dir] [--filter-mode]

    traces_dir    : directory containing the .txt trace files
    output_dir    : where to save the PDF (defaults to traces_dir)
    --filter-mode : keep only samples within the FWHM of the primary KDE peak
                    (use when posteriors are bimodal and you want the dominant mode)

Each .txt file must contain one posterior sample per line (as written by save_traces()).
The HDI (95% highest density interval) is computed with arviz.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks


# ======================================================================================================================
# KDE mode finder
# ======================================================================================================================

def primary_mode_window(samples, n_grid=2000):
    """
    Find the primary (highest) KDE peak and return its FWHM window [lo, hi].

    Parameters
    ----------
    samples : 1-D array
    n_grid  : KDE evaluation grid size

    Returns
    -------
    mode : float  — position of the primary peak
    lo   : float  — left FWHM boundary
    hi   : float  — right FWHM boundary
    """
    kde  = gaussian_kde(samples, bw_method='silverman')
    x    = np.linspace(samples.min(), samples.max(), n_grid)
    y    = kde(x)

    peaks, _ = find_peaks(y)
    if len(peaks) == 0:
        # unimodal — treat the whole range as the "mode window"
        return x[np.argmax(y)], x[0], x[-1]

    primary  = peaks[np.argmax(y[peaks])]
    mode     = x[primary]
    half_max = y[primary] / 2

    left_idx  = np.where(y[:primary] < half_max)[0]
    right_idx = np.where(y[primary:] < half_max)[0]
    lo = x[left_idx[-1]]          if len(left_idx)  else x[0]
    hi = x[primary + right_idx[0]] if len(right_idx) else x[-1]
    return mode, lo, hi


# ======================================================================================================================
# HDI + histogram helper
# ======================================================================================================================

def plot_histo(samples, label, ax, hdi_prob=0.95, bins=30, color='steelblue',
               filter_mode=False, mask=None, map_value=None):
    """
    Plot a posterior histogram with reference value and HDI on ax.

    Parameters
    ----------
    samples   : 1-D array of posterior draws
    label     : parameter name shown in title
    ax        : matplotlib Axes
    hdi_prob  : credible interval probability (default 0.95)
    bins      : number of histogram bins
    color     : fill color
    filter_mode : if True, restrict samples using the joint mask
    mask      : boolean array (joint mode mask); used when filter_mode=True
    map_value : if provided, draw this value as the reference (MAP/best-fit);
                otherwise use the median of stat_samples
    """
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        ax.set_title(f"{label}\n(no data)")
        return

    # Always show the full distribution in the background
    ax.hist(samples, bins=bins, density=True, histtype='stepfilled',
            alpha=0.25, color=color)

    mode, lo, hi = primary_mode_window(samples)

    if filter_mode and mask is not None:
        sel = samples[mask]
        if len(sel) < 10:
            sel = samples
        ax.hist(sel, bins=bins, density=True, histtype='stepfilled',
                alpha=0.55, color=color)
        stat_samples = sel
        suffix = " (mode)"
    else:
        stat_samples = samples
        suffix = ""

    hdi = az.hdi(stat_samples, hdi_prob=hdi_prob)

    # Reference value: MAP if available, else median
    if map_value is not None:
        ref = map_value
        ref_label = "MAP"
    else:
        ref = float(np.median(stat_samples))
        ref_label = "median"

    ax.axvline(ref,    color='crimson', linewidth=1.5, label=ref_label)
    ax.axvline(hdi[0], color='crimson', linewidth=1.0, linestyle='--', alpha=0.7)
    ax.axvline(hdi[1], color='crimson', linewidth=1.0, linestyle='--', alpha=0.7)
    ax.axvline(mode,   color='black',   linewidth=1.0, linestyle=':',  alpha=0.8)

    ax.set_title(f"{label}{suffix}\n{ref_label}={ref:.3g}  [{hdi[0]:.3g} – {hdi[1]:.3g}]",
                 fontsize=8)
    ax.tick_params(labelsize=7)


# ======================================================================================================================
# Parameter definitions: (filename_stem, display_label, unit)
# ======================================================================================================================

PARAMS = [
    ("beta",    "β — Ramp 1 dip",           "°"),
    ("teta",    "θ — Ramp 2 dip",           "°"),
    ("omega",   "ω — Ramp 3 dip",           "°"),
    ("Y_r2",    "Y_r2 — Ramp 1→2 position", "m"),
    ("Y_r3",    "Y_r3 — Ramp 2→3 position", "m"),
    ("W",       "W — Upper hinge width",    "m"),
    ("W2",      "W2 — Lower hinge width",   "m"),
    ("S",       "S — Shortening",           "mm/yr"),
    ("c_vert",  "c_vert — Vertical offset",  "mm"),
    ("c_horiz", "c_horiz — Horiz. offset",   "mm"),
]

COLORS = [
    "steelblue", "darkorange", "seagreen", "mediumpurple",
    "crimson",   "teal",       "goldenrod", "slategray",
    "deepskyblue", "coral",
]


# ======================================================================================================================
# Main
# ======================================================================================================================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    args        = sys.argv[1:]
    filter_mode = '--filter-mode' in args
    args        = [a for a in args if a != '--filter-mode']

    traces_dir = args[0]
    out_dir    = args[1] if len(args) > 1 else traces_dir

    os.makedirs(out_dir, exist_ok=True)

    # ── load all traces ──────────────────────────────────────────────────────
    _ALIASES = {"S": "Smax"}   # backward compat: old runs used Smax instead of S
    loaded = {}
    for stem, label, unit in PARAMS:
        fpath = os.path.join(traces_dir, f'{stem}.txt')
        if not os.path.isfile(fpath) and stem in _ALIASES:
            fpath = os.path.join(traces_dir, f'{_ALIASES[stem]}.txt')
            if os.path.isfile(fpath):
                print(f"  Note: using legacy trace '{_ALIASES[stem]}.txt' for parameter '{stem}'")
        if os.path.isfile(fpath):
            loaded[stem] = (np.loadtxt(fpath), label, unit)
        else:
            print(f"Warning: {fpath} not found — skipping {stem}")

    if not loaded:
        print(f"No trace files found in {traces_dir}")
        sys.exit(1)

    # ── MAP sample from lp.txt (if available) ───────────────────────────────
    lp_file = os.path.join(traces_dir, 'lp.txt')
    if os.path.isfile(lp_file):
        lp = np.loadtxt(lp_file)
        map_idx = int(np.argmax(lp))
        map_values = {stem: loaded[stem][0][map_idx] for stem in loaded}
        print(f"MAP sample: index {map_idx}, lp={lp[map_idx]:.2f}")
    else:
        map_values = None
        print("lp.txt not found — using per-parameter median as reference")

    # ── joint mode filter ────────────────────────────────────────────────────
    # Compute one joint mask: a sample survives only if ALL parameters fall
    # within their respective primary KDE peak window simultaneously.
    # This ensures consistent statistics across parameters (same as plot_model_fit.py).
    if filter_mode:
        n_total = len(next(iter(loaded.values()))[0])
        mask_joint = np.ones(n_total, dtype=bool)
        for stem, _, _ in PARAMS:
            if stem in loaded:
                samples_s = loaded[stem][0]
                _, lo, hi = primary_mode_window(samples_s)
                mask_joint &= (samples_s >= lo) & (samples_s <= hi)
        n_kept = mask_joint.sum()
        print(f"Joint mode filter: {n_kept} / {n_total} samples kept")
        if n_kept < 10:
            print("Warning: joint mode filter too restrictive, falling back to KDE modes")
            mask_joint = None
    else:
        mask_joint = None

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
        map_val = map_values[stem] if map_values is not None else None
        plot_histo(samples, f"{lbl}\n({unit})", axes[k],
                   color=COLORS[k % len(COLORS)], filter_mode=filter_mode,
                   mask=mask_joint, map_value=map_val)

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
            mode, lo, hi = primary_mode_window(samples)
            sel = samples[mask_joint] if filter_mode else samples
            if len(sel) < 10:
                sel = samples
            ax.hist(samples, bins=30, density=True, histtype='stepfilled',
                    alpha=0.2, color=color)
            ax.hist(sel, bins=30, density=True, histtype='stepfilled',
                    alpha=0.45, color=color, label=short)
            ref = map_values[stem] if map_values else float(np.median(sel))
            ax.axvline(ref, color=color, linewidth=1.5, linestyle='-')
            hdi = az.hdi(sel, hdi_prob=0.95)
            ax.axvline(hdi[0], color=color, linewidth=1.0, linestyle='--', alpha=0.7)
            ax.axvline(hdi[1], color=color, linewidth=1.0, linestyle='--', alpha=0.7)
            ax.axvline(mode,   color=color, linewidth=1.0, linestyle=':',  alpha=0.8)

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
