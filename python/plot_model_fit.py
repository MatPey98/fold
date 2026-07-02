#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regenerate the InSAR data vs model predictions figure from saved trace files.

Usage
-----
    python3 fold/python/plot_model_fit.py <input_file> <traces_dir> [output_dir] [--filter-mode]

    input_file   : same input .py file used for optimize_kinematic.py
    traces_dir   : directory containing the .txt trace files (written by save_traces)
    output_dir   : where to save the PDF (defaults to traces_dir)
    --filter-mode: restrict each parameter to the primary KDE peak before computing
                   the mean (useful when posteriors are bimodal)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from os import path

# ======================================================================================================================
# KDE mode filter (same as in plot_histo.py)
# ======================================================================================================================

def primary_mode_window(samples, n_grid=2000):
    """Return (mode, lo, hi) — position and FWHM window of the primary KDE peak."""
    kde  = gaussian_kde(samples, bw_method='silverman')
    x    = np.linspace(samples.min(), samples.max(), n_grid)
    y    = kde(x)

    peaks, _ = find_peaks(y)
    if len(peaks) == 0:
        return x[np.argmax(y)], x[0], x[-1]

    primary  = peaks[np.argmax(y[peaks])]
    mode     = x[primary]
    half_max = y[primary] / 2

    left_idx  = np.where(y[:primary] < half_max)[0]
    right_idx = np.where(y[primary:] < half_max)[0]
    lo = x[left_idx[-1]]           if len(left_idx)  else x[0]
    hi = x[primary + right_idx[0]] if len(right_idx) else x[-1]
    return mode, lo, hi



# ======================================================================================================================
# Argument parsing
# ======================================================================================================================

args        = sys.argv[1:]
filter_mode = '--filter-mode' in args
args        = [a for a in args if a != '--filter-mode']

if len(args) < 2:
    print(__doc__)
    sys.exit(1)

input_file = args[0]
traces_dir = args[1]
out_dir    = args[2] if len(args) > 2 else traces_dir
os.makedirs(out_dir, exist_ok=True)

# ======================================================================================================================
# Load input file (data paths + fixed parameters)
# ======================================================================================================================

try:
    sys.path.append(path.dirname(path.abspath(input_file)))
    modname = path.splitext(path.basename(input_file))[0]
    exec(f"from {modname} import *")
except Exception:
    try:
        exec(open(input_file).read())
    except Exception as e:
        print(f"Problem reading input file: {e}")
        sys.exit(1)

# ======================================================================================================================
# Load data (same as optimize_kinematic.py)
# ======================================================================================================================

from kinematic import compute_fault_and_axial_surfaces
from profile import Profile
from read_data import MNT, Insar

profile = Profile(coupe, chemin_coupe, width)
profile.linspace(n)
y_topo = np.max(profile.abscisse) - profile.abscisse

try:
    _mnt_err = globals().get('mnt_err', None)
    topodata = MNT(mnt, _mnt_err, chemin_mnt)
    z_topo   = topodata.elevations(profile.points)
except Exception as e:
    print(f"Warning: No elevation data ({e})")
    z_topo = np.zeros(len(y_topo))

# InSAR — vertical
insar_data_verti = Insar(insar_vertical, chemin_insar, profile)
abscisses_verti, velocities_verti = insar_data_verti.projection_insar(width)
abscisses_verti = np.max(abscisses_verti) - abscisses_verti
mask = (abscisses_verti < Ymax) & (abscisses_verti > Ymin)
y_insar          = abscisses_verti
z_insar          = velocities_verti
y_insar_filtered = abscisses_verti[mask]
z_insar_filtered = velocities_verti[mask]

# InSAR — horizontal (shortening)
insar_data_horiz = Insar(insar_horizontal, chemin_insar, profile)
abscisses_horiz, velocities_horiz = insar_data_horiz.projection_insar(width)
y_insar_short          = np.max(abscisses_horiz) - abscisses_horiz
z_insar_short          = velocities_horiz
y_insar_short_filtered = y_insar_short[mask]
z_insar_short_filtered = z_insar_short[mask]

# ======================================================================================================================
# Load and filter traces
# ======================================================================================================================

PARAM_NAMES = ["beta", "teta", "omega", "Y_r2", "Y_r3", "W", "W2", "Smax"]

raw = {}
for name in PARAM_NAMES:
    fpath = os.path.join(traces_dir, f'{name}.txt')
    if not os.path.isfile(fpath):
        print(f"Error: trace file not found: {fpath}")
        sys.exit(1)
    raw[name] = np.loadtxt(fpath)

# Apply joint mode filter: keep sample i only if ALL parameters fall within
# their respective primary KDE peak window. This preserves parameter correlations.
suffix = " (mode-filtered)" if filter_mode else ""

n_total = len(raw[PARAM_NAMES[0]])
if filter_mode:
    mask_joint = np.ones(n_total, dtype=bool)
    for name in PARAM_NAMES:
        _, lo, hi = primary_mode_window(raw[name])
        mask_joint &= (raw[name] >= lo) & (raw[name] <= hi)
    n_kept = mask_joint.sum()
    if n_kept < 10:
        print("Warning: joint mode filter too restrictive, falling back to KDE modes")
        mask_joint = None
    else:
        print(f"  {n_kept} / {n_total} samples kept after joint mode filter")
else:
    mask_joint = None

# Mode (KDE peak) of each parameter marginal — always computed on full posterior
modes = {name: primary_mode_window(raw[name])[0] for name in PARAM_NAMES}
print(f"\nParameter modes{suffix}:")
for name, val in modes.items():
    print(f"  {name:6s} = {val:.4g}")

# filtered: samples used for realizations envelope
if mask_joint is not None:
    filtered = {name: raw[name][mask_joint] for name in PARAM_NAMES}
else:
    filtered = {name: raw[name] for name in PARAM_NAMES}

# ======================================================================================================================
# Forward model helpers
# ======================================================================================================================

def forward_model(p):
    """Run the forward model with parameter dict p; return (z_vert, z_horiz)."""
    full = dict(p,
                Y_faille=Y_faille, Z_faille=Z_faille,
                Ymin=Ymin, Ymax=Ymax, di=di, n_tot=n_tot)
    res = compute_fault_and_axial_surfaces(full, y_insar, z_insar)
    # G_Z0 in kinematic.py is initialised at 3307 m; subtract to get displacement
    # relative to initial flat layer — matches optimize_kinematic.py line 135
    z_ref = full.get("Z_ref", 3307.0)
    z_v = np.interp(y_insar_filtered, res["Y_save"], res["Z_save"] - z_ref)
    z_h = np.interp(y_insar_filtered, res["Y_save"], res["horizontal_shortening"])
    return z_v, z_h, res

# ======================================================================================================================
# Draw posterior realizations from filtered samples and average predictions
# E[f(θ)] rather than f(E[θ]) — correct for nonlinear models
# ======================================================================================================================

n_real  = globals().get('n_samples', 100)
n_avail = min(len(filtered["beta"]), n_real)
idx_draw = np.random.choice(len(filtered["beta"]), n_avail, replace=False)

vert_stack  = []
horiz_stack = []
fault_realizations = []

for i, idx in enumerate(idx_draw):
    p = {name: filtered[name][idx] for name in PARAM_NAMES}
    try:
        z_v, z_h, res = forward_model(p)
        vert_stack.append(z_v)
        horiz_stack.append(z_h)
        if "Yfaille" in res and "Zfaille" in res:
            fault_realizations.append((res["Yfaille"], res["Zfaille"]))
    except Exception as e:
        print(f"  Warning: sample {idx} failed ({e})")

if not vert_stack:
    print("Error: no valid forward model evaluations")
    sys.exit(1)

# Mean prediction = E[f(θ)] averaged over filtered samples (not f(mean params))
z_vert_mean  = np.mean(vert_stack,  axis=0)
z_horiz_mean = np.mean(horiz_stack, axis=0)

# Constant offset correction: InSAR data has an arbitrary reference (reference pixel).
# The offset is estimated as the mean residual between data and model predictions,
# then removed so the comparison is on the same reference.
offset_vert  = np.nanmean(z_insar_filtered - z_vert_mean)
offset_horiz = np.nanmean(z_insar_short_filtered - z_horiz_mean)
z_vert_mean  += offset_vert
z_horiz_mean += offset_horiz
print(f"\nConstant offset applied:")
print(f"  vertical  : {offset_vert:+.2f} mm")
print(f"  shortening: {offset_horiz:+.2f} mm")

# Representative geometry:
#   1. MAP sample (highest log-posterior) if lp.txt exists — always a valid joint sample
#   2. Joint-filtered median if filter worked
#   3. Full-posterior median otherwise
lp_file = os.path.join(traces_dir, 'lp.txt')
if os.path.isfile(lp_file):
    lp = np.loadtxt(lp_file)
    map_idx = int(np.argmax(lp))
    mean_params = {name: float(raw[name][map_idx]) for name in PARAM_NAMES}
    print(f"\nMAP parameters (sample #{map_idx}, lp={lp[map_idx]:.2f}):")
else:
    mean_params = {name: float(np.median(raw[name])) for name in PARAM_NAMES}
    print(f"\nMedian parameters (per-parameter, full posterior):")
for name, val in mean_params.items():
    print(f"  {name:6s} = {val:.4g}")
_, _, mean_res = forward_model(mean_params)

# ======================================================================================================================
# Plot
# ======================================================================================================================

plt.rcParams.update({
    "figure.figsize": (6, 4), "figure.dpi": 150, "savefig.dpi": 300,
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "legend.fontsize": 7, "lines.linewidth": 1.2,
})

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# ── Panel 1: vertical ────────────────────────────────────────────────────────
ax1.set_title(f"InSAR data vs model predictions{suffix}")
ax1.plot(y_topo, z_topo, 'k-', linewidth=1, label="Topography")
ax1.set_ylabel("Elevation (m)")
ax1.legend(loc="upper left")

ax1b = ax1.twinx()
ax1b.scatter(y_insar, z_insar, s=2, alpha=0.4, color='tab:blue', label="InSAR vertical")
ax1b.plot(y_insar_filtered, z_vert_mean, color='tab:red', linewidth=1.5,
          label=f"Mean model{suffix}")
ax1b.set_ylabel("Vertical deformation (mm)")
ax1b.grid(True, linestyle='--', alpha=0.1)
ax1b.legend(loc="upper right")

# ── Panel 2: shortening ──────────────────────────────────────────────────────
ax2.plot(y_topo, z_topo, 'k-', linewidth=1, label="Topography")
ax2.set_ylabel("Elevation (m)")
ax2.legend(loc="upper left")

ax2b = ax2.twinx()
ax2b.scatter(y_insar_short, z_insar_short, s=2, alpha=0.4, color='tab:orange',
             label="InSAR shortening")
ax2b.plot(y_insar_filtered, z_horiz_mean, color='tab:green', linewidth=1.5,
          label=f"Mean model{suffix}")
ax2b.set_ylabel("Shortening (mm)")
ax2b.grid(True, linestyle='--', alpha=0.1)
ax2b.legend(loc="upper right")

# ── Panel 3: fault geometry ──────────────────────────────────────────────────
ax3.plot(y_topo, z_topo, 'k-', label='Topography')
ax3.set_ylabel("Depth (m)")

for yf, zf in fault_realizations:
    ax3.plot(yf, zf, 'tab:red', alpha=0.1, linewidth=0.5)

if "Yfaille" in mean_res and "Zfaille" in mean_res:
    ax3.plot(mean_res["Yfaille"], mean_res["Zfaille"],
             color='tab:red', linewidth=1.5, label=f"Mode fault{suffix}")

for i in range(1, 5):
    if f"Y_asurf{i}" in mean_res:
        ax3.plot(mean_res[f"Y_asurf{i}"], mean_res[f"Z_asurf{i}"],
                 'k--', alpha=0.5, linewidth=0.8,
                 label="Axial surfaces" if i == 1 else "")

for i in range(1, 5):
    if f"Ych{i}" in mean_res:
        ax3.scatter(mean_res[f"Ych{i}"], mean_res[f"Zch{i}"],
                    color='tab:red', s=20, alpha=0.5,
                    label="Hinges" if i == 1 else "")

ax3.set_title(f"Posterior fault geometry — {n_avail} realizations{suffix}")
ax3.legend(loc="lower left")
ax3.grid(True, linestyle='--', alpha=0.3)

for ax in [ax1, ax2, ax3]:
    ax.set_xlim(min(y_insar), max(y_insar))
    ax.invert_xaxis()

fig.tight_layout(pad=1.0)

fname = 'model_fit_mode.pdf' if filter_mode else 'model_fit.pdf'
out_path = os.path.join(out_dir, fname)
fig.savefig(out_path, bbox_inches='tight')
print(f"\nSaved: {out_path}")
plt.show()
