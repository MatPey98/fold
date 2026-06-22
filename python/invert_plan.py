#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import sys
from os import path
import getopt

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from profile import Profile
from read_data import Insar, MNT, Seismic
from dip import Dip

# ======================================================================================================================
# Input file
# ======================================================================================================================

def usage():
    print('invert_plan.py infile.py [-h]')
    print('-h  Show this screen')

try:
    opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])
except:
    print("for help use --help")
    sys.exit()

for o in sys.argv:
    if o in ("-h", "--help"):
        usage()
        sys.exit()

if len(sys.argv) == 1:
    usage()
    assert False, "no input file"

if len(sys.argv) > 1:
    try:
        fname = sys.argv[1]
        print(f"Reading input file: {fname}")
        try:
            sys.path.append(path.dirname(path.abspath(fname)))
            modname = path.splitext(path.basename(fname))[0]
            exec(f"from {modname} import *")
        except:
            exec(open(fname).read())
    except Exception as e:
        print(f"Problem in input file: {e}")
        sys.exit()

# ======================================================================================================================
# Helper: load and project a single InSAR dataset
# ======================================================================================================================

def _load_insar(filename, chemin, profile, width, nbins=120):
    """
    Load an InSAR raster, project onto the profile, and compute binned statistics.

    Returns
    -------
    (abscisses, velocities, bin_centers, median, std) or None if loading fails.
    """
    try:
        data = Insar(filename, chemin, profile)
        abscisses, velocities = data.projection_insar(width)
        if len(abscisses) == 0:
            print(f"Warning: No projected InSAR data for {filename}")
            return None
        abscisses = np.max(abscisses) - abscisses
        centers, median, std = data.insar_statistics(width, nbins=nbins)
        centers = np.max(centers) - centers
        return abscisses, velocities, centers, median, std
    except Exception as e:
        print(f"Warning: Could not load InSAR data ({filename}): {e}")
        return None

# ======================================================================================================================
# Profile
# ======================================================================================================================
profile   = Profile(coupe, chemin_coupe, width)
profile.linspace(n)
abscisses = np.max(profile.abscisse) - profile.abscisse
xmin, xmax = abscisses[0], abscisses[-1]

# ======================================================================================================================
# InSAR — load datasets defined in the input file
#
# Input file must define lists of (filename, label) tuples, e.g.:
#   insar_verticals   = [("vertical_2003-2011.tif",   "2003–2011")]
#   insar_shortenings = [("shortening_2003-2011.tif", "2003–2011")]
# ======================================================================================================================

_COLORS = ["dodgerblue", "coral", "darkseagreen", "mediumpurple", "goldenrod", "teal"]
_nbins  = globals().get('nbins', 120)

_insar_verticals   = globals().get('insar_verticals',   [])
_insar_shortenings = globals().get('insar_shortenings', [])

vertical_results   = [(r, label) for (fname, label) in _insar_verticals
                      for r in [_load_insar(fname, chemin_insar, profile, width, _nbins)]]
shortening_results = [(r, label) for (fname, label) in _insar_shortenings
                      for r in [_load_insar(fname, chemin_insar, profile, width, _nbins)]]

# ======================================================================================================================
# Topography (DEM)
# ======================================================================================================================
elevations = None
try:
    _mnt_err  = globals().get('mnt_err', None)  # optional — defaults to sigma=1
    topodata  = MNT(mnt, _mnt_err, chemin_mnt)
    elevations = topodata.elevations(profile.points)
except Exception as e:
    print(f"Warning: No elevation data ({e})")

# ======================================================================================================================
# Strata dip measurements
# ======================================================================================================================
dip = None
try:
    dip = Dip(chemin_pendages)
except Exception as e:
    print(f"Warning: No strata data ({e})")

# ======================================================================================================================
# Fault dip measurements
# ======================================================================================================================
dip_fault = None
try:
    dip_fault = Dip(chemin_fault)
except Exception as e:
    print(f"Warning: No fault data ({e})")

# ======================================================================================================================
# Seismic catalogue
# ======================================================================================================================
abs_seismic = prof_seismic = mag = rms_values = None
try:
    seismic_data = Seismic(seismic, chemin_seismic, profile)
    abs_seismic, prof_seismic, mag, rms_values = seismic_data.projection_seismic(width_seismic)
    abs_seismic = np.max(abs_seismic) - abs_seismic
except Exception as e:
    print(f"Warning: No seismic data ({e})")

# ======================================================================================================================
# Plot
# ======================================================================================================================
fig = plt.figure(layout="constrained", figsize=(8, 8))
gs  = gridspec.GridSpec(2, 1, height_ratios=[1, 2], hspace=0.1)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# ── Upper panel: topography + InSAR ──────────────────────────────────────────
if elevations is not None:
    ax1.plot(abscisses, elevations, color="black")
    ax1.set_ylim(np.nanmin(elevations) - 200, np.nanmax(elevations) + 200)

ax1.set_xlim(xmin, xmax)
ax1.set_xlabel("Distance (m)")
ax1.set_ylabel("Elevation (m)")
ax1.tick_params(axis='y', labelcolor='k')

ax1b = ax1.twinx()

# Vertical displacements
for i, (result, label) in enumerate(vertical_results):
    if result is not None:
        _, _, centers, median, std = result
        color = _COLORS[i % len(_COLORS)]
        ax1b.plot(centers, median, color=color, linewidth=2,
                  label=f"Vertical {label}")
        ax1b.fill_between(centers, median - std, median + std,
                          color=color, alpha=0.2)

# Horizontal shortening
for i, (result, label) in enumerate(shortening_results):
    if result is not None:
        _, _, centers, median, std = result
        color = _COLORS[i % len(_COLORS)]
        ax1b.plot(centers, median, color=color, linewidth=2, linestyle='--',
                  label=f"Shortening {label}")
        ax1b.fill_between(centers, median - std, median + std,
                          color=color, alpha=0.2)

ax1b.set_ylabel("Velocity (mm/yr)", color='r')
ax1b.set_xlim(xmin, xmax)

# Set ylim from the first available shortening dataset
for result, _ in shortening_results:
    if result is not None:
        _, velocities, *_ = result
        ax1b.set_ylim(np.nanmin(velocities) - 10, np.nanmax(velocities) + 10)
        break

ax1b.legend(loc="upper right", fontsize=7)

# ── Lower panel: cross-section ───────────────────────────────────────────────
if elevations is not None:
    ax2.plot(abscisses, elevations, color="black")

ax2.set_xlabel("Horizontal distance (m)")
ax2.set_ylabel("Depth (m)")

# Strata dip measurements
if dip is not None and elevations is not None:
    dip.plot_all_strata(topodata, profile, length_dip, ax2)

# Fault dip measurements
if dip_fault is not None and elevations is not None:
    dip_fault.plot_all_fault(topodata, profile, length_dip_fault, ax2)

# Seismic catalogue
if abs_seismic is not None:
    sizes = 50 + 150 * (mag - np.min(mag)) / (np.max(mag) - np.min(mag))
    sc2 = ax2.scatter(abs_seismic, -np.array(prof_seismic) * 1000 + 4000,
                      c=mag, cmap="YlOrRd", edgecolor="k", s=sizes)
    ax2.errorbar(abs_seismic, -np.array(prof_seismic) * 1000 + 4000,
                 xerr=2000, yerr=2000, fmt='none', ecolor='k', capsize=3)
    cax  = inset_axes(ax2, width="3%", height="30%", loc="lower left", borderpad=1)
    cbar = fig.colorbar(sc2, cax=cax)
    cbar.set_label("Magnitude")

ax2.legend(loc="upper right")
ax2.grid(True)
ax2.axis("equal")
ax2.set_xlim(xmin, xmax)

# ======================================================================================================================
# Save and display
# ======================================================================================================================
output_dir = globals().get('output_dir', '.')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "cross_section.pdf")
fig.savefig(output_path, bbox_inches='tight')
print(f"Figure saved to: {output_path}")

plt.show()
