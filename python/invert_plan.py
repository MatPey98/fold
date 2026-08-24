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
    Abscisses are reversed using profile.l so that InSAR data is aligned with
    the topography (which uses np.max(profile.abscisse) = profile.l as origin).

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
        # Use profile.l (full shapefile length) as the reversal origin so that
        # InSAR data is co-registered with the topography profile.
        abscisses = profile.l - abscisses
        centers, median, std = data.insar_statistics(width, nbins=nbins)
        centers = profile.l - centers
        return abscisses, velocities, centers, median, std
    except Exception as e:
        print(f"Warning: Could not load InSAR data ({filename}): {e}")
        return None

# ======================================================================================================================
# Profile — length is determined by the shapefile (coupe12.shp)
# ======================================================================================================================
profile   = Profile(coupe, chemin_coupe, width)
profile.linspace(n)
abscisses = profile.l - profile.abscisse   # reversed: 0 = far end, profile.l = start
xmin, xmax = abscisses[0], abscisses[-1]  # xmin = profile.l, xmax = 0

# ======================================================================================================================
# InSAR — load datasets defined in the input file
#
# Two supported formats:
#   1. Unified list:
#        insar = [("vert.tif", "vertical 2003-2019"), ("short.tif", "shortening")]
#   2. Separate vertical / shortening lists:
#        insar_verticals   = [("vert.tif",  "2003-2019")]
#        insar_shortenings = [("short.tif", "2003-2019")]
# All files are loaded from chemin_insar.
# ======================================================================================================================

_COLORS = ["dodgerblue", "coral", "darkseagreen", "mediumpurple", "goldenrod", "teal"]
_nbins  = globals().get('nbins', 120)

_insar_list = globals().get('insar', [])
if not _insar_list:
    # Support separate vertical / shortening lists
    _insar_list = (globals().get('insar_verticals',   []) +
                   globals().get('insar_shortenings', []))

insar_results = [(r, label) for (fname, label) in _insar_list
                 for r in [_load_insar(fname, chemin_insar, profile, width, _nbins)]]

# ======================================================================================================================
# Topography (DEM)
#
# mnt         — high-resolution DEM used for dip plane fitting (required)
# mnt_figure  — optional low-resolution DEM used only for the topography profile
#               display; falls back to mnt if not defined
# ======================================================================================================================
elevations   = None
topodata     = None   # high-res DEM used for dip computations
_mnt_err     = globals().get('mnt_err',    None)
_mnt_figure  = globals().get('mnt_figure', None)

try:
    topodata = MNT(mnt, _mnt_err, chemin_mnt)
except Exception as e:
    print(f"Warning: Could not load high-res DEM ({e})")

elev_median = elev_std = None
try:
    _topo_display = MNT(_mnt_figure, None, chemin_mnt) if _mnt_figure else topodata
    if _topo_display is not None:
        # Sample elevations across the full width band (n_transverse cross-track samples)
        # and compute median and std at each profile point.
        _n_trans   = 15
        _offsets   = np.linspace(-profile.w / 2, profile.w / 2, _n_trans)
        _elev_med  = []
        _elev_std  = []
        for pt in profile.points:
            row = []
            for off in _offsets:
                spt = [pt[0] + off * profile.n[0], pt[1] + off * profile.n[1]]
                z = _topo_display.elevations([spt])[0]
                if not np.isnan(z) and z != 0:
                    row.append(z)
            _elev_med.append(np.median(row) if row else np.nan)
            _elev_std.append(np.std(row)   if row else np.nan)
        elev_median = np.array(_elev_med)
        elev_std    = np.array(_elev_std)
        elevations  = elev_median   # keep for ylim computation below
except Exception as e:
    print(f"Warning: No elevation data for display ({e})")

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
abs_seismic = prof_seismic = mag = rms_values = times_seismic = None
try:
    seismic_data = Seismic(seismic, chemin_seismic, profile)
    abs_seismic, prof_seismic, mag, z_errors_seismic, times_seismic = \
        seismic_data.projection_seismic(width_seismic)
    if len(abs_seismic) == 0:
        raise ValueError("no seismic events projected onto profile")
    abs_seismic = profile.l - np.array(abs_seismic)
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
if elev_median is not None:
    ax1.plot(abscisses, elev_median, color="black", lw=1)
    if elev_std is not None:
        ax1.fill_between(abscisses,
                         elev_median - elev_std,
                         elev_median + elev_std,
                         color="gray", alpha=0.35, label="topo ±σ")
    _elev_lo = np.nanmin(elev_median - (elev_std if elev_std is not None else 0))
    _elev_hi = np.nanmax(elev_median + (elev_std if elev_std is not None else 0))
    ax1.set_ylim(_elev_lo - 200, _elev_hi + 200)

ax1.set_xlim(xmin, xmax)
ax1.set_xlabel("Distance (km)")
ax1.set_ylabel("Elevation (km)")
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e3:.1f}'))
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e3:.1f}'))
ax1.tick_params(axis='y', labelcolor='k')

ax1b = ax1.twinx()

# InSAR datasets
_all_velocities = []
for i, (result, label) in enumerate(insar_results):
    if result is not None:
        _, velocities, centers, median, std = result
        color = _COLORS[i % len(_COLORS)]
        ax1b.plot(centers, median, color=color, linewidth=2, label=label)
        ax1b.fill_between(centers, median - std, median + std,
                          color=color, alpha=0.2)
        _all_velocities.append(velocities)

ax1b.set_ylabel("Displacements (mm)", color='k')
ax1b.set_xlim(xmin, xmax)

# ylim from all datasets combined
if _all_velocities:
    _v = np.concatenate(_all_velocities)
    ax1b.set_ylim(np.nanmin(_v) - 10, np.nanmax(_v) + 10)

ax1b.legend(loc="upper right", fontsize=7)

# ── Lower panel: cross-section ───────────────────────────────────────────────
if elevations is not None:
    ax2.plot(abscisses, elevations, color="black")

ax2.set_xlabel("Horizontal distance (km)")
ax2.set_ylabel("Elevation (km)")
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e3:.1f}'))
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e3:.1f}'))

# Strata dip measurements
if dip is not None and elevations is not None:
    dip.plot_all_strata(topodata, profile, length_dip, ax2)

# Fault dip measurements
if dip_fault is not None and elevations is not None:
    dip_fault.plot_all_fault(topodata, profile, length_dip_fault, ax2)

# Seismic catalogue
if abs_seismic is not None and len(abs_seismic) > 0:
    mag      = np.array(mag)
    zerr_m   = np.array(z_errors_seismic, dtype=float)   # depth error (m)
    times_yr = np.array(times_seismic) # decimal years
    # Depths in catalogue are in km below the Earth's surface.
    # Convert to elevation (m) using mean surface elevation as reference:
    #   elev = mean_surface_elev - depth_km × 1000
    _mean_surf = float(np.nanmean(elevations)) if elevations is not None else 4000.
    depths_m = -np.array(prof_seismic) * 1000 + _mean_surf

    # Point size proportional to magnitude (area ∝ M)
    mag_min, mag_max = mag.min(), mag.max()
    sizes = 20 + 200 * ((mag - mag_min) / (mag_max - mag_min + 1e-9)) ** 2

    # Color = time (decimal year)
    sc2 = ax2.scatter(abs_seismic, depths_m,
                      c=times_yr, cmap="plasma",
                      edgecolor="k", linewidths=0.4,
                      s=sizes, zorder=3)

    # Vertical error bars: depthError from catalogue (m), default = 5 km
    ax2.errorbar(abs_seismic, depths_m,
                 yerr=zerr_m,
                 fmt='none', ecolor='gray', alpha=0.5, capsize=2, zorder=2)

    cax  = inset_axes(ax2, width="3%", height="30%", loc="lower left", borderpad=1)
    cbar = fig.colorbar(sc2, cax=cax)
    cbar.set_label("Year")

    # Legend for magnitudes
    for m_leg in np.linspace(mag_min, mag_max, 3):
        s_leg = 20 + 200 * ((m_leg - mag_min) / (mag_max - mag_min + 1e-9)) ** 2
        ax2.scatter([], [], s=s_leg, color='gray', edgecolor='k',
                    linewidths=0.4, label=f"M {m_leg:.1f}")
    ax2.legend(loc="upper right", title="Magnitude", fontsize=7, title_fontsize=7)

ax2.legend(loc="upper right")
ax2.grid(True)

# ======================================================================================================================
# Force shared x limits, max depth, and equal-scale cross-section
# ======================================================================================================================
# Shared x range across all panels
for _ax in [ax1, ax1b, ax2]:
    _ax.set_xlim(xmin, xmax)

# Y limits for cross-section.
# bottom_km : lower y-axis limit in km  (e.g. -2 → 2 km below sea level)
# top_km    : upper y-axis limit in km  (e.g.  4 → 4 km above sea level)
_profile_len_km = (xmax - xmin) / 1000.

_topo_top_km = (float(np.nanmax(elevations)) / 1000. + 1.) if elevations is not None else 5.
if 'bottom_km' in globals():
    _bottom_km = float(globals()['bottom_km'])
else:
    _bottom_km = -1.   # default

if 'top_km' in globals():
    _top_km = float(globals()['top_km'])
else:
    _top_km = _topo_top_km

ax2.set_ylim(bottom=_bottom_km * 1000, top=_top_km * 1000)

# Cross-section at scale: resize the figure height so the lower panel
# fits at 1 m horizontal = 1 m vertical (equal aspect).
_x_range = xmax - xmin                                       # m
_y_range = (_top_km - _bottom_km) * 1000                    # m
_fig_w   = fig.get_size_inches()[0]                      # inches
# ax2 occupies 2/3 of figure height (height_ratios=[1,2]).
# Required figure height so ax2 box has equal aspect:
_ax2_w_frac = 0.85   # approximate axes width fraction of figure width
_ax2_h_frac = 2 / 3  # from height_ratios=[1,2]
_required_ax2_h = (_fig_w * _ax2_w_frac) * (_y_range / _x_range)  # inches
_fig_h = max(6., _required_ax2_h / _ax2_h_frac)
fig.set_size_inches(_fig_w, _fig_h)

ax2.set_aspect('equal', adjustable='box')
ax2.set_ylim(bottom=_bottom_km * 1000, top=_top_km * 1000)  # re-apply after set_aspect

# ======================================================================================================================
# Save and display
# ======================================================================================================================
output_dir = globals().get('output_dir', '.')
os.makedirs(output_dir, exist_ok=True)
_coupe_name = os.path.splitext(os.path.basename(coupe))[0]
output_path = os.path.join(output_dir, f"cross_section_{_coupe_name}.pdf")
fig.savefig(output_path, bbox_inches='tight')
print(f"Figure saved to: {output_path}")

plt.show()
