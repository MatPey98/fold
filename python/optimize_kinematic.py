# !/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
# Disable pytensor C compilation — required when libpython is a static (.a) library
# not compiled with -fPIC (common on servers). Falls back to numpy backend.
os.environ.setdefault('PYTENSOR_FLAGS', 'cxx=')

import matplotlib
matplotlib.use('Agg')   # non-interactive backend — saves PDFs without opening windows
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import warnings
import pytensor
import pytensor.tensor as pt
import sys
import os
from os import path
import getopt

from kinematic import compute_fault_and_axial_surfaces
from profile import Profile
from read_data import *

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def usage():
    print('optimize_kinematic.py infile.py [-h]')
    print('-h Show this screen')

# Load input file
import shutil

try:
    opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])
except:
    print("for help use --help")
    sys.exit()

for o in sys.argv:
    if o in ("-h", "--help"):
        usage()
        sys.exit()

if 1 == len(sys.argv):
    usage()
    assert False, "no input file"
    sys.exit()

fname = sys.argv[1]
print('Read input file {0} '.format(fname))
try:
    try:
        sys.path.append(path.dirname(path.abspath(fname)))
        exec("from " + path.basename(fname) + " import *")
    except:
        exec(open(fname).read())
except Exception as e:
    print('Problem in input file:', e)
    sys.exit()

# Output directory — output_dir from input file, else wdir/output
output_dir = globals().get('output_dir', os.path.join(globals().get('wdir', '.'), 'output'))
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# Copy input file to output directory for reproducibility
shutil.copy2(fname, os.path.join(output_dir, path.basename(fname)))

# ======================================================================================================================
# NUMBER OF SEGMENTS — auto-detected from prior bounds present in the input file
# ======================================================================================================================
# 3 segments : UY_r3, Uomega, UW2 all defined  → beta→hinge→teta→hinge→omega
# 2 segments : Uteta, UY_r2, UW defined         → beta→hinge→teta
# 1 segment  : only Ubeta (and US) defined       → beta only
if 'n_segments' in globals():
    n_segments = int(globals()['n_segments'])
elif all(k in globals() for k in ('UY_r3', 'Uomega', 'UW2')):
    n_segments = 3
elif all(k in globals() for k in ('Uteta', 'UY_r2', 'UW')):
    n_segments = 2
else:
    n_segments = 1

print(f"Model: {n_segments}-segment fault-bend fold")

# ── Active parameters and their Uniform priors ────────────────────────────────
if n_segments == 3:
    PARAM_NAMES  = ["beta", "teta", "omega", "Y_r2", "Y_r3", "W", "W2", "S"]
    PARAM_PRIORS = {
        "beta": Ubeta, "teta": Uteta, "omega": Uomega,
        "Y_r2": UY_r2, "Y_r3": UY_r3, "W": UW, "W2": UW2, "S": US,
    }
elif n_segments == 2:
    PARAM_NAMES  = ["beta", "teta", "Y_r2", "W", "S"]
    PARAM_PRIORS = {
        "beta": Ubeta, "teta": Uteta, "Y_r2": UY_r2, "W": UW, "S": US,
    }
else:  # 1 segment
    PARAM_NAMES  = ["beta", "S"]
    PARAM_PRIORS = {"beta": Ubeta, "S": US}

# ======================================================================================================================
# DATA LOADING
# ======================================================================================================================
# Profile --------------------------------------------------------------------------------------------------------------
profile = Profile(coupe, chemin_coupe, width)
profile.linspace(n)
y_topo = np.max(profile.abscisse) - profile.abscisse

# Topography -----------------------------------------------------------------------------------------------------------
try:
    _mnt_err = globals().get('mnt_err', None)
    topodata = MNT(mnt, _mnt_err, chemin_mnt)
    z_topo = topodata.elevations(profile.points)
except:
    print('Warning: No elevation data')


# InSAR ----------------------------------------------------------------------------------------------------------------
def load_insar_data():
    # Vertical
    insar_data_verti = Insar(d_vert, chemin_insar, profile)
    abscisses_insar_verti, velocities_verti = insar_data_verti.projection_insar(width)
    abscisses_insar_verti = np.max(abscisses_insar_verti) - abscisses_insar_verti
    mask = (abscisses_insar_verti < Ymax) & (abscisses_insar_verti > Ymin)
    y_vert = abscisses_insar_verti[mask]
    z_insar_filtered = velocities_verti[mask]
    y_insar = abscisses_insar_verti
    z_insar = velocities_verti

    # Horizontal data (shortening)
    insar_data_horiz = Insar(d_horz, chemin_insar, profile)
    abscisses_insar_horiz, velocities_horiz = insar_data_horiz.projection_insar(width)
    y_insar_short = np.max(abscisses_insar_horiz) - abscisses_insar_horiz
    z_insar_short = velocities_horiz
    y_horiz = y_insar_short[mask]
    z_insar_short_filtered = z_insar_short[mask]

    return y_vert, z_insar_filtered, z_insar, y_insar, z_insar_short, y_insar_short, z_insar_short_filtered, y_horiz

# Load data once — not at each iteration
y_vert, z_insar_filtered, z_insar, y_insar, z_insar_short, y_insar_short, z_insar_short_filtered, y_horiz = load_insar_data()

def data():
    return z_insar_filtered, z_insar_short_filtered

# ======================================================================================================================
# FORWARD MODEL
# ======================================================================================================================
def forward_model(param_dict):
    """
    Compute expected deformation from a dict of active parameter values.
    Fixed parameters (Y_fault, Z_fault, …) are injected here.
    """
    params = dict(param_dict,
                  Y_fault=Y_fault, Z_fault=Z_fault,
                  Ymin=Ymin, Ymax=Ymax, di=di,
                  n_segments=n_segments)

    results = compute_fault_and_axial_surfaces(params, y_insar, z_insar)

    # Geometric validity checks (conditional on n_segments)
    invalid = False
    if n_segments >= 2:
        Ych1 = results["Ych1"]
        # After applying slip S along ramp beta, the fault tip moves to
        # Y_fault + S*cos(beta). Hinge 1 must not exceed that position.
        beta_rad  = np.deg2rad(param_dict["beta"])
        fault_tip = Y_fault + param_dict["S"] * np.cos(beta_rad)
        invalid = invalid or bool(Ych1 > fault_tip)
    if n_segments == 2:
        Ych2 = results["Ych2"]
        # Hinge 2 must remain within the model domain
        invalid = invalid or bool(np.any(Ych2 < Ymin))
    if n_segments >= 3:
        Ych2 = results["Ych2"]
        Ych3 = results["Ych3"]
        Ych4 = results["Ych4"]
        # Hinge ordering + hinge 4 must remain within the model domain
        invalid = invalid or bool(np.any(Ych2 < Ych3)) or bool(Ych4 < Ymin)

    vert_interp = np.interp(y_vert,  results["Y_def"], results["Z_def"] - Z_fault)
    horz_interp = np.interp(y_horiz, results["Y_def"], results["horizontal_def"])

    if np.any(np.isnan(vert_interp)) or np.any(np.isinf(vert_interp)) or invalid:
        vert_interp = np.random.normal(0, 1e-3, size=len(vert_interp))
    if np.any(np.isnan(horz_interp)) or np.any(np.isinf(horz_interp)) or invalid:
        horz_interp = np.random.normal(0, 1e-3, size=len(horz_interp))

    return vert_interp, horz_interp

# ======================================================================================================================
# PYTENSOR OPERATOR (ForwardModelOp)
# ======================================================================================================================
class ForwardModelOp(pytensor.graph.op.Op):
    itypes = [pt.dvector]
    otypes = [pt.dvector]

    def perform(self, node, inputs, outputs):
        model = inputs[0]
        try:
            param_dict = {name: float(val) for name, val in zip(PARAM_NAMES, model)}
            f_vertical, f_horizontal = forward_model(param_dict)
            if (np.any(np.isnan(f_vertical))  or np.any(np.isinf(f_vertical)) or
                np.any(np.isnan(f_horizontal)) or np.any(np.isinf(f_horizontal))):
                raise ValueError("forward_model returned NaN or inf values.")
            outputs[0][0] = np.concatenate([f_vertical, f_horizontal])
        except Exception as e:
            print(f"Error in ForwardModelOp: {e}")
            outputs[0][0] = np.ones(len(y_vert) + len(y_horiz)) * 1e6

forward_op = ForwardModelOp()

# ======================================================================================================================
# BAYESIAN INFERENCE
# ======================================================================================================================
def _compute_initvals():
    """
    Return a parameter dict that satisfies all geometric ordering constraints.
    Used as starting point for each chain to avoid degenerate initial states
    (e.g. Y_r2 ≈ Y_r3 or beta < teta) that cause 0% acceptance rate.
    """
    iv = {}
    lo, hi = PARAM_PRIORS["beta"];    iv["beta"] = (lo + hi) / 2
    lo, hi = PARAM_PRIORS["S"];        iv["S"] = (lo + hi) / 2
    if n_segments >= 2:
        lo, hi = PARAM_PRIORS["teta"]
        iv["teta"] = min((lo + hi) / 2, iv["beta"] - 5)
        lo, hi = PARAM_PRIORS["Y_r2"]
        iv["Y_r2"] = lo + 0.75 * (hi - lo)   # upper quartile of prior
        lo, hi = PARAM_PRIORS["W"]
        iv["W"] = (lo + hi) / 2
    if n_segments >= 3:
        lo, hi = PARAM_PRIORS["omega"]
        iv["omega"] = min((lo + hi) / 2, iv["teta"] - 5)
        lo, hi = PARAM_PRIORS["Y_r3"]
        iv["Y_r3"] = lo + 0.25 * (hi - lo)   # lower quartile of prior
        lo, hi = PARAM_PRIORS["W2"]
        iv["W2"] = (lo + hi) / 2
        # Enforce Y_r2 > Y_r3 with a safety margin
        if iv["Y_r2"] <= iv["Y_r3"] + 500:
            iv["Y_r2"] = iv["Y_r3"] + 2000
    return iv


def run_inversion():
    initvals = _compute_initvals()
    print(f"Initial values: { {k: f'{v:.1f}' for k, v in initvals.items()} }")

    with pm.Model() as model:
        # Kinematic priors — built dynamically from PARAM_PRIORS
        priors = {name: pm.Uniform(name, lower=lo, upper=hi)
                  for name, (lo, hi) in PARAM_PRIORS.items()}

        # InSAR reference-level offsets — InSAR data has an arbitrary constant;
        # without these parameters S would absorb both shape and absolute level.
        _sigma_c_vert  = float(globals().get('sigma_c_vert',  30))
        _sigma_c_horiz = float(globals().get('sigma_c_horiz', 30))
        c_vert  = pm.Normal("c_vert",  mu=0, sigma=_sigma_c_vert)
        c_horiz = pm.Normal("c_horiz", mu=0, sigma=_sigma_c_horiz)

        # Stack kinematic parameters into model vector (order must match PARAM_NAMES)
        model_vec = pt.stack([priors[name] for name in PARAM_NAMES])
        mu_kin = forward_op(model_vec)
        f_vertical   = mu_kin[:len(y_vert)]  + c_vert
        f_horizontal = mu_kin[len(y_vert):]  + c_horiz
        mu = pm.Deterministic("mu", pt.concatenate([f_vertical, f_horizontal]))

        d_obs_vert, d_obs_horiz = data()
        pm.Normal("InSAR_Vertical",   mu=f_vertical,   sigma=sigma_vert,  observed=d_obs_vert)
        pm.Normal("InSAR_Horizontal", mu=f_horizontal, sigma=sigma_horiz, observed=d_obs_horiz)

        # Geometric ordering constraints (conditional on n_segments)
        if n_segments == 3:
            pm.Potential("invalid_parameters",
                         pm.math.switch(
                             (priors["beta"]  < priors["teta"])  |
                             (priors["teta"]  < priors["omega"]) |
                             (priors["Y_r2"]  < priors["Y_r3"]), -1e6, 0))
        elif n_segments == 2:
            pm.Potential("invalid_parameters",
                         pm.math.switch(priors["beta"] < priors["teta"], -1e6, 0))

        trace = pm.sample(
            draws=niter,
            tune=nburn,
            chains=chains,
            cores=cores,
            step=pm.Metropolis(),   # no fixed scaling — PyMC auto-tunes per chain
            initvals=initvals,
            progressbar=True,
        )
    return model, trace


# ======================================================================================================================
# VISUALIZATION
# ======================================================================================================================
def plot_results(trace):
    plt.style.use("seaborn-v0_8-deep")
    az.style.use("default")
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300

    all_vars = PARAM_NAMES + ["c_vert", "c_horiz"]
    summary = az.summary(trace, var_names=all_vars)
    print("\nPosterior parameter summary (including offsets):")
    print(summary)

    # Trace and posterior plots -----------------------------------------------------------------------------------------
    plt.rcParams.update({'font.size': 6})
    try:
        az.plot_trace(trace, var_names=PARAM_NAMES, compact=True, figsize=(5, 3), combined=True)
        plt.gcf().savefig(os.path.join(output_dir, 'trace.pdf'), bbox_inches='tight')

        az.plot_posterior(trace, var_names=PARAM_NAMES, kind='hist', textsize=6, figsize=(5, 3))
        plt.gcf().savefig(os.path.join(output_dir, 'posterior.pdf'), bbox_inches='tight')

        az.plot_pair(trace, var_names=PARAM_NAMES, kind='hexbin', marginals=True, textsize=6, figsize=(5, 3))
        plt.gcf().savefig(os.path.join(output_dir, 'corner.pdf'), bbox_inches='tight')

        angle_params  = [p for p in PARAM_NAMES if p in ("beta", "teta", "omega", "S")]
        geom_params   = [p for p in PARAM_NAMES if p in ("Y_r2", "Y_r3", "W", "W2")]
        offset_params = ["c_vert", "c_horiz"]
        if angle_params:
            az.plot_forest(trace, var_names=angle_params, combined=True, hdi_prob=0.95,
                           textsize=6, linewidth=1, markersize=2, figsize=(5, 3))
            plt.gcf().savefig(os.path.join(output_dir, 'forest_angles.pdf'), bbox_inches='tight')
        if geom_params:
            az.plot_forest(trace, var_names=geom_params, combined=True, hdi_prob=0.95,
                           textsize=6, linewidth=1, markersize=2, figsize=(5, 3))
            plt.gcf().savefig(os.path.join(output_dir, 'forest_geometry.pdf'), bbox_inches='tight')
        az.plot_forest(trace, var_names=offset_params, combined=True, hdi_prob=0.95,
                       textsize=6, linewidth=1, markersize=2, figsize=(5, 2))
        plt.gcf().savefig(os.path.join(output_dir, 'forest_offsets.pdf'), bbox_inches='tight')
    except Exception as e:
        print(f"Warning: arviz plotting failed ({e})")

    # Model figure formatting ------------------------------------------------------------------------------------------
    plt.rcParams.update({
        "figure.figsize": (6, 4), "figure.dpi": 150, "savefig.dpi": 300,
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "legend.fontsize": 7, "lines.linewidth": 1.2,
        "axes.linewidth": 0.8, "xtick.major.size": 3, "ytick.major.size": 3,
        "grid.linestyle": "--", "grid.alpha": 0.1
    })

    # Posterior mean prediction
    mu_all       = trace.posterior["mu"].mean(dim=["chain", "draw"]).values
    f_vertical   = mu_all[:len(y_vert)]
    f_horizontal = mu_all[len(y_vert):]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Panel 1: vertical ------------------------------------------------------------------------------------------------
    ax1.set_title("InSAR data vs model predictions")
    ax1.plot(y_topo, z_topo, 'k-', linewidth=1, label="Topography")
    ax1.set_ylabel("Elevation (m)")
    ax1.legend(loc="upper left")

    ax1b = ax1.twinx()
    ax1b.scatter(y_insar, z_insar, s=2, alpha=0.5, color='tab:blue', label="InSAR vertical uplift")
    ax1b.plot(y_vert, f_vertical, color='tab:red', linewidth=1.5, label="Mean model predictions")
    ax1b.set_ylabel("Vertical deformation (mm)")
    ax1b.grid(True, linestyle='--', alpha=0.1)
    ax1b.legend(loc="upper right")

    # Panel 2: horizontal (shortening) --------------------------------------------------------------------------------
    ax2.plot(y_topo, z_topo, 'k-', linewidth=1, label="Topography")
    ax2.set_ylabel("Elevation (m)")
    ax2.legend(loc="upper left")

    ax2b = ax2.twinx()
    ax2b.scatter(y_insar_short, z_insar_short, s=2, alpha=0.5, color='tab:orange', label="N022° InSAR shortening")
    ax2b.plot(y_horiz, f_horizontal, color='tab:green', linewidth=1.5, label="Mean model predictions")
    ax2b.grid(True, linestyle='--', alpha=0.1)
    ax2b.set_ylabel("Shortening (mm)")
    ax2b.legend(loc="upper right")

    # Panel 3: posterior fault geometry --------------------------------------------------------------------------------
    ax3.plot(y_topo, z_topo, 'k-', label='Topography')
    ax3.set_ylabel("Depth (m)")

    # Extract and filter posterior samples
    samples = {name: trace.posterior[name].values.flatten() for name in PARAM_NAMES}
    valid   = np.ones(len(samples[PARAM_NAMES[0]]), dtype=bool)
    for name in PARAM_NAMES:
        valid &= ~np.isnan(samples[name])
    for name in PARAM_NAMES:
        samples[name] = samples[name][valid]

    # Draw random posterior realizations
    if len(samples[PARAM_NAMES[0]]) > 0:
        num_samples  = min(n_samples, len(samples[PARAM_NAMES[0]]))
        sample_idxs  = np.random.choice(len(samples[PARAM_NAMES[0]]), num_samples, replace=False)

        for idx in sample_idxs:
            param_dict = {name: samples[name][idx] for name in PARAM_NAMES}
            kin_params = dict(param_dict, Y_fault=Y_fault, Z_fault=Z_fault,
                              Ymin=Ymin, Ymax=Ymax, di=di, n_segments=n_segments)
            try:
                res = compute_fault_and_axial_surfaces(kin_params, y_insar, z_insar)
                if "Y_fault_trace" in res:
                    ax3.plot(res["Y_fault_trace"], res["Z_fault_trace"],
                             'tab:red', alpha=0.1, linewidth=0.5)
            except Exception as e:
                print(f"Error computing fault for sample {idx}: {e}")

    # Mean fault
    mean_param_dict = {name: np.nanmean(samples[name]) for name in PARAM_NAMES}
    mean_kin_params = dict(mean_param_dict, Y_fault=Y_fault, Z_fault=Z_fault,
                           Ymin=Ymin, Ymax=Ymax, di=di, n_segments=n_segments)
    try:
        mean_results = compute_fault_and_axial_surfaces(mean_kin_params, y_insar, z_insar)

        ax3.plot(mean_results["Y_fault_trace"], mean_results["Z_fault_trace"],
                 color='tab:red', linewidth=1.5, label="Mean fault")

        for i in range(1, 5):
            if f"Y_asurf{i}" in mean_results:
                ax3.plot(mean_results[f"Y_asurf{i}"], mean_results[f"Z_asurf{i}"],
                         'k--', alpha=0.5, linewidth=0.8,
                         label="Axial surfaces" if i == 1 else "")

        for i in range(1, 5):
            if f"Ych{i}" in mean_results:
                ax3.scatter(mean_results[f"Ych{i}"], mean_results[f"Zch{i}"],
                            color='tab:red', s=20, alpha=0.5,
                            label="Hinges" if i == 1 else "")

        # Fix ax3 y-limits: top = topo max + 2 km, bottom = deepest fault - 2 km
        z3_max = np.nanmax(z_topo) + 2000
        z3_min = np.nanmin(mean_results["Z_fault_trace"]) - 2000
        ax3.set_ylim(z3_min, z3_max)

    except Exception as e:
        print(f"Error computing mean fault: {e}")

    ax3.legend(loc="lower left")
    realisations = cores * (nburn + niter)
    ax3.set_title(f"Posterior fault geometry — {n_segments} segments, {realisations} iterations")
    ax3.grid(True, linestyle='--', alpha=0.5)

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(min(y_insar), max(y_insar))
        ax.invert_xaxis()

    fig.tight_layout(pad=1.0)
    fig.savefig(os.path.join(output_dir, 'model_fit.pdf'), bbox_inches='tight')
    print(f"Figures saved to: {output_dir}")
    plt.show()

# ======================================================================================================================
# SAVE TRACES
# ======================================================================================================================
def save_traces(trace):
    """Save all posterior samples to text files (one file per parameter).

    Files are written to output_dir/traces/. Each file contains all chains
    concatenated (flattened), one value per line.
    Also saves lp.txt (log-posterior) so plot_model_fit can find the MAP sample.
    """
    traces_dir = os.path.join(output_dir, 'traces')
    os.makedirs(traces_dir, exist_ok=True)
    for var in PARAM_NAMES + ["c_vert", "c_horiz"]:
        samples = trace.posterior[var].values.flatten()
        fname   = os.path.join(traces_dir, f'{var}.txt')
        np.savetxt(fname, samples, fmt='%.6f')
        print(f"  Saved {var:8s} → {fname}")

    # Save log-likelihood as lp proxy (MAP identification in plot_model_fit)
    try:
        mu_samples = trace.posterior["mu"].values          # (chains, draws, n_obs)
        n_vert = len(y_vert)
        f_v = mu_samples[..., :n_vert]
        f_h = mu_samples[..., n_vert:]
        d_v, d_h = data()
        lp = (
            -0.5 * np.sum((d_v - f_v) ** 2 / sigma_vert  ** 2, axis=-1)
            -0.5 * np.sum((d_h - f_h) ** 2 / sigma_horiz ** 2, axis=-1)
        ).flatten()
        np.savetxt(os.path.join(traces_dir, 'lp.txt'), lp, fmt='%.6f')
        print(f"  Saved lp       → {os.path.join(traces_dir, 'lp.txt')}")
    except Exception as e:
        print(f"  Warning: could not save lp.txt ({e})")
    print(f"Traces saved to: {traces_dir}")
    return traces_dir


# ======================================================================================================================
# MAIN EXECUTION
# ======================================================================================================================
if __name__ == "__main__":
    print("###################################################################")
    print("#        Bayesian Inversion for the Kinematic Model               #")
    print("###################################################################")
    print("\nStarting Bayesian inference:")
    model, trace = run_inversion()
    save_traces(trace)
    plot_results(trace)
