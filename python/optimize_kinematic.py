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
    print('invert_plan.py infile.py [-h]')
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
# DATA LOADING
# ======================================================================================================================
# Profile --------------------------------------------------------------------------------------------------------------
profile = Profile(coupe, chemin_coupe, width)  # Build profile from endpoints and azimuth
profile.linspace(n)                            # Discretize into n points along the profile length
y_topo = np.max(profile.abscisse) - profile.abscisse

# Topography -----------------------------------------------------------------------------------------------------------
try:
    _mnt_err = globals().get('mnt_err', None)  # optional — defaults to None (sigma=1)
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
def forward_model(beta, teta, omega, Y_r2, Y_r3, W, W2, Smax):
    """
    Compute expected deformation with scalar parameter values.
    """
    params = {
        "beta": beta, "teta": teta, "omega": omega,
        "Y_r2": Y_r2, "Y_r3": Y_r3,
        "W": W, "W2": W2, "Smax": Smax,
        "Y_fault": Y_fault, "Z_fault": Z_fault,
        "Ymin": Ymin, "Ymax": Ymax,
        "di": di,
    }

    results = compute_fault_and_axial_surfaces(params, y_insar, z_insar)

    Ych1 = results["Ych1"]
    Zch1 = results["Zch1"]
    Ych2 = results["Ych2"]
    Ych3 = results["Ych3"]
    Ych4 = results["Ych4"]

    # Geometric validity checks:
    # 1. Hinge ordering (frontal hinge must be ahead of rear hinge)
    # 2. Base of ramp1 (Ych1) must not exceed the surface fault trace (Y_fault):
    #    if W is too large, Ych1 = Y_r2 + hypo*cos(beta) > Y_fault which is
    #    geometrically impossible (hinge base beyond surface trace)
    # 3. Base of ramp2 hinge (Ych4) must remain within the model domain:
    #    if W2 is too large, Ych4 = Y_r3 - hypo2*cos(omega) < Ymin
    hinge_beyond_surface   = Ych1 > Y_fault
    hinge2_outside_domain  = Ych4 < Ymin
    invalid = (np.any(Ych2 < Ych3) or np.any(np.isnan(Y_r2))
               or hinge_beyond_surface or hinge2_outside_domain)

    vert_interp          = np.interp(y_vert,  results["Y_def"], results["Z_def"] - Z_fault)
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
            model_values = [float(val) for val in model]
            beta, teta, omega, Y_r2, Y_r3, W, W2, Smax = model_values
            f_vertical, f_horizontal = forward_model(beta, teta, omega, Y_r2, Y_r3, W, W2, Smax)
            if np.any(np.isnan(f_vertical)) or np.any(np.isinf(f_vertical)) or np.any(np.isnan(f_horizontal)) or np.any(np.isinf(f_horizontal)):
                raise ValueError("forward_model returned NaN or inf values.")
            outputs[0][0] = np.concatenate([f_vertical, f_horizontal])
        except Exception as e:
            print(f"Error in ForwardModelOp: {e}")
            outputs[0][0] = np.ones(len(y_vert) + len(y_horiz)) * 1e6

forward_op = ForwardModelOp()

# ======================================================================================================================
# BAYESIAN INFERENCE
# ======================================================================================================================
def run_inversion():
    with pm.Model() as model:
        # Prior definitions within the model context
        beta = pm.Uniform("beta", lower=Ubeta[0], upper=Ubeta[1])
        teta = pm.Uniform("teta", lower=Uteta[0], upper=Uteta[1])
        omega = pm.Uniform("omega", lower=Uomega[0], upper=Uomega[1])
        Y_r2 = pm.Uniform("Y_r2", lower=UY_r2[0], upper=UY_r2[1])
        Y_r3 = pm.Uniform("Y_r3", lower=UY_r3[0], upper=UY_r3[1])
        W = pm.Uniform("W", lower=UW[0], upper=UW[1])
        W2 = pm.Uniform("W2", lower=UW2[0], upper=UW2[1])
        Smax = pm.Uniform("Smax", lower=USmax[0], upper=USmax[1])

        # Stack parameters into model vector
        model_vec = pt.stack([beta, teta, omega, Y_r2, Y_r3, W, W2, Smax])
        mu = forward_op(model_vec)
        f_vertical  = mu[:len(y_vert)]
        f_horizontal = mu[len(y_vert):]
        mu = pm.Deterministic("mu", mu)

        d_obs_vert, d_obs_horiz = data()

        pm.Normal("InSAR_Vertical",   mu=f_vertical,   sigma=sigma_vert,  observed=d_obs_vert)
        pm.Normal("InSAR_Horizontal", mu=f_horizontal, sigma=sigma_horiz, observed=d_obs_horiz)

        # Penalize invalid parameter combinations
        pm.Potential("invalid_parameters",
                     pm.math.switch(
                         (beta < teta) | (teta < omega) | (Y_r2 < Y_r3), -1e6, 0)
                     )

        # Bayesian inference sampling
        trace = pm.sample(
            draws=niter,
            tune=nburn,
            chains=chains,
            cores=cores,
            step=pm.Metropolis(scaling=10),
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

    # Parameter summary
    var_names = ["beta", "teta", "omega", "Y_r2", "Y_r3", "W", "W2", "Smax"]

    summary = az.summary(trace, var_names=var_names)
    print("\nPosterior parameter summary:")
    print(summary)

    # Trace and posterior plots -----------------------------------------------------------------------------------------
    plt.rcParams.update({'font.size': 6})
    try:
        az.plot_trace(trace, var_names=var_names, compact=True, figsize=(5, 3), combined=True)
        plt.gcf().savefig(os.path.join(output_dir, 'trace.pdf'), bbox_inches='tight')

        az.plot_posterior(trace, var_names=var_names, kind='hist', textsize=6, figsize=(5, 3))
        plt.gcf().savefig(os.path.join(output_dir, 'posterior.pdf'), bbox_inches='tight')

        az.plot_pair(trace, var_names=var_names, kind='hexbin', marginals=True, textsize=6, figsize=(5, 3))
        plt.gcf().savefig(os.path.join(output_dir, 'corner.pdf'), bbox_inches='tight')

        az.plot_forest(trace, var_names=["beta", "teta", "omega", "Smax"], combined=True, hdi_prob=0.95, textsize=6,
                       linewidth=1, markersize=2, figsize=(5, 3))
        plt.gcf().savefig(os.path.join(output_dir, 'forest_angles.pdf'), bbox_inches='tight')

        az.plot_forest(trace, var_names=["Y_r2", "Y_r3", "W", "W2"], combined=True, hdi_prob=0.95, textsize=6,
                       linewidth=1, markersize=2, figsize=(5, 3))
        plt.gcf().savefig(os.path.join(output_dir, 'forest_geometry.pdf'), bbox_inches='tight')
    except Exception as e:
        print(f"Warning: arviz plotting failed ({e})")

    # Model figure formatting ------------------------------------------------------------------------------------------
    plt.rcParams.update({
        "figure.figsize": (6, 4),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "lines.linewidth": 1.2,
        "axes.linewidth": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "grid.linestyle": "--",
        "grid.alpha": 0.1
    })

    # Results: vertical and horizontal displacements
    mu_all = trace.posterior["mu"].mean(dim=["chain", "draw"]).values
    f_vertical  = mu_all[:len(y_vert)]
    f_horizontal = mu_all[len(y_vert):]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Panel 1: InSAR data vs model predictions -------------------------------------------------------------------------
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

    # Panel 2: Horizontal (shortening) --------------------------------------------------------------------------------
    ax2.plot(y_topo, z_topo, 'k-', linewidth=1, label="Topography")
    ax2.set_ylabel("Elevation (m)")
    ax2.legend(loc="upper left")

    ax2b = ax2.twinx()
    ax2b.scatter(y_insar_short, z_insar_short, s=2, alpha=0.5, color='tab:orange', label="N022° InSAR shortening")
    ax2b.plot(y_horiz, f_horizontal, color='tab:green', linewidth=1.5, label="Mean model predictions")
    ax2b.grid(True, linestyle='--', alpha=0.1)
    ax2b.set_ylabel("Shortening (mm)")
    ax2b.legend(loc="upper right")

    # Panel 3: Posterior fault geometry --------------------------------------------------------------------------------
    ax3.plot(y_topo, z_topo, 'k-', label='Topography')
    ax3.set_ylabel("Depth (m)")

    # Extract posterior samples
    beta_samples  = trace.posterior["beta"].values.flatten()
    teta_samples  = trace.posterior["teta"].values.flatten()
    omega_samples = trace.posterior["omega"].values.flatten()
    Y_r2_samples  = trace.posterior["Y_r2"].values.flatten()
    Y_r3_samples  = trace.posterior["Y_r3"].values.flatten()
    W_samples     = trace.posterior["W"].values.flatten()
    W2_samples    = trace.posterior["W2"].values.flatten()
    Smax_samples  = trace.posterior["Smax"].values.flatten()

    # Filter NaN values
    valid_indices = (
        ~np.isnan(beta_samples)  & ~np.isnan(teta_samples)  & ~np.isnan(omega_samples) &
        ~np.isnan(Y_r2_samples)  & ~np.isnan(Y_r3_samples)  &
        ~np.isnan(W_samples)     & ~np.isnan(W2_samples)     & ~np.isnan(Smax_samples)
    )

    beta_samples  = beta_samples[valid_indices]
    teta_samples  = teta_samples[valid_indices]
    omega_samples = omega_samples[valid_indices]
    Y_r2_samples  = Y_r2_samples[valid_indices]
    Y_r3_samples  = Y_r3_samples[valid_indices]
    W_samples     = W_samples[valid_indices]
    W2_samples    = W2_samples[valid_indices]
    Smax_samples  = Smax_samples[valid_indices]

    # Draw random posterior realizations
    if len(beta_samples) > 0:
        num_samples = min(n_samples, len(beta_samples))
        sample_indices = np.random.choice(len(beta_samples), num_samples, replace=False)

        for idx in sample_indices:
            params_sample = {
                "beta": beta_samples[idx], "teta": teta_samples[idx],
                "omega": omega_samples[idx],
                "Y_r2": Y_r2_samples[idx], "Y_r3": Y_r3_samples[idx],
                "W": W_samples[idx], "W2": W2_samples[idx],
                "Smax": Smax_samples[idx],
                "Y_fault": Y_fault, "Z_fault": Z_fault,
                "Ymin": Ymin, "Ymax": Ymax,
                "di": di,
            }

            try:
                sample_results = compute_fault_and_axial_surfaces(params_sample, y_insar, z_insar)
                if "Y_fault_trace" in sample_results and "Z_fault_trace" in sample_results:
                    ax3.plot(sample_results["Y_fault_trace"], sample_results["Z_fault_trace"], 'tab:red', alpha=0.1, linewidth=0.5)
            except Exception as e:
                print(f"Error computing fault for sample {idx}: {e}")

    # Mean fault
    beta_mean  = np.nanmean(beta_samples)
    teta_mean  = np.nanmean(teta_samples)
    omega_mean = np.nanmean(omega_samples)
    Y_r2_mean  = np.nanmean(Y_r2_samples)
    Y_r3_mean  = np.nanmean(Y_r3_samples)
    W_mean     = np.nanmean(W_samples)
    W2_mean    = np.nanmean(W2_samples)
    Smax_mean  = np.nanmean(Smax_samples)

    params_mean = {
        "beta": beta_mean, "teta": teta_mean, "omega": omega_mean,
        "Y_r2": Y_r2_mean, "Y_r3": Y_r3_mean,
        "W": W_mean, "W2": W2_mean, "Smax": Smax_mean,
        "Y_fault": Y_fault, "Z_fault": Z_fault,
        "Ymin": Ymin, "Ymax": Ymax,
        "di": di,
    }

    try:
        mean_results = compute_fault_and_axial_surfaces(params_mean, y_insar, z_insar)

        if "Y_fault_trace" in mean_results and "Z_fault_trace" in mean_results:
            ax3.plot(mean_results["Y_fault_trace"], mean_results["Z_fault_trace"], color='tab:red', linewidth=1.5,
                     label="Mean fault")

        # Mean axial surfaces (4 surfaces indexed 1–4)
        for i in range(1, 5):
            y_key = f"Y_asurf{i}"
            z_key = f"Z_asurf{i}"
            if y_key in mean_results and z_key in mean_results:
                ax3.plot(mean_results[y_key], mean_results[z_key], 'k--', alpha=0.5, linewidth=0.8,
                         label="Axial surfaces" if i == 1 else "")

        # Mean hinges (4 hinges indexed 1–4)
        for i in range(1, 5):
            y_key = f"Ych{i}"
            z_key = f"Zch{i}"
            if y_key in mean_results and z_key in mean_results:
                ax3.scatter(mean_results[y_key], mean_results[z_key], color='tab:red', s=20,
                            label="Hinges" if i == 1 else "", alpha=0.5)
    except Exception as e:
        print(f"Error computing mean fault: {e}")

    ax3.legend(loc="lower left")
    realisations = cores * (nburn + niter)
    ax3.set_title(f"Posterior fault representation with {realisations} realizations")
    ax3.grid(True, linestyle='--', alpha=0.5)

    # Fix ax3 y-limits: top = topo max + 2 km, bottom = deepest fault point - 2 km
    z3_max = np.nanmax(z_topo) + 1000
    if "Z_fault_trace" in mean_results:
        z3_min = np.nanmin(mean_results["Z_fault_trace"]) - 2000
    else:
        z3_min = -2000
    ax3.set_ylim(z3_min, z3_max)

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
    var_names = ["beta", "teta", "omega", "Y_r2", "Y_r3", "W", "W2", "Smax"]
    traces_dir = os.path.join(output_dir, 'traces')
    os.makedirs(traces_dir, exist_ok=True)
    for var in var_names:
        samples = trace.posterior[var].values.flatten()
        fname = os.path.join(traces_dir, f'{var}.txt')
        np.savetxt(fname, samples, fmt='%.6f')
        print(f"  Saved {var:8s} → {fname}")
    # Save log-likelihood (used as lp proxy for MAP identification).
    # Metropolis does not store lp in sample_stats; compute from the mu Deterministic
    # already saved in the posterior (Gaussian likelihood, sigma_vert=10, sigma_horiz=50).
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
