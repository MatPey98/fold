# !/usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import warnings
import pytensor
import pytensor.tensor as pt
import sys
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

    # load input file


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
    print('No input file')
    sys.exit()

fname = sys.argv[1]
exec(open(fname).read())
if len(sys.argv) > 1:
    try:
        fname = sys.argv[1]
        print('Read input file {0} '.format(fname))
        try:
            sys.path.append(path.dirname(path.abspath(fname)))
            exec("from " + path.basename(fname) + " import *")
        except:
            exec(open(fname).read())

    except Exception as e:
        print('Problem in input file')
        sys.exit()

# ======================================================================================================================
# LECTURE DES DONNÉES
# ======================================================================================================================
# Profil ---------------------------------------------------------------------------------------------------------------
profile = Profile(coupe, chemin_coupe, width)  # lis la coupe à partir des pts extrèmes et son azimut
profile.linspace(n)  # Discrétise la coupe en n points en fonction de sa longueur
y_topo = np.max(profile.abscisse) - profile.abscisse  #

# Topo -----------------------------------------------------------------------------------------------------------------
try:
    topodata = MNT(mnt, mnt_err, chemin_mnt)
    z_topo = topodata.elevations(profile.points)
except:
    print('Warning: No elevation data')


# InSAR ----------------------------------------------------------------------------------------------------------------
def load_insar_data():
    # vertical
    insar_data_verti = Insar(insar_vertical, chemin_insar, profile)
    abscisses_insar_verti, velocities_verti = insar_data_verti.projection_insar(width)
    abscisses_insar_verti = np.max(abscisses_insar_verti) - abscisses_insar_verti
    mask = (abscisses_insar_verti < Ymax) & (abscisses_insar_verti > Ymin)
    y_insar_filtered = abscisses_insar_verti[mask]
    z_insar_filtered = velocities_verti[mask]
    y_insar = abscisses_insar_verti
    z_insar = velocities_verti

    # Données horizontales (raccourcissement)
    insar_data_horiz = Insar(insar_horizontal, chemin_insar, profile)
    abscisses_insar_horiz, velocities_horiz = insar_data_horiz.projection_insar(width)
    y_insar_short = np.max(abscisses_insar_horiz) - abscisses_insar_horiz
    z_insar_short = velocities_horiz
    y_insar_short_filtered = y_insar_short[mask]
    z_insar_short_filtered = z_insar_short[mask]

    return y_insar_filtered, z_insar_filtered, z_insar, y_insar, z_insar_short, y_insar_short, z_insar_short_filtered, y_insar_short_filtered


# Initialiser les données une seule fois au lieu de faire le calcul à chaque itération
y_insar_filtered, z_insar_filtered, z_insar, y_insar, z_insar_short, y_insar_short, z_insar_short_filtered, y_insar_short_filtered = load_insar_data()


def data():
    return z_insar_filtered, z_insar_short_filtered


def Cov():
    sigmad = np.ones_like(z_insar_filtered) * 10
    return np.diag(1.0 / sigmad ** 2)

# ======================================================================================================================
# MODÈLE DIRECT (forward_model)
# ======================================================================================================================
def forward_model(beta, teta, omega, Y_r2, Y_r3, W, W2, Smax):
    """
    Fonction pour calculer la déformation attendue avec des valeurs numériques.
    """
    params = {
        "beta": beta,
        "teta": teta,
        "omega": omega,
        "Y_r2": Y_r2,
        "Y_r3": Y_r3,
        "W": W,
        "W2": W2,
        "Smax": Smax,
        "Y_faille": Y_faille,
        "Z_faille": Z_faille,
        "Ymin": Ymin,
        "Ymax": Ymax,
        "Ymax2": Ymax2,
        "Zhaut": Zhaut,
        "n_strata": n_strata,
        "di": di,
        "ite_s": ite_s,
        "n_tot": n_tot,
        "Y_topo": y_topo,
        "Z_topo": z_topo

    }

    # print(f"Paramètres d'entrée : beta={beta}, teta={teta}, omega={omega}, Y_r2={Y_r2}, Y_r3={Y_r3}, W={W}, W2={W2}, Smax={Smax}")

    results = compute_fault_and_axial_surfaces(params, y_topo, z_topo, y_insar, z_insar)

    Ych2 = results["Ych2"]
    Ych3 = results["Ych3"]

    Z_interp = np.interp(y_insar_filtered, results["Y_save"], results["Z_save"] - 3307)
    raccourcissement_interp = np.interp(y_insar_filtered, results["Y_save"], results["raccourcissement_horiz"])

    if np.any(np.isnan(Z_interp)) or np.any(np.isinf(Z_interp)) or np.any(Ych2 < Ych3) or np.any(np.isnan(Y_r2)):
        # print("Z_interp contient des NaN ou des inf, utilisation de valeurs aléatoires")
        Z_interp = np.random.normal(0, 1e-3, size=len(Z_interp))
    if np.any(np.isnan(raccourcissement_interp)) or np.any(np.isinf(raccourcissement_interp)) or np.any(Ych2 < Ych3) or np.any(np.isnan(Y_r2)):
        # print("raccourcissement_interp contient des NaN ou des inf, utilisation de valeurs aléatoires")
        raccourcissement_interp = np.random.normal(0, 1e-3, size=len(raccourcissement_interp))
    return Z_interp, raccourcissement_interp


# ======================================================================================================================
# OPÉRATEUR PYTENSOR (ForwardModelOp)
# ======================================================================================================================
class ForwardModelOp(pytensor.graph.op.Op):
    itypes = [pt.dvector]
    otypes = [pt.dvector]

    def perform(self, node, inputs, outputs):
        theta = inputs[0]
        try:
            theta_values = [float(val) for val in theta]
            beta, teta, omega, Y_r2, Y_r3, W, W2, Smax = theta_values
            z_vert, z_horiz = forward_model(beta, teta, omega, Y_r2, Y_r3, W, W2, Smax)
            if np.any(np.isnan(z_vert)) or np.any(np.isinf(z_vert)) or np.any(np.isnan(z_horiz)) or np.any(np.isinf(z_horiz)):
                raise ValueError("forward_model a retourné des NaN ou des inf.")
            outputs[0][0] = np.concatenate([z_vert, z_horiz])
        except Exception as e:
            print(f"Erreur dans ForwardModelOp: {e}")
            outputs[0][0] = np.ones(2 * len(y_insar_filtered)) * 1e6


forward_op = ForwardModelOp()


# ======================================================================================================================
# INFÉRENCE BAYÉSIENNE
# ======================================================================================================================
def run_inversion():
    with pm.Model() as model:
        # Définition des priors à l'intérieur du contexte du modèle
        beta = pm.Uniform("beta", lower=Ubeta[0], upper=Ubeta[1])
        teta = pm.Uniform("teta", lower=Uteta[0], upper=Uteta[1])
        omega = pm.Uniform("omega", lower=Uomega[0], upper=Uomega[1])
        Y_r2 = pm.Uniform("Y_r2", lower=UY_r2[0], upper=UY_r2[1])
        Y_r3 = pm.Uniform("Y_r3", lower=UY_r3[0], upper=UY_r3[1])
        W = pm.Uniform("W", lower=UW[0], upper=UW[1])
        W2 = pm.Uniform("W2", lower=UW2[0], upper=UW2[1])
        Smax = pm.Uniform("Smax", lower=USmax[0], upper=USmax[1])

        # Empiler les paramètres dans un vecteur theta
        theta = pt.stack([beta, teta, omega, Y_r2, Y_r3, W, W2, Smax])
        mu = forward_op(theta)
        mu_vert = mu[:len(y_insar_filtered)]
        mu_horiz = mu[len(y_insar_filtered):]
        mu = pm.Deterministic("mu", mu)

        d_obs_vert, d_obs_horiz = data()
        #sigma_vert = np.sqrt(1.0 / np.diag(Cov()))
        #sigma_horiz = np.ones_like(d_obs_horiz) * 1.0
        sigma_vert = 10
        sigma_horiz = 50

        pm.Normal("InSAR_Vertical", mu=mu_vert, sigma=sigma_vert, observed=d_obs_vert)
        pm.Normal("InSAR_Horizontal", mu=mu_horiz, sigma=sigma_horiz, observed=d_obs_horiz)

        # Filtrer les paramètres abbérants
        pm.Potential("invalid_parameters",
                     pm.math.switch(
                         (beta < teta) | (teta < omega) | (Y_r2 < Y_r3), -1e6, 0)
                     )

        # Sampling de l'inférence Bayésienne
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
# VISUALISATION
# ======================================================================================================================
def plot_results(trace):
    plt.style.use("seaborn-v0_8-deep")
    az.style.use("default")
    plt.rcParams["figure.dpi"] = 150  # affichage écran
    plt.rcParams["savefig.dpi"] = 300  # export haute qualité

    # Résumé des paramètres
    var_names = ["beta", "teta", "omega", "Y_r2", "Y_r3", "W", "W2", "Smax"]

    # Filtrer les valeurs NaN
    filtered_trace = {k: v for k, v in trace.posterior.items() if not np.any(np.isnan(v))}

    summary = az.summary(filtered_trace, var_names=var_names)
    print("\nRésumé des paramètres postérieurs :")
    print(summary)

    # Graphiques des traces et posteriors ----------------------------------------------------------------------------------
    plt.rcParams.update({'font.size': 6})
    try:
        az.plot_trace(trace, var_names=var_names, compact=True, figsize=(5, 3), combined=True)
        az.plot_posterior(trace, var_names=var_names, kind='hist', textsize=6, figsize=(5, 3))
        az.plot_pair(trace, var_names=var_names, kind='hexbin', marginals=True, textsize=6, figsize=(5, 3))
        az.plot_forest(trace, var_names=["beta", "teta", "omega", "Smax"], combined=True, hdi_prob=0.95, textsize=6,
                       linewidth=1, markersize=2, figsize=(5, 3))
        az.plot_forest(trace, var_names=["Y_r2", "Y_r3", "W", "W2"], combined=True, hdi_prob=0.95, textsize=6,
                       linewidth=1, markersize=2, figsize=(5, 3))
    except:
        print("arviz shut down")

    # mise en forme figure du modèle
    plt.rcParams.update({
        # Taille & résolution
        "figure.figsize": (6, 4),
        "figure.dpi": 150,
        "savefig.dpi": 300,

        # Police
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,

        # Lignes
        "lines.linewidth": 1.2,

        # Axes
        "axes.linewidth": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,

        # Grille discrète
        "grid.linestyle": "--",
        "grid.alpha": 0.1
    })

    # Résultats : déplacements verticaux et horizontaux
    mu_all = trace.posterior["mu"].mean(dim=["chain", "draw"]).values
    mu_vert = mu_all[:len(y_insar_filtered)]
    mu_horiz = mu_all[len(y_insar_filtered):]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot 1 -----------------------------------------------------------------------------------------------------------
    # Sous-graphique 1 : Comparaison des données InSAR avec les prédictions du modèle
    # Topo
    ax1.set_title("Comparaison données InSAR vs prédictions du modèle")
    ax1.plot(y_topo, z_topo, 'k-', linewidth=1, label="Topographie")
    ax1.set_ylabel("Altitude (m)")
    ax1.legend(loc="upper left")

    # Insar
    ax1b = ax1.twinx()
    ax1b.scatter(y_insar, z_insar, s=2, alpha=0.5, color='tab:blue', label="Soulèvement vertical InSAR")
    
    # Vertical
    ax1b.plot(y_insar_filtered, mu_vert, color='tab:red', linewidth=1.5, label="Prédictions moyennes du modèle")###
    ax1b.set_ylabel("Déformation verticale (mm)")
    ax1b.grid(True, linestyle='--', alpha=0.1)
    ax1b.legend(loc="upper right")

    # Horizontal

    # Plot 2 -----------------------------------------------------------------------------------------------------------
    # Topo
    ax2.plot(y_topo, z_topo, 'k-', linewidth=1, label="Topographie")
    ax2.set_ylabel("Altitude (m)")
    ax2.legend(loc="upper left")

    ax2b = ax2.twinx()
    ax2b.scatter(y_insar_short, z_insar_short, s=2, alpha=0.5, color='tab:orange', label="Raccourcissement N022° InSAR")###
    ax2b.plot(y_insar_filtered, mu_horiz, color='tab:green', linewidth=1.5, label="Prédictions moyennes du modèle")###
    ax2b.grid(True, linestyle='--', alpha=0.1)
    ax2b.set_ylabel("Raccourcissement (mm)")
    ax2b.legend(loc="upper right")

    # Plot 3 -----------------------------------------------------------------------------------------------------------
    # topo
    ax3.plot(y_topo, z_topo, 'k-', label='Topographie')
    ax3.set_ylabel("Profondeur (m)")

    # Récupérer les échantillons postérieurs des paramètres
    beta_samples = trace.posterior["beta"].values.flatten()
    teta_samples = trace.posterior["teta"].values.flatten()
    omega_samples = trace.posterior["omega"].values.flatten()
    Y_r2_samples = trace.posterior["Y_r2"].values.flatten()
    Y_r3_samples = trace.posterior["Y_r3"].values.flatten()
    W_samples = trace.posterior["W"].values.flatten()
    W2_samples = trace.posterior["W2"].values.flatten()
    Smax_samples = trace.posterior["Smax"].values.flatten()

    # Filtrer les valeurs NaN
    valid_indices = ~np.isnan(beta_samples) & ~np.isnan(teta_samples) & ~np.isnan(omega_samples) & \
                    ~np.isnan(Y_r2_samples) & ~np.isnan(Y_r3_samples) & ~np.isnan(W_samples) & \
                    ~np.isnan(W2_samples) & ~np.isnan(Smax_samples)

    beta_samples = beta_samples[valid_indices]
    teta_samples = teta_samples[valid_indices]
    omega_samples = omega_samples[valid_indices]
    Y_r2_samples = Y_r2_samples[valid_indices]
    Y_r3_samples = Y_r3_samples[valid_indices]
    W_samples = W_samples[valid_indices]
    W2_samples = W2_samples[valid_indices]
    Smax_samples = Smax_samples[valid_indices]

    # Sélectionner des échantillons aléatoires pour visualiser plusieurs réalisations
    if len(beta_samples) > 0:
        num_samples = min(n_samples, len(beta_samples))
        sample_indices = np.random.choice(len(beta_samples), num_samples, replace=False)

        for idx in sample_indices:
            params_sample = {
                "beta": beta_samples[idx],
                "teta": teta_samples[idx],
                "omega": omega_samples[idx],
                "Y_r2": Y_r2_samples[idx],
                "Y_r3": Y_r3_samples[idx],
                "W": W_samples[idx],
                "W2": W2_samples[idx],
                "Smax": Smax_samples[idx],
                "Y_faille": Y_faille,
                "Z_faille": Z_faille,
                "Ymin": Ymin,
                "Ymax": Ymax,
                "Ymax2": Ymax2,
                "Zhaut": Zhaut,
                "n_strata": n_strata,
                "di": di,
                "ite_s": ite_s,
                "n_tot": n_tot,
            }

            try:
                sample_results = compute_fault_and_axial_surfaces(params_sample, y_topo, z_topo, y_insar, z_insar)
                # Tracer la faille pour chaque échantillon
                if "Yfaille" in sample_results and "Zfaille" in sample_results:
                    ax3.plot(sample_results["Yfaille"], sample_results["Zfaille"], 'tab:red', alpha=0.1, linewidth=0.5)
            except Exception as e:
                print(f"Erreur lors du calcul de la faille pour l'échantillon {idx}: {e}")

    # Tracer la faille moyenne
    beta_mean = np.nanmean(beta_samples)
    teta_mean = np.nanmean(teta_samples)
    omega_mean = np.nanmean(omega_samples)
    Y_r2_mean = np.nanmean(Y_r2_samples)
    Y_r3_mean = np.nanmean(Y_r3_samples)
    W_mean = np.nanmean(W_samples)
    W2_mean = np.nanmean(W2_samples)
    Smax_mean = np.nanmean(Smax_samples)

    params_mean = {
        "beta": beta_mean,
        "teta": teta_mean,
        "omega": omega_mean,
        "Y_r2": Y_r2_mean,
        "Y_r3": Y_r3_mean,
        "W": W_mean,
        "W2": W2_mean,
        "Smax": Smax_mean,
        "Y_faille": Y_faille,
        "Z_faille": Z_faille,
        "Ymin": Ymin,
        "Ymax": Ymax,
        "Ymax2": Ymax2,
        "Zhaut": Zhaut,
        "n_strata": n_strata,
        "di": di,
        "ite_s": ite_s,
        "n_tot": n_tot,
    }

    try:
        mean_results = compute_fault_and_axial_surfaces(params_mean, y_topo, z_topo, y_insar, z_insar)

        # Tracer la faille moyenne en gras
        if "Yfaille" in mean_results and "Zfaille" in mean_results:
            ax3.plot(mean_results["Yfaille"], mean_results["Zfaille"], color='tab:red', linewidth=1.5,
                     label="Faille moyenne")

        # Tracer les surfaces axiales moyennes
        for i in range(1, n_samples):
            y_key = f"Y_asurf{i}"
            z_key = f"Z_asurf{i}"
            if y_key in mean_results and z_key in mean_results:
                ax3.plot(mean_results[y_key], mean_results[z_key], 'k--', alpha=0.5, linewidth=0.8,
                         label="Surfaces axiales" if i == 1 else "")

        # Tracer les charnières moyennes
        for i in range(1, n_samples):
            y_key = f"Ych{i}"
            z_key = f"Zch{i}"
            if y_key in mean_results and z_key in mean_results:
                ax3.scatter(mean_results[y_key], mean_results[z_key], color='tab:red', s=20,
                            label="Charnières" if i == 1 else "", alpha=0.5)
    except Exception as e:
        print(f"Erreur lors du calcul de la faille moyenne: {e}")

    ax3.legend(loc="lower left")
    realisations = cores * (nburn + niter)
    ax3.set_title(f"Représentation de la faille à postériori avec {realisations} réalisations")
    ax3.grid(True, linestyle='--', alpha=0.5)

    # Limites des axes
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(min(y_insar), max(y_insar))
        ax.invert_xaxis()

    fig.tight_layout(pad=1.0)
    plt.show()

# ======================================================================================================================
# EXÉCUTION
# ======================================================================================================================
if __name__ == "__main__":
    print("###################################################################")
    print("#        Inversion Bayésienne pour le modèle cinématique          #")
    print("###################################################################")
    print("\nLancement de l'inférence bayésienne :")
    model, trace = run_inversion()
    plot_results(trace)
