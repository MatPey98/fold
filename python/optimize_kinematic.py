#!/usr/bin/env python3
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

# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=RuntimeWarning)


# ======================================================================================================================
# Paramètres du profil
# ======================================================================================================================
width = 100 # Largeur du profil
width_seismic = 10000 # Largeur des projections de mécanisme au foyer sur le profil
n = 1000 # Echantillonnage des points le long du profil (résolution du profil)
nbins = 50 # Pas d'interpolation pour la médiane des points InSAR

# ======================================================================================================================
# Paramètres MCMC
# ======================================================================================================================
niter = 1  # Nombre total d'itérations
nburn = 1   # Nombre d'itérations de "burn-in"
n_samples = 4  # Nombre de réalisations à afficher
chains = 4     # Nombre de chaînes
cores = 4      # Nombre de cœurs du CPU utilisés

# ======================================================================================================================
# Paramètres aprioris : U = [lower, upper] ou N = [mu, sigma]
# ======================================================================================================================
Ubeta = [30,50]
Uteta = [7.5, 29]
Uomega = [1, 7]
# NY_r2 = [16000, 2000]
UY_r2 = [11000, 17000]
UY_r3 = [1000, 10000]
UW = [1000, 4000]
UW2 = [1000, 4000]
USmax = [10, 100]

# ======================================================================================================================
# Paramètres fixes
# ======================================================================================================================
Y_faille = 18520 # Emplacement de la faille en surface
Z_faille = 3430 # Emplacement de la faille en surface
Ymin = 1000 # Calcul de la faille
Ymax = 17800 # Calcul de la faille
Ymax2 = 14000
Zhaut = 1800 # Hauteur des charnières
n_strata = 20
di = 4000
ite_s = 1
n_tot = 1

# ======================================================================================================================
# Data
# ======================================================================================================================
wdir = '/data/scratch/mathieu/qaidam/'
### Profile :
coupe = "coupe9.shp"
chemin_coupe = wdir + "/carto/profile/"

### insar :
insar_vertical = "vertical_2003-2011_mm_crop03_UTM.tif"
chemin_insar = wdir + "/data/insar/decomp2026/"

### Mnt :
mnt = "cop_dem30_92_101_34_40_crop_UTM_v2.tif"
mnt_err = "DSM_triangulation_errors_merged.tif"
chemin_mnt = wdir + "/data/DEM/"



# ======================================================================================================================
# INPUT PARAMÈTRES : optimize_kinematic.py input_file.py
# ======================================================================================================================
def usage():
  print('invert_plan.py infile.py [-h]')
  print('-h Show this screen')

#load input file
try:
    opts,args = getopt.getopt(sys.argv[1:], "h", ["help"])
except:
    print("for help use --help")
    sys.exit()

for o in sys.argv:
    if o in ("-h","--help"):
       usage()
       sys.exit()

if 1==len(sys.argv):
  usage()
  assert False, "no input file"
  print('No input file')
  sys.exit()

fname=sys.argv[1]
exec(open(fname).read())
if len(sys.argv)>1:
  try:
    fname=sys.argv[1]
    print('Read input file {0} '.format(fname))
    try:
      sys.path.append(path.dirname(path.abspath(fname)))
      exec ("from "+path.basename(fname)+" import *")
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
    insar_data_verti = Insar(insar_vertical, chemin_insar, profile)
    abscisses_insar_verti, velocities_verti = insar_data_verti.projection_insar(width)
    abscisses_insar_verti = np.max(abscisses_insar_verti) - abscisses_insar_verti
    mask = (abscisses_insar_verti < Ymax) & (abscisses_insar_verti > Ymin)
    y_insar_filtered = abscisses_insar_verti[mask]
    z_insar_filtered = velocities_verti[mask]
    y_insar = abscisses_insar_verti
    z_insar = velocities_verti
    return y_insar_filtered, z_insar_filtered, z_insar, y_insar

# Initialiser les données une seule fois au lieu de faire le calcul à chaque itérations
y_insar_filtered, z_insar_filtered, z_insar, y_insar = load_insar_data()

def data():
    return z_insar_filtered

def Cov():
    sigmad = np.ones_like(z_insar_filtered) * 10
    return np.diag(1.0 / sigmad**2)

# Seismic --------------------------------------------------------------------------------------------------------------
try:
    seismic = Seismic(seismic, chemin_seismic, profile)
    abs_seismic, prof_seismic, mag, rms_values = seismic.projection_seismic(width_seismic)
    abs_seismic = np.max(abs_seismic) - abs_seismic
except:
    print('Warning: No seimsic data')

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
    }

    # print(f"Paramètres d'entrée : beta={beta}, teta={teta}, omega={omega}, Y_r2={Y_r2}, Y_r3={Y_r3}, W={W}, W2={W2}, Smax={Smax}")

    results = compute_fault_and_axial_surfaces(params)

    # print(f"Résultats de compute_fault_and_axial_surfaces : Y_save={results['Y_save']}, Z_save={results['Z_save']}")

    y_insar_filtered, _, _,_ = load_insar_data()
    Z_interp = np.interp(y_insar_filtered, results["Y_save"], results["Z_save"] - 3307)

    # print(f"Z_interp : {Z_interp}")

    if np.any(np.isnan(Z_interp)) or np.any(np.isinf(Z_interp)):
        print("Z_interp contient des NaN ou des inf, utilisation de valeurs aléatoires")
        Z_interp = np.random.normal(0, 1e-3, size=len(Z_interp))
    return Z_interp

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
            mu = forward_model(beta, teta, omega, Y_r2, Y_r3, W, W2, Smax)
            if np.any(np.isnan(mu)) or np.any(np.isinf(mu)):
                raise ValueError("forward_model a retourné des NaN ou des inf.")
            outputs[0][0] = np.asarray(mu, dtype=np.float64)
        except Exception as e:
            print(f"Erreur dans ForwardModelOp: {e}")
            y_insar_filtered, _, _, _ = load_insar_data()
            outputs[0][0] = np.ones_like(y_insar_filtered, dtype=np.float64) * 1e6

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
        # Y_r2 = pm.Normal("Y_r2", mu=NY_r2[0], sigma=NY_r2[1])
        Y_r3 = pm.Uniform("Y_r3", lower=UY_r3[0], upper=UY_r3[1])
        W = pm.Uniform("W", lower=UW[0], upper=UW[1])
        W2 = pm.Uniform("W2", lower=UW2[0], upper=UW2[1])
        Smax = pm.Uniform("Smax", lower=USmax[0], upper=USmax[1])

        # Empiler les paramètres dans un vecteur theta
        theta = pt.stack([beta, teta, omega, Y_r2, Y_r3, W, W2, Smax])
        mu = forward_op(theta)
        mu = pm.Deterministic("mu", mu)

        d_obs = data()
        sigma = np.sqrt(1.0 / np.diag(Cov()))

        pm.Normal("InSAR_Data", mu=mu, sigma=sigma, observed=d_obs)

        # timer = chains*nburn+niter*cores
        # with tqdm(total=timer, desc="Sampling") as pbar:
        trace = pm.sample(
            draws=niter,
            tune=nburn,
            chains=chains,
            cores=cores,
            step=pm.Metropolis(),
            progressbar=True,
            # callback=lambda trace, draw: pbar.update(1)
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

    # Création des figures
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

# Graphiques des traces et posteriors ----------------------------------------------------------------------------------
    try:
        az.plot_trace(trace, var_names=var_names, compact=True, figsize=(5, 3), combined=True)
        az.plot_posterior(trace, var_names=var_names, kind='hist',textsize=5, figsize=(5, 3))
        az.plot_pair(trace, var_names=var_names,kind='hexbin', marginals=True, textsize=5,figsize=(5, 3))
        az.plot_forest(trace, var_names=["beta", "teta", "omega", "Smax"],combined=True, hdi_prob=0.95, textsize=5, linewidth=1, markersize=2, figsize=(5, 3))
        az.plot_forest(trace, var_names=["Y_r2", "Y_r3", "W", "W2"],combined=True, hdi_prob=0.95, textsize=5, linewidth=1, markersize=2, figsize=(5, 3))
    except:
        print("arviz shut down")

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1b = ax1.twinx()
    # Sous-graphique 1 : Comparaison données InSAR vs prédictions du modèle
    # y_insar_filtered, z_insar_filtered, z_insar, y_insar = load_insar_data()
    posterior_predictions = trace.posterior["mu"].mean(dim=["chain", "draw"]).values

    # Topo
    # ax1.plot(y_topo, z_topo, 'k-', linewidth=1,  label="Topographie")
    ax1.plot(y_topo, z_topo, 'k-', linewidth=1, label="Topographie")
    ax1.set_title("Comparaison données InSAR vs prédictions du modèle")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_ylabel("Altitude (m)")

    # InSAR et prédictions postérieures
    ax1b.scatter(y_insar, z_insar, s=2, alpha=0.5, color='tab:blue', label="Elévation verticale (InSAR)")
    ax1b.plot(y_insar_filtered, posterior_predictions, color='tab:red', linewidth=1.5, label="Prédictions moyennes du modèle")
    ax1b.set_xlabel("Position (m)")
    ax1b.set_ylabel("Déformation (mm)")
    ax1b.legend(loc="upper right")

    # Sous-graphique 2 : Représentation de la faille à postériori
    # Tracer la topographie
    # ax2.plot(y_topo, z_topo, 'k-', label='Topographie')
    ax2.plot(y_topo, z_topo, 'k-', label='Topographie')

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

    # Sélectionner quelques échantillons aléatoires pour visualiser plusieurs réalisations
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
                sample_results = compute_fault_and_axial_surfaces(params_sample)

                # Tracer la faille pour chaque échantillon
                if "Yfaille" in sample_results and "Zfaille" in sample_results:
                    ax2.plot(sample_results["Yfaille"], sample_results["Zfaille"], 'tab:red', alpha=0.1, linewidth=0.5)

            #     # Tracer les surfaces axiales pour chaque échantillon
            #     for i in range(1, 5):
            #         y_key = f"Y_asurf{i}"
            #         z_key = f"Z_asurf{i}"
            #         if y_key in sample_results and z_key in sample_results:
            #             ax2.plot(sample_results[y_key], sample_results[z_key], '--k', alpha=0.1)
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
        mean_results = compute_fault_and_axial_surfaces(params_mean)

        # Tracer la faille moyenne en gras
        if "Yfaille" in mean_results and "Zfaille" in mean_results:
            ax2.plot(mean_results["Yfaille"], mean_results["Zfaille"], color='tab:red', linewidth=1.5, label="Faille moyenne")

        # Tracer les surfaces axiales moyennes
        for i in range(1, n_samples):
            y_key = f"Y_asurf{i}"
            z_key = f"Z_asurf{i}"
            if y_key in mean_results and z_key in mean_results:
                ax2.plot(mean_results[y_key], mean_results[z_key], 'k--', alpha=0.5, linewidth=0.8, label="Surfaces axiales" if i == 1 else "")

        # Tracer les charnières moyennes
        for i in range(1, n_samples):
            y_key = f"Ych{i}"
            z_key = f"Zch{i}"
            if y_key in mean_results and z_key in mean_results:
                ax2.scatter(mean_results[y_key], mean_results[z_key], color='tab:red', s=20, label="Charnières" if i == 1 else "", alpha=0.5)
    except Exception as e:
        print(f"Erreur lors du calcul de la faille moyenne: {e}")

    ax2.set_xlabel("Position sur le profil (m)")
    ax2.set_ylabel("Profondeur (m)")
    ax2.legend(loc="lower left")
    ax2.set_title(f"Représentation de la faille à postériori avec {n_samples} réalisations")
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Lim topo
    ax1.set_xlim(min(y_insar), max(y_insar))
    ax1.set_ylim(min(z_topo) - 200, max(z_topo) + 200)

    # Lim déformations
    ax1b.set_xlim(min(y_insar), max(y_insar))
    ax1b.set_ylim(min(z_insar)-10, max(z_insar)+10)

    # fig 2
    ax2.set_xlim(min(y_insar), max(y_insar))
    ax1.invert_xaxis()
    ax2.invert_xaxis()
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
