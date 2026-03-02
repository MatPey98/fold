#!/usr/bin/env python3
# -*- coding:utf-8 -*-

__projet__ = "coupe_pendages_insar"
__nom_fichier__ = "invert_plan.py"
__author__ = "Mathieu Peyrache"
__date__ = "février 2026"

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import getopt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from terrain_profile import Profile
from insar import Insar
from mnt import MNT
from dip import Dip
from seismic import Seismic
from cinematic import compute_fault_and_axial_surfaces

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
    print('Read input file {0}'.format(fname))
    try:
      sys.path.append(path.dirname(path.abspath(fname)))
      exec ("from "+path.basename(fname)+" import *")
    except:
      exec(open(fname).read())

  except Exception as e: 
    print('Problem in input file')
    sys.exit()

# ======================================================================================================================
# Initialiser le profil et l'abscisse
# ======================================================================================================================
profile = Profile(coupe, chemin_coupe, width) # lis la coupe à partir des pts extrèmes et son azimut
profile.linspace(n) # Discrétise la coupe en n points en fonction de sa longueur
abscisses = np.max(profile.abscisse) - profile.abscisse #
x0 = abscisses[0]
xmin = abscisses[0]
xmax = abscisses[-1]

# ======================================================================================================================
# InSAR
# ======================================================================================================================
### vertical
try :
  insar_data_verti = Insar(insar_vertical, chemin_insar, profile)
  abscisses_insar_verti, velocities_verti = insar_data_verti.projection_insar(width)
  abscisses_insar_verti = np.max(abscisses_insar_verti) - abscisses_insar_verti

  # Vérifier qu’il y a bien des données projetées
  if len(abscisses_insar_verti) > 0:
      bin_centers_verti, median_vel_verti, std_vel_verti = insar_data_verti.insar_statistics(width, nbins=120)
      bin_centers_verti = np.max(bin_centers_verti) - bin_centers_verti
  else:
      print("Warning: No projected InSAR data")
      abscisses_insar_verti = np.array([])
      velocities_verti = np.array([])
      bin_centers_verti = np.array([])
      median_vel_verti = np.array([])
      std_vel_verti = np.array([])
except :
  print('Warning: No InSAR data')

### shortening
try :
  insar_data_short = Insar(insar_shortening, chemin_insar, profile)
  abscisses_insar_short, velocities_short = insar_data_short.projection_insar(width)
  abscisses_insar_short = np.max(abscisses_insar_short) - abscisses_insar_short

  # Vérifier qu’il y a bien des données projetées
  if len(abscisses_insar_short) > 0:
      bin_centers_short, median_vel_short, std_vel_short = insar_data_short.insar_statistics(width, nbins=120)
      bin_centers_short = np.max(bin_centers_short) - bin_centers_short
  else:
      print("Warning: No projected InSAR data")
      abscisses_insar_short = np.array([])
      velocities_short = np.array([])
      bin_centers_short = np.array([])
      median_vel_short = np.array([])
      std_vel_short = np.array([])
except :
  print('Warning: No InSAR data')
# ======================================================================================================================
# MNT
# ======================================================================================================================
try :
  topodata = MNT(mnt, mnt_err, chemin_mnt)
  elevations = topodata.elevations(profile.points)
except :
  print('Warning: No elevation data')
# ======================================================================================================================
# strata
# ======================================================================================================================
try :
  directory_strata = chemin_pendages
  dip = Dip(directory_strata)
except:
  print('Warning: No strata data')

# ======================================================================================================================
# fault
# ======================================================================================================================
try:
  directory_fault = chemin_fault
  dip_fault = Dip(directory_fault)
except:
  print('warning: No fault data')

# ======================================================================================================================
# seismic
# ======================================================================================================================
try:
    seismic = Seismic(seismic, chemin_seismic, profile)
    abs_seismic, prof_seismic, mag = seismic.projection_seismic(width_seismic)
    abs_seismic = np.max(abs_seismic) - abs_seismic
except:
    print('Warning: No seimsic data')

# ======================================================================================================================
# cinematic
# ======================================================================================================================
params = {
    "beta": 32,  # Faille raide
    "Ymax": 20000,  

    "teta": 7.5, # Deuxième segment plus plat
    "Ymax2": 14000, 

    "omega": 5,  # Troisième segment presque horizontal
    "Ypref": 5000, 

    "Ymin": -3000,  # Début du calcul de la faille

    "Zdec": -1307, # Profondeur de référence
    "ldec": 20000, # Longueur de référence pour le calcul des rampes. Utilisé pour le calcul des intersections entre rampes.

    "sigma": -1, # Angle de cisaillement (en degrés)
    
    "W": 7000,     # Largeur de la première charnière
    "W2": 3000,    # Largeur de la deuxième charnière
    "Zhaut": 1800, # Limite supérieure des surfaces axiales

    "n_strata": 20, # Nombre de strates
    "di": 4000, # Pas de discrétisation
    "ite_s": 1, # Nombre d'itérations
    "Y_topo": abscisses,  # Vos données topo
    "Z_topo": elevations,  # Vos données topo
    "Y_insar": abscisses_insar_verti,  # Vos données InSAR
    "Z_insar": velocities_verti,  # Vos données InSAR
    "Smax": 2.5, # raccourcissement
    "n_tot": 1
}

results = compute_fault_and_axial_surfaces(params) # Kinematic model results

# ======================================================================================================================
# Plot
# ======================================================================================================================

fig = plt.figure(layout="constrained", figsize=(8, 8))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2], hspace=0.1)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

### Upper plot
# Topo
ax1.plot(abscisses, elevations, color="black")
ax1.set_xlim(xmin, xmax)
ax1.set_xlabel("Distance (m)")
ax1.set_ylabel("Altitude (m)")
ax1.set_ylim([3200, 4200])
ax1.tick_params(axis='y', labelcolor='k')

# InSAR
ax1b = ax1.twinx()
ax1b.scatter(abscisses_insar_verti, velocities_verti, s=2, alpha=0.1, label="InSAR vertical velocities")
ax1b.plot(bin_centers_verti, median_vel_verti, color="dodgerblue", linewidth=1, alpha=0.5, label="Median")
ax1b.fill_between(bin_centers_verti, median_vel_verti - std_vel_verti, median_vel_verti + std_vel_verti, color="dodgerblue", alpha=0.2, label="±1 std")

ax1b.scatter(abscisses_insar_short, velocities_short, s=2, alpha=0.1, label="InSAR vertical velocities")
ax1b.plot(bin_centers_short, median_vel_short, color="coral", linewidth=1, alpha=0.5, label="Median")
ax1b.fill_between(bin_centers_short, median_vel_short - std_vel_short, median_vel_short + std_vel_short, color="coral", alpha=0.2, label="±1 std")

ax1b.set_ylabel("Velocity", color ='r')
ax1b.set_xlim(xmin, xmax)
ax1b.set_ylim(-6, 6)
ax1b.legend(loc="upper right")

# Cinematic
ax1b.plot(results["Y_save"] - 1640, results["Z_save"] - 3307, '-b', label='Déformation calculée')

### Lower plot
# Topo
ax2.plot(abscisses, elevations, color="black")
ax2.set_xlabel('Distance horizontale (m)')
ax2.set_ylabel('Profondeur (m)')

# Strata
dip.print_all(topodata, profile, length_dip, ax2)

# Fault
dip_fault.print_all_fault(topodata, profile, length_dip_fault, ax2)

# Seismic
sc2 = ax2.scatter(abs_seismic, -np.array(prof_seismic) * 1000, c=mag, cmap="YlOrRd", edgecolor="k", s=50)
cax = inset_axes(ax2, width="3%", height="30%", loc="lower left", borderpad=1)
cbar = fig.colorbar(sc2, cax=cax)
cbar.set_label("Magnitude")

# Cinematic
ax2.plot(results["Yfaille"] - 1640, results["Zfaille"], '-r', label='Faille')
ax2.legend(loc="upper right")
ax2.grid(True)
ax2.legend()

ax2.set_xlim([xmin, xmax])

# ======================================================================================================================
# Save figure
# ======================================================================================================================
output_dir = "/data/scratch/mathieu/qaidam/output_fold_qaidam/"

os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir,"coupe3_figure1.png")

plt.savefig(output_path, dpi=300, bbox_inches='tight')

print(f'figure sauvegardé dans : {output_path}')

plt.show()