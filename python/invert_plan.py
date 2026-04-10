#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import sys
from os import path
import getopt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from profile import Profile
from read_data import Insar, MNT, Seismic
from dip import Dip

# ======================================================================================================================
# Donnée entrée
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
profile = Profile(coupe, chemin_coupe, width)  # lis la coupe à partir des pts extrèmes et son azimut
profile.linspace(n)  # Discrétise la coupe en n points en fonction de sa longueur
abscisses = np.max(profile.abscisse) - profile.abscisse  #
x0 = abscisses[0]
xmin = abscisses[0]
xmax = abscisses[-1]

# ======================================================================================================================
# InSAR
# ======================================================================================================================
### vertical 2003-2011
try:
    insar_data_verti_0311 = Insar(insar_vertical_0311, chemin_insar, profile)
    abscisses_insar_verti_0311, velocities_verti_0311 = insar_data_verti_0311.projection_insar(width)
    abscisses_insar_verti_0311 = np.max(abscisses_insar_verti_0311) - abscisses_insar_verti_0311

    # Vérifier qu’il y a bien des données projetées
    if len(abscisses_insar_verti_0311) > 0:
        bin_centers_verti_0311, median_vel_verti_0311, std_vel_verti_0311 = insar_data_verti_0311.insar_statistics(width, nbins=120)
        bin_centers_verti_0311 = np.max(bin_centers_verti_0311) - bin_centers_verti_0311
    else:
        print("Warning: No projected InSAR data")
except:
    print('Warning: No InSAR data')

### Vertical 2011-2019
try:
    insar_data_verti_1119 = Insar(insar_vertical_1119, chemin_insar, profile)
    abscisses_insar_verti_1119, velocities_verti_1119 = insar_data_verti_1119.projection_insar(width)
    abscisses_insar_verti_1119 = np.max(abscisses_insar_verti_1119) - abscisses_insar_verti_1119

    # Vérifier qu’il y a bien des données projetées
    if len(abscisses_insar_verti_1119) > 0:
        bin_centers_verti_1119, median_vel_verti_1119, std_vel_verti_1119 = insar_data_verti_1119.insar_statistics(width, nbins=120)
        bin_centers_verti_1119 = np.max(bin_centers_verti_1119) - bin_centers_verti_1119
    else:
        print("Warning: No projected InSAR data")
except:
    print('Warning: No InSAR data')

### Vertical 2003-2019
try:
    insar_data_verti_0319 = Insar(insar_vertical_0319, chemin_insar, profile)
    abscisses_insar_verti_0319, velocities_verti_0319 = insar_data_verti_0319.projection_insar(width)
    abscisses_insar_verti_0319 = np.max(abscisses_insar_verti_0319) - abscisses_insar_verti_0319

    # Vérifier qu’il y a bien des données projetées
    if len(abscisses_insar_verti_0319) > 0:
        bin_centers_verti_0319, median_vel_verti_0319, std_vel_verti_0319 = insar_data_verti_0319.insar_statistics(width, nbins=120)
        bin_centers_verti_0319 = np.max(bin_centers_verti_0319) - bin_centers_verti_0319
    else:
        print("Warning: No projected InSAR data")
except:
    print('Warning: No InSAR data')

### Cosismique 2004 
try:
    insar_data_LOS_co2004 = Insar(insar_LOS_co2004, chemin_insar, profile)
    abscisses_insar_LOS_co2004, velocities_LOS_co2004 = insar_data_LOS_co2004.projection_insar(width)
    abscisses_insar_LOS_co2004 = np.max(abscisses_insar_LOS_co2004) - abscisses_insar_LOS_co2004

    # Vérifier qu’il y a bien des données projetées
    if len(abscisses_insar_LOS_co2004) > 0:
        bin_centers_LOS_co2004, median_vel_LOS_co2004, std_vel_LOS_co2004 = insar_data_LOS_co2004.insar_statistics(width, nbins=120)
        bin_centers_LOS_co2004 = np.max(bin_centers_LOS_co2004) - bin_centers_LOS_co2004
    else:
        print("Warning: No projected InSAR data")
except:
    print('Warning: No InSAR data')

### Shortening 2003-2011
try:
    insar_data_short_0311 = Insar(insar_short_0311, chemin_insar, profile)
    abscisses_insar_short_0311, velocities_short_0311 = insar_data_short_0311.projection_insar(width)
    abscisses_insar_short_0311 = np.max(abscisses_insar_short_0311) - abscisses_insar_short_0311

    # Vérifier qu’il y a bien des données projetées
    if len(abscisses_insar_short_0311) > 0:
        bin_centers_short_0311, median_vel_short_0311, std_vel_short_0311 = insar_data_short_0311.insar_statistics(width, nbins=120)
        bin_centers_short_0311 = np.max(bin_centers_short_0311) - bin_centers_short_0311
    else:
        print("Warning: No projected InSAR data")
except:
    print('Warning: No InSAR data')

### Shortening 2011-2019
try:
    insar_data_short_1119 = Insar(insar_short_1119, chemin_insar, profile)
    abscisses_insar_short_1119, velocities_short_1119 = insar_data_short_1119.projection_insar(width)
    abscisses_insar_short_1119 = np.max(abscisses_insar_short_1119) - abscisses_insar_short_1119

    # Vérifier qu’il y a bien des données projetées
    if len(abscisses_insar_short_1119) > 0:
        bin_centers_short_1119, median_vel_short_1119, std_vel_short_1119 = insar_data_short_1119.insar_statistics(width, nbins=120)
        bin_centers_short_1119 = np.max(bin_centers_short_1119) - bin_centers_short_1119
    else:
        print("Warning: No projected InSAR data")
except:
    print('Warning: No InSAR data')

### Shortening 2003-2019
try:
    insar_data_short_0319 = Insar(insar_short_0319, chemin_insar, profile)
    abscisses_insar_short_0319, velocities_short_0319 = insar_data_short_0319.projection_insar(width)
    abscisses_insar_short_0319 = np.max(abscisses_insar_short_0319) - abscisses_insar_short_0319

    # Vérifier qu’il y a bien des données projetées
    if len(abscisses_insar_short_0319) > 0:
        bin_centers_short_0319, median_vel_short_0319, std_vel_short_0319 = insar_data_short_0319.insar_statistics(width, nbins=120)
        bin_centers_short_0319 = np.max(bin_centers_short_0319) - bin_centers_short_0319
    else:
        print("Warning: No projected InSAR data")
except:
    print('Warning: No InSAR data')

# ======================================================================================================================
# MNT
# ======================================================================================================================
try:
    topodata = MNT(mnt, mnt_err, chemin_mnt)
    elevations = topodata.elevations(profile.points)
except:
    print('Warning: No elevation data')
# ======================================================================================================================
# strata
# ======================================================================================================================
try:
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
    abs_seismic, prof_seismic, mag, rms_values = seismic.projection_seismic(width_seismic)
    abs_seismic = np.max(abs_seismic) - abs_seismic
except:
    print('Warning: No seimsic data')

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
ax1.set_ylim([np.min(elevations)-200, np.max(elevations)+200])
ax1.tick_params(axis='y', labelcolor='k')

# InSAR
# vertical
ax1b = ax1.twinx()
try:
    #ax1b.scatter(abscisses_insar_verti_0311, velocities_verti_0311, s=2, alpha=0.1, color="blue")
    ax1b.plot(bin_centers_verti_0311, median_vel_verti_0311, color="dodgerblue", linewidth=2, alpha=1, label="InSAR déplacements verticaux 2003-2011")
    ax1b.fill_between(bin_centers_verti_0311, median_vel_verti_0311 - std_vel_verti_0311, median_vel_verti_0311 + std_vel_verti_0311,
                      color="dodgerblue", alpha=0.2)

    #ax1b.scatter(abscisses_insar_verti_1119, velocities_verti_1119, s=2, alpha=0.1, color="orange")
    ax1b.plot(bin_centers_verti_1119, median_vel_verti_1119, color="coral", linewidth=2, alpha=1, label="InSAR déplacements verticaux 2011-2019")
    ax1b.fill_between(bin_centers_verti_1119, median_vel_verti_1119 - std_vel_verti_1119, median_vel_verti_1119 + std_vel_verti_1119, color="coral",
                      alpha=0.2)

    #ax1b.scatter(abscisses_insar_verti_0319, velocities_verti_0319, s=2, alpha=0.1, color="green")
    ax1b.plot(bin_centers_verti_0319, median_vel_verti_0319, color="darkseagreen", linewidth=2, alpha=1, label="InSAR déplacements verticaux 2003-2019 ")
    ax1b.fill_between(bin_centers_verti_0319, median_vel_verti_0319 - std_vel_verti_0319, median_vel_verti_0319 + std_vel_verti_0319, color="darkseagreen",
                      alpha=0.2)
except:
    print("can't print vertical data")

# Co-seismic 2004
try:
    #ax1b.scatter(abscisses_insar_LOS_co2004, velocities_LOS_co2004, s=2, alpha=0.1, color="blue")
    ax1b.plot(bin_centers_LOS_co2004, median_vel_LOS_co2004, color="red", linewidth=2, alpha=1, label="InSAR Co-sismique 2004")
    ax1b.fill_between(bin_centers_LOS_co2004, median_vel_LOS_co2004 - std_vel_LOS_co2004, median_vel_LOS_co2004 + std_vel_LOS_co2004,
                   color="red", alpha=0.2)
except:
    print("can't print co-seismic 2004 data")

# Shortening
try:
    #ax1b.scatter(abscisses_insar_short_0311, velocities_short_0311, s=2, alpha=0.1, color="blue")
    ax1b.plot(bin_centers_short_0311, median_vel_short_0311, color="dodgerblue", linewidth=2, alpha=1, label="InSAR déplacements horizontaux 2003-2011")
    ax1b.fill_between(bin_centers_short_0311, median_vel_short_0311 - std_vel_short_0311, median_vel_short_0311 + std_vel_short_0311,
                      color="dodgerblue", alpha=0.2)

    #ax1b.scatter(abscisses_insar_short_1119, velocities_short_1119, s=2, alpha=0.1, color="orange")
    ax1b.plot(bin_centers_short_1119, median_vel_short_1119, color="coral", linewidth=2, alpha=1, label="InSAR déplacements horizontaux 2011-2019")
    ax1b.fill_between(bin_centers_short_1119, median_vel_short_1119 - std_vel_short_1119, median_vel_short_1119 + std_vel_short_1119, color="coral",
                      alpha=0.2)

    #ax1b.scatter(abscisses_insar_short_0319, velocities_short_0319, s=2, alpha=0.1, color="green")
    ax1b.plot(bin_centers_short_0319, median_vel_short_0319, color="darkseagreen", linewidth=2, alpha=1, label="InSAR déplacements horizontaux 2003-2019")
    ax1b.fill_between(bin_centers_short_0319, median_vel_short_0319 - std_vel_short_0319, median_vel_short_0319 + std_vel_short_0319, color="darkseagreen",
                      alpha=0.2)
except:
    print("can't print shortening data")

ax1b.set_ylabel("Velocity", color='r')
ax1b.set_xlim(xmin, xmax)
ax1b.set_ylim(np.min(velocities_short_0319) - 10, np.max(velocities_short_0319) + 10)
ax1b.legend(loc="upper right")

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
sizes = 50 + 150 * (mag - np.min(mag)) / (np.max(mag) - np.min(mag)) # Normalisation de la taille des points
sc2 = ax2.scatter(abs_seismic, -np.array(prof_seismic) * 1000 + 4000, c=mag, cmap="YlOrRd", edgecolor="k", s=sizes)
ax2.errorbar(abs_seismic, -np.array(prof_seismic) * 1000 + 4000, xerr=2000, yerr=2000, fmt='none', ecolor='k', capsize=3)
cax = inset_axes(ax2, width="3%", height="30%", loc="lower left", borderpad=1)
cbar = fig.colorbar(sc2, cax=cax)
cbar.set_label("Magnitude")

ax2.legend(loc="upper right")
ax2.grid(True)
ax2.legend()
ax2.axis("equal")

ax2.set_xlim([xmin, xmax])

plt.show()

# ======================================================================================================================
# Save figure
# ======================================================================================================================
output_dir = "/data/scratch/mathieu/qaidam/output/coupes_invert_plan/"

os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir,"coupe_figure1.png")

plt.savefig(output_path, dpi=300, bbox_inches='tight')

print(f'figure sauvegardé dans : {output_path}')

plt.show()
