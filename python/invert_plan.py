#!/usr/bin/env python3
# -*- coding:utf-8 -*-

__projet__ = "coupe_pendages_insar"
__nom_fichier__ = "invert_plan.py"
__author__ = "Mathieu Peyrache"
__date__ = "février 2026"

import matplotlib.pyplot as plt
import numpy as np
import sys
import getopt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from terrain_profile import Profile
from insar import Insar
from mnt import MNT
from dip import Dip
from seismic import Seismic


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
abscisses = profile.abscisse #
x0 = abscisses[0]
xmin = abscisses[0]
xmax = abscisses[-1]

# ======================================================================================================================
# InSAR
# ======================================================================================================================
try :

  insar_data = Insar(insar, chemin_insar)

  band = insar_data.raster.read(1)
  rows, cols = band.shape

  abscisses_insar = []
  velocities = []

  for row in range(rows):
      for col in range(cols):

          x, y = insar_data.raster.xy(row, col)
          proj = profile.get_projection((x, y), width)

          if proj is not None:

              xpp, ypp = proj
              value = band[row, col]

              if not np.isnan(value):
                  abscisses_insar.append(xpp)
                  velocities.append(value)

except:
    print('Warning: No insar data')

# ======================================================================================================================
# MNT
# ======================================================================================================================
topodata = MNT(mnt, mnt_err, chemin_mnt)
elevations = topodata.elevations(profile.points)

# ======================================================================================================================
# strata
# ======================================================================================================================
directory_path = chemin_pendages
dip = Dip(directory_path)

# ======================================================================================================================
# seismic
# ======================================================================================================================
try:
    seismic = Seismic(seismic, chemin_seismic, profile)
    abs_seismic, prof_seismic, mag = seismic.projection_seismic(width_seismic)
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
# InSAR
sc1 = ax1.scatter(abscisses_insar, velocities, s=2, label="InSAR vertical velocities")
ax1.set_ylabel("Velocity")
ax1.set_xlim(xmin, xmax)
ax1.legend(loc="upper right")

### Lower plot
# Topo
ax2.plot(abscisses, elevations, color="black")
ax2.set_xlim(xmin, xmax)
ax2.set_xlabel("Distance (m)")
ax2.set_ylabel("Altitude (m)")

# Strata
dip.print_all(topodata, profile, length_dip, x0)

# Seismic
sc2 = ax2.scatter(abs_seismic, -np.array(prof_seismic) * 1000, c=mag, cmap="YlOrRd", edgecolor="k", s=50)
cax = inset_axes(ax2, width="3%", height="30%", loc="lower left", borderpad=1)
cbar = fig.colorbar(sc2, cax=cax)
cbar.set_label("Magnitude")

plt.show()