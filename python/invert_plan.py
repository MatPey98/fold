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

# ======================================================================================================================
# InSAR
# ======================================================================================================================
### Coupe
plt.subplot(2, 1, 1)

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

plt.scatter(abscisses_insar, velocities, s=2)

### Map + coupe

# ======================================================================================================================
# MNT
# ======================================================================================================================
plt.subplot(2, 1, 2)

topodata = MNT(mnt, chemin_mnt)
elevations = topodata.elevations(profile.points)
topodata.print(abscisses, elevations)

# ======================================================================================================================
# strata
# ======================================================================================================================
directory_path = chemin_pendages
dip = Dip(directory_path)
dip.print_all(topodata, profile, length_dip, x0) # l = longueur traits de pendages

plt.xlabel("Distance (m)")
plt.ylabel("Altitude (m)")
plt.axis("equal")

plt.xlim(abscisses[0], abscisses[-1]+1000)

# ======================================================================================================================
# seismic
# ======================================================================================================================
try:
    seismic = Seismic(seismic, chemin_seismic, profile)
    abs_seismic, prof_seismic, mag = seismic.projection_seismic(width_seismic)
    seismic.print_seismic(abs_seismic, prof_seismic, mag)
except:
    print('Warning: No seimsic data')

plt.show()
# ======================================================================================================================
# Uncertainties
# ======================================================================================================================
