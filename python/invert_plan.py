# -*- coding:utf-8 -*-
__projet__ = "coupe_pendages_insar"
__nom_fichier__ = "algoV2"
__author__ = "Mathieu Peyrache"
__date__ = "février 2026"

import matplotlib.pyplot as plt
import numpy as np
from terrain_profile import Profile
from insar import Insar
from mnt import MNT
from dip import Dip
from seismic import Seismic

# ======================================================================================================================
# Paramètres
# ======================================================================================================================
width = 1000 # Largeur du profil
width_seismic = 1000
length_dip = 500 # Longueur des traits de pendages
n = 1000 # Echantillonnage des points le long du profil (résolution du profil)

# ======================================================================================================================
# Données
# ======================================================================================================================

### Profile :
coupe = "coupe9.shp"
chemin_coupe = "../sandbox/profile/"

### insar :
insar = "vertical_2014-2019_mmyr_crop03_UTM.tif"
chemin_insar = "../sandbox/insar/"

### Mnt :
mnt = "cop_dem30_92_101_34_40_crop_UTM_v2.tif"
chemin_mnt = "../sandbox/dem/"

### Pendages :
chemin_pendages = "../sandbox/strata/profile9/"

### Seismic :
#seismic = 'usgs03-14.csv'
#chemin_seismic = '../sandbox/seismic/'

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
