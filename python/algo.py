import os
import csv

import matplotlib.pyplot as plt
import numpy as np

from profile import Profile
from insar import Insar
from mnt import MNT
from dip import Dip


l = 100
n = 1000

####################################################
### import your pofile
####################################################
profile = Profile("coupe7.shp","/home/geodynamique/Documents/mathieu/scripts/fold/sandbox/profile/")
profile.get_extreme_points()
profile.get_azimuth()
profile.linspace(n)

abscisses = [np.sqrt(profile.points[i][0]**2 + profile.points[i][1]**2) for i in range(n + 1)]
x0 = abscisses[0]
abscisses = [abscisses[i] - x0 for i in range(n + 1)]
plt.subplot(2, 1, 1)

####################################################
### import InSAR data
####################################################
list_insar = []
list_insar.append(Insar("vertical_2014-2019_mmyr_crop03_UTM.tif","/home/geodynamique/Documents/mathieu/scripts/fold/sandbox/insar/"))
for insar in list_insar:
    velocities = insar.velocities(profile.points)
    a, v = insar.smooth_profile(abscisses, velocities)
    plt.xlim(abscisses[0], abscisses[-1])
    plt.ylabel("Elevation (mm/an)")
    insar.print(a, v)

with open('coordonnees.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(len(a)):
        writer.writerow([a[i], v[i]])
        
####################################################
### import DEM
####################################################
plt.subplot(2, 1, 2)
topodata = MNT("cop_dem30_92_101_34_40_crop_UTM_v2.tif", "/home/geodynamique/Documents/mathieu/scripts/fold/sandbox/dem/")
elevations = topodata.elevations(profile.points)
topodata.print(abscisses, elevations)

####################################################
### import strata
####################################################
directory_path = "/home/geodynamique/Documents/mathieu/scripts/fold/sandbox/strata/profile7/"
dip = Dip(directory_path)
dip.print_all(topodata, profile, l, x0)

plt.xlabel("Distance (m)")
plt.ylabel("Altitude (m)")
plt.axis("equal")
plt.xlim(abscisses[0], abscisses[-1])
plt.show()
