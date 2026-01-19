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

profile = Profile("coupe3.shp", "C:/Users/gaspa/Desktop/stage_CRPG/qgis/carto/coupes/")
profile.get_extreme_points()
profile.get_azimuth()
profile.linspace(n)

abscisses = [np.sqrt(profile.points[i][0]**2 + profile.points[i][1]**2) for i in range(n + 1)]
x0 = abscisses[0]
abscisses = [abscisses[i] - x0 for i in range(n + 1)]

plt.subplot(2, 1, 1)
list_insar = []
list_insar.append(Insar("up_reprojete.tif", "C:/Users/gaspa/Desktop/stage_CRPG/qgis/carto/insar/"))
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

plt.subplot(2, 1, 2)
topodata = MNT("dsm_denoised_filtered.tif", "C:/Users/gaspa/Desktop/stage_CRPG/qgis/carto/MNT/")
elevations = topodata.elevations(profile.points)
topodata.print(abscisses, elevations)

directory_path = "C:/Users/gaspa/Desktop/stage_CRPG/qgis/carto/points/EST/"
dip = Dip(directory_path)
dip.print_all(topodata, profile, l, x0)

plt.xlabel("Distance (m)")
plt.ylabel("Altitude (m)")
plt.axis("equal")
plt.xlim(abscisses[0], abscisses[-1])
plt.show()