#!/usr/bin/env python3

import os
import rasterio
import numpy as np
from pyproj import CRS 
import geopandas as gpd 
import matplotlib.pyplot as plt
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point

# ===========================================================
# Parameters
# ===========================================================

# strata
azimuth = 0
dip = 20
P = np.array([0,0,0]) 

# cross section position
x0 = 50 # position de la coupe
# rajouter le changement d'orientation pour la coupe

# lim of the model
zmin = -100
zmax = 40

# ===========================================================
# create mesh
# ===========================================================

x = np.arange(0,100,1)
y = np.arange(0,100,1)
X, Y = np.meshgrid(x,y)

# ===========================================================
# topo
# ===========================================================

Z_topo = 10*np.sin(X/10) + 10*np.cos(Y/10)

# ===========================================================
# cross section 
# ===========================================================
n_section = np.array([1,0,0])
d_section = -x0

Yv, Zv = np.meshgrid(y, np.linspace(zmin, zmax, 100)) # build surface
Xv = np.full_like(Yv, x0) # return an array with pts or nan
Z_topo_section = 10 * np.sin(x0/10) + 10 * np.cos(Yv/10)
Zv_intersect = np.where(Zv <= Z_topo_section, Zv, np.nan)

# ===========================================================
# strata
# ===========================================================

def plane_from_dip_azimuth(X, Y, azimuth, dip, P):

    azimuth_rad = np.radians(azimuth)
    dip_rad = np.radians(dip)

    # unit vector
    nx = np.sin(dip_rad) * np.sin(azimuth_rad)
    ny = np.sin(dip_rad) * np.cos(azimuth_rad)
    nz = np.cos(dip_rad)

    Z = P[2] - (nx*(X-P[0]) + ny*(Y-P[1])) / nz
    n = np.array([nx, ny, nz])
    d = -np.dot(n, P)

    return Z, n, d

Z_strata, n_strata, d_strata = plane_from_dip_azimuth(X, Y, azimuth, dip, P)

### strata limit
Z_strata = np.where(Z_strata <= Z_topo , Z_strata, np.nan)
Z_strata = np.where(Z_strata >= zmin, Z_strata, np.nan)

# ===========================================================
# intersection
# ===========================================================

def intersection_plane(n1, d1, n2, d2):
    """

    n1, n2 :
    d1, d2 :
    return :
    """

    n1 = n1 / np.linalg.norm(n1)  # secure unit
    n2 = n2 / np.linalg.norm(n2)

    direction = np.cross(n1, n2) # direction de la droite d'intersection
    norm_dir = np.linalg.norm(direction)

    if norm_dir < 1e-10:
        return ValueError("parralel planes")

    A = np.vstack([n1, n2, direction])
    B = - np.array([d1, d2, 0])

    point = np.linalg.lstsq(A, B, rcond=None)[0]

    return point, direction

### INtersection strata / cross section
point_line, direction_line = intersection_plane(n_strata, d_strata, n_section, d_section)

t = np.linspace(-200, 200, 1000)
x_line = point_line[0] + direction_line[0] * t
y_line = point_line[1] + direction_line[1] * t
z_line = point_line[2] + direction_line[2] * t

z_topo_line = 10*np.sin(x_line/10) + 10*np.cos(y_line/10)
mask_line = (z_line <= z_topo_line) & (z_line >= zmin)
x_line = x_line[mask_line]
y_line = y_line[mask_line]
z_line = z_line[mask_line]

### Intersection strata / topo
z_diff = Z_strata - Z_topo
mask_surface = np.abs(z_diff) < 0.5

x_int = X[mask_surface]
y_int = Y[mask_surface]
z_int = Z_topo[mask_surface]

### Intersection Topo / cross section
y_profil = y
x_profil = np.full_like(y, x0)
z_profil = 10*np.sin(x0/10) + 10*np.cos(y/10)

# ===========================================================
# Plot 3D
# ===========================================================

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

### plot surfaces
ax.plot_surface(X, Y, Z_strata, alpha=0.5)
ax.plot_surface(Xv, Yv, Zv_intersect, alpha=0.5) 
ax.plot_surface(X, Y, Z_topo, alpha =0.7)

### plot intersection line
ax.plot(x_line, y_line, z_line, linewidth=2, color = 'b')
ax.plot(x_profil, y_profil, z_profil, linewidth=2, color = 'r')
ax.scatter(x_int, y_int, z_int, s=5, color = 'b')

ax.set_xlim(0,100)
ax.set_ylim(0,100)
ax.set_zlim(zmin,zmax)

ax.set_xlabel("X (Est-Ouest)")
ax.set_ylabel("Y (Nord-Sud)")
ax.set_zlabel("Z (Profondeur)")
plt.show()

# ===========================================================
# export shp data
# ===========================================================

### export shapefile
output_dir = "output_create_synthetic"
os.makedirs(output_dir, exist_ok=True)
crs_utm = CRS.from_epsg(32647) # epsg of the project


### cross section line
line_profile = LineString(zip(x_profil, y_profil))
gdf_profile = gpd.GeoDataFrame({"type": ["profile"]}, geometry=[line_profile], crs = crs_utm)
gdf_profile.to_file(os.path.join(output_dir, "profile.shp"))


### strata points
points_strata = [Point(x,y) for x, y in zip(x_int, y_int)]
gdf_points_strata = gpd.GeoDataFrame({"type": ["strata_points"] * len(points_strata)}, geometry=points_strata, crs = crs_utm)
gdf_points_strata.to_file(os.path.join(output_dir, "strata_points.shp"))


### Topo tiff
pixel_size = 1 # résolution

#origine du raster :
x_min = x.min()
y_max = y.max()

transform = from_origin(x_min, y_max, pixel_size, pixel_size)

output_tif = os.path.join(output_dir, "mnt.tif")

Z_topo_flipped = np.flipud(Z_topo)

with rasterio.open(
    output_tif,
    "w",
    driver="GTiff",
    height=Z_topo.shape[0],
    width=Z_topo.shape[1],
    count=1,
    dtype=Z_topo.dtype,
    crs="EPSG:32647",
    transform=transform,
) as dst:
    dst.write(Z_topo_flipped, 1)

print("data exported to :", output_dir)