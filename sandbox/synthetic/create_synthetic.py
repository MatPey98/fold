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
dip = 70
P = np.array([50,50,0]) 

# cross section position
azimuth_section = 45
P_section = np.array([50,50,0]) # position de la coupe

# Resolution
resolution = 0.1

# lim of the model
zmin = -100
zmax = 40

xmin = 0
xmax = 100

ymin= 0
ymax = 100

# ===========================================================
# create mesh
# ===========================================================

x = np.arange(0,100,resolution)
y = np.arange(0,100,resolution)
X, Y = np.meshgrid(x,y)

# ===========================================================
# topo
# ===========================================================

Z_topo = 10*np.sin(X/10) + 10*np.cos(Y/10)

# ===========================================================
# cross section 
# ===========================================================
def section_from_azimuth(azimuth_section, P_section):

    azimuth_rad = np.radians(azimuth_section)

    dx = -np.cos(azimuth_rad)
    dy = np.sin(azimuth_rad)

    n_section = np.array([-dy, dx, 0])
    n_section = n_section / np.linalg.norm(n_section)

    d_section = -np.dot(n_section, P_section)

    t = np.linspace(-100, 100, 100)
    z = np.linspace(zmin, zmax, 100)

    T, Z_section = np.meshgrid(t,z)

    X_section = P_section[0] + dx * T
    Y_section = P_section[1] + dy * T

    Z_topo_section = 10 * np.sin(X_section/10) + 10 * np.cos(Y_section/10)

    # cross section lim
    Y_section = np.where(Y_section <= ymax, Y_section, np.nan)
    Y_section = np.where(Y_section >= ymin, Y_section, np.nan)
    X_section = np.where(X_section <= xmax, X_section, np.nan)
    X_section = np.where(X_section >= xmin, X_section, np.nan)
    Z_section = np.where(Z_section <= Z_topo_section, Z_section, np.nan)

    return X_section, Y_section, Z_section, n_section, d_section

X_section, Y_section, Z_section, n_section, d_section = section_from_azimuth(azimuth_section, P_section)

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

    Z_strata = P[2] - (nx*(X-P[0]) + ny*(Y-P[1])) / nz
    n = np.array([nx, ny, nz])
    d = -np.dot(n, P)

    ### strata limit
    Z_strata = np.where(Z_strata <= Z_topo , Z_strata, np.nan)
    Z_strata = np.where(Z_strata >= zmin, Z_strata, np.nan)

    return Z_strata, n, d

Z_strata, n_strata, d_strata = plane_from_dip_azimuth(X, Y, azimuth, dip, P)



# ===========================================================
# intersection between surfaces
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
        raise ValueError("parralel planes")

    A = np.vstack([n1, n2, direction])
    B = - np.array([d1, d2, 0])

    point = np.linalg.lstsq(A, B, rcond=None)[0]

    return point, direction

### Intersection strata / cross section
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
mask_surface = np.abs(z_diff) < 0.1

x_int = X[mask_surface]
y_int = Y[mask_surface]
z_int = Z_topo[mask_surface]

### Intersection Topo / cross section
t_profile = np.linspace(-100, 100, 1000)

azimuth_rad = np.radians(azimuth_section)
dx = -np.cos(azimuth_rad)
dy = np.sin(azimuth_rad)

x_profil = P_section[0] + dx * t_profile
y_profil = P_section[1] + dy * t_profile
z_profil = 10 * np.sin(x_profil/10) + 10 * np.cos(y_profil/10)

mask = (x_profil>=xmin) & (x_profil<=xmax) & (y_profil>=ymin) & (y_profil<=ymax)
x_profil = x_profil[mask]
y_profil = y_profil[mask]
z_profil = z_profil[mask]

# ===========================================================
# Plot 3D
# ===========================================================

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

### plot surfaces
ax.plot_surface(X, Y, Z_strata, alpha=0.5) 
ax.plot_surface(X, Y, Z_topo, alpha =0.7)
ax.plot_surface(X_section, Y_section, Z_section, alpha = 0.5)

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

### init export shapefile
output_dir = "output_create_synthetic"
output_strata_dir = os.path.join(output_dir, "strata")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_strata_dir, exist_ok=True)

crs_utm = CRS.from_epsg(32647) # epsg of the project


# cross section line
line_profile = LineString(zip(x_profil, y_profil))
gdf_profile = gpd.GeoDataFrame({"type": ["profile"]}, geometry=[line_profile], crs = crs_utm)
gdf_profile.to_file(os.path.join(output_dir, "profile.shp"))


# strata points
points_strata = [Point(x,y) for x, y in zip(x_int, y_int)]
gdf_points_strata = gpd.GeoDataFrame({"type": ["strata_points"] * len(points_strata)}, geometry=points_strata, crs = crs_utm)
gdf_points_strata.to_file(os.path.join(output_strata_dir, "strata_points.shp"))


# Topo tiff
pixel_size = resolution # résolution

# raster origin :
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