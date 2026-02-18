#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd 
from pyproj import CRS 

# ===========================================================
# Parameters
# ===========================================================

# strata
azimuth = 0
dip = 69
P = ([0,0,40]) 

# cross section position
x0 = 50

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

Yv, Zv = np.meshgrid(y, np.linspace(-100, 40, 100)) # build surface
Xv = np.full_like(Yv, x0)
Z_topo_section = np.sin(x0/100) - np.cos(Yv/100)
Zv_intersect = np.where(Zv <= Z_topo_section, Zv, np.nan)

# ===========================================================
# strata
# ===========================================================

def plan(X, Y, azimuth, dip, P):

    azimuth_rad = np.radians(azimuth)
    dip_rad = np.radians(dip)

    # normal vector
    nx = np.sin(dip_rad) * np.sin(azimuth_rad)
    ny = np.sin(dip_rad) * np.cos(azimuth_rad)
    nz = np.cos(dip_rad)

    Z = P[2] - (nx*(X-P[0]) + ny*(Y-P[1])) / nz
    n = np.array([nx, ny, nz])
    d = -np.dot(n, P)
    # a = - np.tan(dip) * np.sin(az)
    # b = - np.tan(dip) * np.cos(az)

    # Z = a * X + b * Y + c

    return Z, n, d

Z_strata, n, d = plan(X, Y, azimuth, dip, P)

# ===========================================================
# intersection
# ===========================================================

def intersection_line(n1, d1, n2, d2):

    direction = np.cross(n1, n2)

    A = np.vstack([n1, n2, direction])
    B = - np.array([d1,d2, 0])

    point = np.linalg.solve(A, B)

    return point, direction

# ===========================================================
# linspace
# ===========================================================

# ===========================================================
# Plot 3D
# ===========================================================

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z_strata)
ax.plot_surface(Xv, Yv, Zv_intersect) 
ax.plot_surface(X, Y, Z_topo)

ax.set_xlim(0,100)
ax.set_ylim(0,100)
ax.set_zlim(-100,40)

ax.set_xlabel("X")
ax.set_ylabel("Y (Nord)")
ax.set_zlabel("Z")
plt.show()

# ===========================================================
# Create data
# ===========================================================
