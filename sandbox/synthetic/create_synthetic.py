#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# ===========================================================
# create mesh
# ===========================================================

x = np.arange(0,100,1)
y = np.arange(0,100,1)
X, Y = np.meshgrid(x,y)

# ===========================================================
# topo
# ===========================================================

Z_topo = X/100 + Y/100

# ===========================================================
# cross section 
# ===========================================================

x0 = 50 # cross section position

Yv, Zv = np.meshgrid(y, np.linspace(-100, 40, 100)) # build surface
Xv = np.full_like(Yv, x0)
Z_topo_section = x0/100 - Yv/100
Zv_intersect = np.where(Zv <= Z_topo_section, Zv, np.nan)

# ===========================================================
# strata
# ===========================================================

c =20
z_dip = -Y + c
Z_dip = np.where(z_dip <= Z_topo, z_dip, np.nan)

# ===========================================================
# fault
# ===========================================================

# ===========================================================
# intersection
# ===========================================================

def intersection_line(n1, d1, n2, d2):
    direction = np.cross(n1, n2)
    A = np.vstack([n1, n2, direction])
    b = -np.array([d1, d2, 0])
    point = np.linalg.solve(A, b)
    return point, direction

# general surface
n1 = np.array([1, 0, 0])
d1 = -x0

n2 = np.array([0, 1, 1])
d2 = -c

point, direction = intersection_line(n1, d1, n2, d2)

t = np.linspace(-100, 100, 400)
X_line = point[0] + direction[0]*t
Y_line = point[1] + direction[1]*t
Z_line = point[2] + direction[2]*t

Z_topo_line = X_line/100 + Y_line/100
mask = Z_line <= Z_topo_line

X_line = X_line[mask]
Y_line = Y_line[mask]
Z_line = Z_line[mask]

# ===========================================================
# intersection cross section / topo
# ===========================================================

X_topo_vert = np.full_like(y, x0)
Y_topo_vert = y
Z_topo_vert = Z_topo[:, x0]

# ===========================================================
# intersection strata / topo
# ===========================================================
epsilon = 0.5

mask_intersection = np.abs(z_dip - Z_topo) < epsilon

X_topo_dip = X[mask_intersection]
Y_topo_dip = Y[mask_intersection]
Z_topo_dip = Z_topo[mask_intersection]

# ===========================================================
# plot 3d
# ===========================================================

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

# Surfaces
ax.plot_surface(X, Y, Z_topo, alpha=0.4)
ax.plot_surface(Xv, Yv, Zv_intersect, alpha=0.5)
ax.plot_surface(X, Y, Z_dip, alpha=0.6)

# Lignes
ax.plot(X_line, Y_line, Z_line, linewidth=4)                 # intersection
ax.plot(X_topo_vert, Y_topo_vert, Z_topo_vert, linewidth=4)  # cross section / Topo
ax.scatter(X_topo_dip, Y_topo_dip, Z_topo_dip, s=5)          # strata / Topo

ax.set_xlim(0,100)
ax.set_ylim(0,100)
ax.set_zlim(-100,40)

ax.set_xlabel("X")
ax.set_ylabel("Y (Nord)")
ax.set_zlabel("Z")

plt.show()

# ===========================================================
# create synthetic data
# ===========================================================



# ===========================================================
# create synthetic data with uncertainties 
# ===========================================================
