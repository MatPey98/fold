#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# ===========================================================
# Strata parameters
# ===========================================================

azimuth = 0
dip = 69
c = 20

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

x0 = 50 # cross section position
Yv, Zv = np.meshgrid(y, np.linspace(-100, 40, 100)) # build surface
Xv = np.full_like(Yv, x0)
Z_topo_section = np.sin(x0/100) - np.cos(Yv/100)
Zv_intersect = np.where(Zv <= Z_topo_section, Zv, np.nan)

# ===========================================================
# strata
# ===========================================================

def plan(Z, Y, azimuth, dip, c):

    az = np.radians(azimuth)
    d = np.radians(dip)

    a = - np.tan(dip) * np.sin(az)
    b = - np.tan(dip) * np.cos(az)

    Z = a * X + b * Y + c

    return Z

Z_strata = plan(X, Y, azimuth, dip, c)

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
