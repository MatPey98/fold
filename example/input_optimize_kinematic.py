#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ======================================================================================================================
# Profile parameters
# ======================================================================================================================
width = 2000              # Profile width (km)
width_seismic = 10000   # Width for focal mechanism projections onto the profile
n = 1000                 # Sampling of points along the profile (profile resolution)
nbins = 50               # Interpolation step for the InSAR median

# ======================================================================================================================
# MCMC parameters
# ======================================================================================================================
niter = 2000  # Total number of iterations
nburn = 1500   # Number of burn-in iterations
n_samples = 500  # Number of posterior draws to display
chains = 9     # Number of MCMC chains
cores = 9      # Number of CPU cores used

# ======================================================================================================================
# Prior parameters: U = Uniform[lower, upper]  or  N = Normal[mu, sigma]
# ======================================================================================================================
Ubeta  = [1,80]        # Uniform prior on fault dip angle beta (degrees)
Uteta  = [1, 60]        # Uniform prior on fault dip angle teta (degrees)
Uomega = [1, 40]          # Uniform prior on omega parameter
UY_r2  = [5000, 24500]   # Uniform prior on ramp position Y_r2 (m)
UY_r3  = [5000, 22000]    # Uniform prior on ramp position Y_r3 (m)
UW     = [0, 20000]    # Uniform prior on fault width W (m)
UW2    = [0, 20000]     # Uniform prior on second fault width W2 (m)
USmax  = [50, 200]        # Uniform prior on maximum slip Smax (mm)

# ======================================================================================================================
# Fixed parameters
# ======================================================================================================================
Y_faille = 24800   # Surface trace of the fault — along-profile position (m)
Z_faille = 3430    # Surface trace of the fault — elevation (m)
Ymin     = 5000    # Fault computation start position (m)
Ymax     = 24800   # Fault computation end position (m)
di       = 4000    # Number of Y-grid steps (di+1 material points along profile)
n_tot    = 1       # Total number of models
Z_ref    = 3307.0  # Reference elevation of initial flat strata in kinematic.py (G_Z0)

# ======================================================================================================================
# Data paths
# ======================================================================================================================
wdir = './work/'
output_dir = wdir + 'fold/output_2000iter/'

### Profile shapefile
coupe            = "coupe12.shp"
chemin_coupe     = wdir + "/carto/coupes/"

### insar :
insar_vertical = "vertical_2003-2019_mm_crop03_no2004_UTM.tif"
insar_horizontal = "shortening_2003-2019_mm_crop03_no2004_UTM.tif"
chemin_insar = wdir + "/data/insar/decomp2026/"


### Digital Elevation Model (DEM)
mnt              = "cop_dem30_92_101_34_40_crop03_UTM.tif"
# mnt_err          = "DSM_triangulation_errors_merged.tif"
chemin_mnt       = wdir + "/data/DEM/"
