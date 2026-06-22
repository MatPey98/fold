#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ======================================================================================================================
# Profile parameters
# ======================================================================================================================
width = 100              # Profile width (km)
width_seismic = 10000   # Width for focal mechanism projections onto the profile
n = 1000                 # Sampling of points along the profile (profile resolution)
nbins = 50               # Interpolation step for the InSAR median

# ======================================================================================================================
# MCMC parameters
# ======================================================================================================================
niter = 1000  # Total number of iterations
nburn = 800   # Number of burn-in iterations
n_samples = 4  # Number of posterior draws to display
chains = 4     # Number of MCMC chains
cores = 8      # Number of CPU cores used

# ======================================================================================================================
# Prior parameters: U = Uniform[lower, upper]  or  N = Normal[mu, sigma]
# ======================================================================================================================
Ubeta  = [30, 50]         # Uniform prior on fault dip angle beta (degrees)
Uteta  = [7.5, 29]        # Uniform prior on fault dip angle teta (degrees)
Uomega = [1, 7]           # Uniform prior on omega parameter
# NY_r2 = [16000, 2000]   # (commented out) Normal prior on Y_r2
UY_r2  = [11000, 17000]   # Uniform prior on ramp position Y_r2 (m)
UY_r3  = [1000, 10000]    # Uniform prior on ramp position Y_r3 (m)
UW     = [1000, 4000]     # Uniform prior on fault width W (m)
UW2    = [1000, 4000]     # Uniform prior on second fault width W2 (m)
USmax  = [10, 200]        # Uniform prior on maximum slip Smax (mm)

# ======================================================================================================================
# Fixed parameters
# ======================================================================================================================
Y_faille = 18520   # Surface trace of the fault — along-profile position (m)
Z_faille = 3430    # Surface trace of the fault — elevation (m)
Ymin     = 1000    # Fault computation start position (m)
Ymax     = 17800   # Fault computation end position (m)
Ymax2    = 14000   # Second fault computation end position (m)
di       = 4000    # Layer thickness (m)
n_tot    = 1       # Total number of models

# ======================================================================================================================
# Data paths
# ======================================================================================================================
wdir = './work/'

### Profile shapefile
coupe            = "coupe9.shp"
chemin_coupe     = wdir + "/data/carto/profile/"

### InSAR decomposed velocity fields
insar_vertical   = "vertical_2003-2011_mm_crop03_UTM.tif"
insar_horizontal = "shortening_2003-2011_mm_crop03_UTM.tif"
chemin_insar     = wdir + "/data/insar/decomp2026/"

### Digital Elevation Model (DEM)
mnt              = "cop_dem30_92_101_34_40_crop_UTM_v2.tif"
# mnt_err          = "DSM_triangulation_errors_merged.tif"
chemin_mnt       = wdir + "/data/DEM/"
