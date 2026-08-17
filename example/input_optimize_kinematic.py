#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ======================================================================================================================
# Data paths
# ======================================================================================================================
wdir = './work/'

# ======================================================================================================================
# Profile parameters
# ======================================================================================================================
width = 2000              # Profile width (km)
n = 1000                 # Sampling of points along the profile (profile resolution)
nbins = 50               # Interpolation step for the InSAR median
### Profile shapefile
chemin_coupe     = wdir + "/carto/coupes/"
coupe            = "coupe12.shp"

# ======================================================================================================================
# MCMC parameters
# ======================================================================================================================
niter = 6000  # Total number of iterations
nburn = 5000   # Number of burn-in iterations
n_samples = 1000  # Number of posterior draws to display
chains = 9     # Number of MCMC chains
cores = 9      # Number of CPU cores used

# ======================================================================================================================
# Prior parameters: U = Uniform[lower, upper]  or  N = Normal[mu, sigma]
# ======================================================================================================================
Ubeta  = [0,90]        # Uniform prior on fault dip angle beta (degrees)
Uteta  = [0, 90]        # Uniform prior on fault dip angle teta (degrees)
UY_r2  = [5000, 26000]   # Uniform prior on ramp position Y_r2 (m)
UW     = [0, 10000]    # Uniform prior on fault width W (m)
US  = [0, 200]        # Uniform prior on slip S (mm)
Uomega = [0, 90]          # Uniform prior on omega parameter
UY_r3  = [5000, 24000]    # Uniform prior on ramp position Y_r3 (m)
UW2    = [0, 10000]     # Uniform prior on second fault width W2 (m)
sigma_c_vert  = 30        # Normal prior σ on vertical InSAR reference offset (mm)
sigma_c_horiz = 30        # Normal prior σ on horizontal InSAR reference offset (mm)

output_dir = wdir + 'fold/output_6000iter_3seg/'

# ======================================================================================================================
# Fixed parameters
# ======================================================================================================================
Y_fault = 24800   # Surface trace of the fault — along-profile position (m)
Z_fault = 3600    # Surface trace of the fault — elevation (m)
Ymin     = 10000    # Fault computation start position (m)
Ymax     = 24800   # Fault computation end position (m)
di           = 2000    # Number of Y-grid steps (di+1 material points along profile)

### InSAR data
d_vert  = "vertical_2003-2019_mm_crop03_no2004_UTM.tif"
d_horz  = "shortening_2003-2019_mm_crop03_no2004_UTM.tif"
# sigma_vert   = 5.0   # Observation uncertainty — vertical InSAR (mm)
# sigma_horiz  = 25.0   # Observation uncertainty — horizontal InSAR (mm)
sigma_vert   = 10.0   # Observation uncertainty — vertical InSAR (mm)
sigma_horiz  = 50.0   # Observation uncertainty — horizontal InSAR (mm)
chemin_insar = wdir + "/data/insar/decomp2026/"

### Digital Elevation Model (DEM)
mnt              = "cop_dem30_92_101_34_40_crop03_UTM.tif"
chemin_mnt       = wdir + "/data/DEM/"
