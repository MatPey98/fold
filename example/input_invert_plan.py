#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ======================================================================================================================
# Base directory
# ======================================================================================================================
wdir = '/Users/simon/scratch/dev_fold/work/data/'

# ======================================================================================================================
# Profile parameters
# ======================================================================================================================
width          = 1000   # Swath half-width (m)
width_seismic  = 15000  # Swath half-width for seismic projection (m)
length_dip     = 1000   # Length of strata dip segments on the plot (m)
length_dip_fault = 1500 # Length of fault dip segments on the plot (m)
n              = 1000   # Number of points along the profile

# ======================================================================================================================
# Profile shapefile
# ======================================================================================================================
coupe        = "coupe9.shp"
chemin_coupe = wdir + "carto/profile/"

# ======================================================================================================================
# Strata and fault dip measurements
# ======================================================================================================================
chemin_pendages = wdir + "carto/strata/profile9/"
chemin_fault    = wdir + "carto/fault/profile9/"

# ======================================================================================================================
# InSAR datasets
#
# Each list contains (filename, label) tuples; all files are loaded from chemin_insar.
# Add or comment out entries to control which datasets are plotted.
# Vertical (solid line) and shortening (dashed line) are shown on the same axis.
# Colors are assigned automatically from a palette.
# ======================================================================================================================
chemin_insar = wdir + "insar/decomp2026/"

insar_verticals = [
    # ("vertical_2003-2011_mm_crop03_UTM.tif",   "2003–2011"),
    # ("vertical_2014-2019_mmyr_crop03_UTM.tif", "2014–2019"),
    ("vertical_2003-2019_mm_crop03_UTM.tif", "2014–2019"),
]

insar_shortenings = [
    # ("shortening_2003-2011_mm_crop03_UTM.tif",   "2003–2011"),
    # ("shortening_2014-2019_mmyr_crop03_UTM.tif", "2014–2019"),
    ("shortening_2003-2019_mm_crop03_UTM.tif", "2003–2019"),
]

# ======================================================================================================================
# Digital Elevation Model (DEM)
# ======================================================================================================================
mnt         = "cop_dem30_92_101_34_40_crop_UTM_v2.tif"
mnt_err     = "DSM_triangulation_errors_merged.tif"   # optional — remove to use sigma=1
chemin_mnt  = wdir + "DEM/"

# ======================================================================================================================
# Seismic catalogue
# ======================================================================================================================
seismic        = 'usgs-cmt-zha2013-sun2012.csv'
chemin_seismic = wdir + 'seismic/'

# ======================================================================================================================
# Output
# ======================================================================================================================
output_dir = '/Users/simon/scratch/dev_fold/work/fold/output/'
