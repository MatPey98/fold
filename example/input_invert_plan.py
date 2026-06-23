#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ======================================================================================================================
# Base directory
# ======================================================================================================================
wdir = '/Users/simon/scratch/dev_fold/work/'

# ======================================================================================================================
# Profile parameters
# ======================================================================================================================
width          = 5000   # Swath half-width (m)
width_seismic  = 15000  # Swath half-width for seismic projection (m)
length_dip     = 1000   # Length of strata dip segments on the plot (m)
length_dip_fault = 1500 # Length of fault dip segments on the plot (m)
n              = 1000   # Number of points along the profile

chemin_coupe = wdir + "carto/coupes/"
chemin_fault    = wdir + "carto/fault/profile9/"

# ======================================================================================================================
# Profile shapefile
# ======================================================================================================================
# coupe        = "coupe8.shp" # section 1
# chemin_pendages = wdir + "carto/strata/coupe8/"

# coupe        = "coupe6.shp" # section 2
# chemin_pendages = wdir + "carto/strata/coupe6/"
#
# coupe        = "coupe2.shp" # section 3
# chemin_pendages = wdir + "carto/strata/coupe2/"
#
# coupe        = "coupe10bashaut.shp" # section 4
# chemin_pendages = wdir + "carto/strata/coupe10/"

coupe        = "coupe5.shp" # section 5
chemin_pendages = wdir + "carto/strata/coupe5/"

# coupe        = "coupe1.shp" # section 6
# chemin_pendages = wdir + "carto/strata/coupe1/"

# coupe        = "coupe13.shp" # section 7
# chemin_pendages = wdir + "carto/strata/coupe13/"


# ======================================================================================================================
# InSAR datasets
#
# Each list contains (filename, label) tuples; all files are loaded from chemin_insar.
# Add or comment out entries to control which datasets are plotted.
# Vertical (solid line) and shortening (dashed line) are shown on the same axis.
# Colors are assigned automatically from a palette.
# ======================================================================================================================
chemin_insar = wdir + "data/insar/decomp2026/"

insar_verticals = [
    # ("vertical_2003-2011_mm_crop03_no2004_UTM.tif",   "2003–2011"),
    # ("vertical_2011-2019_mm_crop03_no2004_UTM.tif", "2011–2019"),
    ("vertical_2003-2019_mm_crop03_no2004_UTM.tif", "2003–2019"),
]

insar_shortenings = [
    # ("shortening_2003-2011_mm_crop03_no2004_UTM.tif",   "2003–2011"),
    # ("shortening_2011-2019_mm_crop03_no2004_UTM.tif", "2014–2019"),
    ("shortening_2003-2019_mm_crop03_no2004_UTM.tif", "2003–2019"),
]

# ======================================================================================================================
# Digital Elevation Model (DEM)
# ======================================================================================================================
mnt_figure         = "cop_dem30_92_101_34_40_crop03_UTM.tif"
mnt         = "dsm_denoised_filtered.tif"
mnt_err     = "DSM_triangulation_errors_merged.tif"   # optional — remove to use sigma=1
chemin_mnt  = wdir + "data/DEM/"

# ======================================================================================================================
# Seismic catalogue
# ======================================================================================================================
seismic        = 'usgs-cmt-zha2013-sun2012.csv'
chemin_seismic = wdir + 'data/seismic/'

# ======================================================================================================================
# Output
# ======================================================================================================================
output_dir = '/Users/simon/scratch/dev_fold/work/fold/output/'
