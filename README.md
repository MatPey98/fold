# Kinematic Fault Optimization

Bayesian inversion of a fault-and-fold bend using InSAR surface velocity fields and MCMC sampling. The model fits a fault-bend fold geometry (dip angles, slip, ramp positions) to observed vertical and horizontal displacements extracted along a cross-section profile.

## Overview

The workflow:
1. Projects InSAR velocity maps onto a user-defined profile (shapefile)
2. Computes median displacement per bin along the profile
3. Runs MCMC sampling (via PyMC) to estimate posterior distributions over fault parameters
4. Outputs posterior draws and diagnostic plots

## Dependencies

```
python >= 3.9
numpy
scipy
pymc >= 5
pytensor
rasterio
geopandas
matplotlib
```

Install with:
```bash
pip install numpy scipy pymc rasterio geopandas matplotlib
```

## Configuration — `input_optimize_kinematic.py`

All model parameters are set in this single configuration file.

### Profile parameters

| Parameter | Description |
|-----------|-------------|
| `width` | Half-width of the swath profile (km) |
| `width_seismic` | Projection width for focal mechanisms onto the profile |
| `n` | Number of sampling points along the profile |
| `nbins` | Number of bins for InSAR median computation |

### MCMC parameters

| Parameter | Description |
|-----------|-------------|
| `niter` | Total number of MCMC iterations |
| `nburn` | Burn-in iterations (discarded from posterior) |
| `n_samples` | Number of posterior draws to display |
| `chains` | Number of independent MCMC chains |
| `cores` | CPU cores used for parallel sampling |

### Prior distributions

Priors are either **Uniform** `U = [lower, upper]` or **Normal** `N = [mu, sigma]`:

| Parameter | Prior | Description |
|-----------|-------|-------------|
| `Ubeta` | U[30, 50] | Fault dip angle β (degrees) |
| `Uteta` | U[7.5, 29] | Fault dip angle θ (degrees) |
| `Uomega` | U[1, 7] | Fold interlimb angle ω |
| `UY_r2` | U[11000, 17000] | Along-profile position of ramp 2 (m) |
| `UY_r3` | U[1000, 10000] | Along-profile position of ramp 3 (m) |
| `UW` | U[1000, 4000] | Fault width W (m) |
| `UW2` | U[1000, 4000] | Second fault width W2 (m) |
| `USmax` | U[10, 100] | Maximum slip Smax (mm/yr) |

### Fixed parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `Y_faille` | 18520 m | Along-profile position of the surface fault trace |
| `Z_faille` | 3430 m | Elevation of the surface fault trace |
| `Ymin` | 1000 m | Start of fault model domain |
| `Ymax` | 17800 m | End of fault model domain |
| `Zhaut` | 1800 m | Hinge elevation |
| `n_strata` | 20 | Number of stratigraphic layers |
| `di` | 4000 m | Layer thickness |

### Data paths

```
work/
├── carto/profile/       # Profile shapefile (coupe9.shp)
├── data/insar/decomp2026/
│   ├── vertical_2003-2011_mm_crop03_UTM.tif    # Vertical velocity (mm/yr)
│   └── shortening_2003-2011_mm_crop03_UTM.tif  # Horizontal shortening (mm/yr)
└── data/DEM/
    ├── cop_dem30_92_101_34_40_crop_UTM_v2.tif   # Copernicus DEM 30 m
    └── DSM_triangulation_errors_merged.tif       # DEM uncertainty
```

## Usage

```bash
# 1. Edit the configuration file to match your data and study area
nano input_optimize_kinematic.py

# 2. Run the inversion
python optimize_kinematic.py

# 3. Results are written to work/output/
```

### Minimal example

```python
# Run with 2 chains and 500 iterations for a quick test
niter  = 500
nburn  = 400
chains = 2
cores  = 2
```

Then execute:
```bash
python optimize_kinematic.py --config input_optimize_kinematic.py
```

## Output

- Posterior distributions for all free parameters
- Predicted vs. observed displacement profiles
- Fault geometry reconstruction with uncertainty bounds
- Trace plots and Gelman-Rubin convergence diagnostics (R̂)

## Reference geometry

The fault-bend fold model follows the kinematic framework of Suppe (1983) and Mitra (1990), adapted for MCMC-based uncertainty quantification with InSAR observations.

## License

MIT
