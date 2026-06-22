# Kinematic Fault Optimization

Bayesian inversion of a fault-bend fold using InSAR surface velocity fields and MCMC sampling. The model fits a tri-ramp fault geometry (dip angles, slip, ramp positions, hinge widths) to observed vertical and horizontal displacements extracted along a cross-section profile.

## Overview

The workflow:
1. Projects InSAR velocity maps onto a user-defined profile (shapefile)
2. Filters data within the profile swath and the model domain `[Ymin, Ymax]`
3. Runs MCMC sampling (via PyMC + Metropolis) to estimate posterior distributions over fault parameters
4. Outputs posterior draws, convergence diagnostics, and fault geometry plots

## File structure

```
fold/
├── python/
│   ├── optimize_kinematic.py   # Main script: data loading, MCMC, plots
│   ├── kinematic.py            # Forward model: fault geometry + strata deformation
│   ├── read_data.py            # I/O classes: Insar, MNT, Seismic
│   └── profile.py              # Profile class: projection of 2D data onto cross-section
└── README.md
work/
└── fold/
    └── input_optimize_kinematic.py   # Configuration file (edit this)
```

## Dependencies

```
python >= 3.9
numpy
pymc >= 5
pytensor
arviz
rasterio
geopandas
matplotlib
pyproj
pandas
```

Install with:
```bash
pip install numpy pymc arviz rasterio geopandas matplotlib pyproj pandas
```

## Usage

```bash
python ./fold/python/optimize_kinematic.py ./work/fold/input_optimize_kinematic.py
```

### Quick test (2 chains, few iterations)

```python
# in input_optimize_kinematic.py
niter  = 100
nburn  = 50
chains = 2
cores  = 2
```

## Configuration — `input_optimize_kinematic.py`

All model parameters are set in this single file.

### Profile

| Parameter | Description |
|-----------|-------------|
| `width` | Half-width of the swath profile (km) |
| `n` | Number of sampling points along the profile |

### MCMC

| Parameter | Description |
|-----------|-------------|
| `niter` | Number of MCMC draws per chain |
| `nburn` | Burn-in iterations (discarded) |
| `n_samples` | Number of posterior realizations to display in the fault plot |
| `chains` | Number of independent MCMC chains (≥ 4 recommended) |
| `cores` | CPU cores for parallel sampling (set equal to `chains`) |

**Convergence**: R̂ < 1.01 is required; increase `niter` or `chains` if not reached.

### Prior distributions

Priors are Uniform `U = [lower, upper]`:

| Parameter | Description |
|-----------|-------------|
| `Ubeta` | Fault dip angle β on ramp 1 (degrees) |
| `Uteta` | Fault dip angle θ on ramp 2 (degrees) |
| `Uomega` | Fault dip angle ω on ramp 3 (degrees) |
| `UY_r2` | Along-profile position of the ramp 1→2 transition (m) |
| `UY_r3` | Along-profile position of the ramp 2→3 transition (m) |
| `UW` | Width of the first (upper) hinge (m) |
| `UW2` | Width of the second (lower) hinge (m) |
| `USmax` | Maximum shortening Smax (mm/yr) |

Constraint enforced during sampling: β > θ > ω (otherwise the geometry is invalid).

### Fixed parameters

| Parameter | Description |
|-----------|-------------|
| `Y_faille` | Along-profile position of the surface fault trace (m) |
| `Z_faille` | Elevation of the surface fault trace (m) |
| `Ymin` | Start of the model domain — also used to filter InSAR data (m) |
| `Ymax` | End of the model domain — also used to filter InSAR data (m) |
| `di` | Number of material points used to discretize the deforming layer |
| `n_tot` | Number of incremental shortening steps (1 = apply Smax in one step) |

### Data paths

| Variable | Description |
|----------|-------------|
| `coupe` | Profile shapefile filename |
| `chemin_coupe` | Path to the profile shapefile |
| `insar_vertical` | Vertical velocity raster (mm/yr, GeoTIFF) |
| `insar_horizontal` | Horizontal shortening raster (mm/yr, GeoTIFF) |
| `chemin_insar` | Path to InSAR rasters |
| `mnt` | DEM raster (GeoTIFF) |
| `mnt_err` | DEM uncertainty raster (GeoTIFF) — **optional**, defaults to σ = 1 if omitted |
| `chemin_mnt` | Path to DEM rasters |

Expected directory layout:
```
work/
├── data/carto/profile/          # Profile shapefile
├── data/insar/decomp2026/       # Vertical and horizontal InSAR rasters
└── data/DEM/                    # DEM and optional uncertainty raster
```

## Kinematic model

The forward model (`kinematic.py`) implements a tri-ramp fault-bend fold:

- **Three ramps** at dip angles β, θ, ω connected by two curved hinges of width W and W2
- **Four axial surfaces** partition the hanging wall into deformation zones
- Material points are advected through the hinge zones following kink-band kinematics
- The model outputs vertical uplift and horizontal shortening at the surface, interpolated to InSAR observation positions

The likelihood assumes independent Gaussian noise on vertical (`σ = 10 mm`) and horizontal (`σ = 50 mm`) components.

## Output

Running the script produces:

- **Trace plots** — MCMC chain evolution per parameter
- **Posterior histograms** — marginal distributions
- **Corner plot** — pairwise parameter correlations
- **Forest plots** — HDI 95% intervals
- **Model fit plot** — predicted vs. observed vertical and horizontal displacements
- **Fault geometry plot** — posterior fault and axial surface realizations

## Reference geometry

The kinematic framework follows Suppe (1983) and Mitra (1990), adapted for MCMC-based uncertainty quantification with InSAR observations.

## License

MIT
