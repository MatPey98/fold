# Kinematic Fault Optimization

Bayesian inversion of a fault-bend fold using InSAR surface velocity fields and MCMC sampling. The model fits a tri-ramp fault geometry (dip angles, slip, ramp positions, hinge widths) to observed vertical and horizontal displacements extracted along a cross-section profile.

**Recommended workflow:**
1. `invert_plan.py` — visualize all available data on the profile and measure fault/strata dip from the DEM to constrain the priors
2. `optimize_kinematic.py` — run the Bayesian inversion using priors informed by step 1
3. `plot_histo.py` — plot posterior histograms from the trace files saved by step 2

---

## File structure

```
fold/
├── python/
│   ├── invert_plan.py          # Step 1: data visualization + dip inversion from DEM
│   ├── optimize_kinematic.py   # Step 2: Bayesian MCMC inversion
│   ├── plot_histo.py           # Step 3: posterior histogram plots from trace files
│   ├── kinematic.py            # Forward model: tri-ramp fault geometry + kink-band kinematics
│   ├── dip.py                  # Dip class: plane fitting + projection onto profile
│   ├── read_data.py            # I/O classes: Insar, MNT, Seismic
│   └── profile.py              # Profile class: swath projection of 2D spatial data
└── README.md
work/
└── fold/
    ├── input_invert_plan.py          # Configuration for step 1
    └── input_optimize_kinematic.py   # Configuration for step 2
```

---

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
scipy
```

Install with:
```bash
pip install numpy pymc arviz rasterio geopandas matplotlib pyproj pandas scipy
```

---

## Step 1 — Data visualization and dip inversion (`invert_plan.py`)

Maps all available observations onto a cross-section profile: InSAR velocity fields (vertical and horizontal, multiple time periods), topography, strata and fault dip measurements from DEM V-shapes in valleys, and seismic catalogue.

Dip angles are estimated by fitting a plane to 3D points digitized from a GIS shapefile using weighted least squares (weights = inverse DEM triangulation error). Uncertainty is propagated via Monte Carlo.

**Usage:**
```bash
python ./fold/python/invert_plan.py ./work/fold/input_invert_plan.py
```

**Output:** `cross_section.png` saved to `output_dir` (configurable in the input file).

### Configuration — `input_invert_plan.py`

#### Profile

| Parameter | Description |
|-----------|-------------|
| `coupe` | Profile shapefile filename |
| `chemin_coupe` | Path to the profile shapefile |
| `width` | Swath half-width (km) |
| `n` | Number of points along the profile |

#### InSAR datasets (all optional)

Datasets are defined as lists of `(filename, label)` tuples; all files are loaded from `chemin_insar`. Add, remove, or comment out entries to control which datasets appear on the plot. Colors are assigned automatically. Vertical profiles are drawn as solid lines, shortening profiles as dashed lines.

```python
insar_verticals = [
    ("vertical_2003-2011_mm_crop03_UTM.tif", "2003–2011"),
    ("vertical_2014-2019_mm_crop03_UTM.tif", "2014–2019"),  # comment out to hide
]

insar_shortenings = [
    ("shortening_2003-2011_mm_crop03_UTM.tif", "2003–2011"),
]
```

| Variable | Description |
|----------|-------------|
| `insar_verticals` | List of `(filename, label)` — vertical velocity rasters (mm/yr) |
| `insar_shortenings` | List of `(filename, label)` — horizontal shortening rasters (mm/yr) |
| `chemin_insar` | Path to the InSAR raster directory |

#### DEM and dip measurements

| Parameter | Description |
|-----------|-------------|
| `mnt` | High-resolution DEM raster — used for dip plane fitting (strata and fault) |
| `mnt_figure` | Low-resolution DEM raster — **optional**, used only for the topography profile display; falls back to `mnt` if omitted |
| `mnt_err` | DEM uncertainty raster — **optional**, defaults to σ = 1 if omitted |
| `chemin_mnt` | Path to DEM rasters (applies to both `mnt` and `mnt_figure`) |
| `chemin_pendages` | Directory containing strata dip shapefiles (one .shp per measurement) |
| `chemin_fault` | Directory containing fault dip shapefiles |
| `length_dip` | Length of strata dip segments on the plot (m) |
| `length_dip_fault` | Length of fault dip segments on the plot (m) |

#### Seismic catalogue (optional)

| Parameter | Description |
|-----------|-------------|
| `seismic` | CSV filename with columns: longitude, latitude, depth, mag, rms |
| `chemin_seismic` | Path to the seismic CSV |
| `width_seismic` | Swath half-width for seismic projection (m) |

#### Output

| Parameter | Description |
|-----------|-------------|
| `output_dir` | Directory for saved figures (default: current directory) |

---

## Step 2 — Bayesian MCMC inversion (`optimize_kinematic.py`)

Inverts the kinematic fault model against InSAR vertical and horizontal velocity profiles using PyMC + Metropolis sampling.

**Usage:**
```bash
python ./fold/python/optimize_kinematic.py ./work/fold/input_optimize_kinematic.py
```

### Configuration — `input_optimize_kinematic.py`

#### Profile

| Parameter | Description |
|-----------|-------------|
| `width` | Swath half-width (km) |
| `n` | Number of points along the profile |

#### MCMC

| Parameter | Description |
|-----------|-------------|
| `niter` | Number of MCMC draws per chain |
| `nburn` | Burn-in iterations (discarded) |
| `n_samples` | Number of posterior realizations shown in the fault geometry plot |
| `chains` | Number of independent chains (≥ 4 recommended) |
| `cores` | CPU cores for parallel sampling (set equal to `chains`) |

**Convergence**: R̂ < 1.01 required; increase `niter` or `chains` if not reached.

#### Prior distributions (Uniform)

| Parameter | Description |
|-----------|-------------|
| `Ubeta` | Ramp 1 dip angle β (degrees) |
| `Uteta` | Ramp 2 dip angle θ (degrees) |
| `Uomega` | Ramp 3 dip angle ω (degrees) |
| `UY_r2` | Along-profile position of ramp 1→2 transition (m) |
| `UY_r3` | Along-profile position of ramp 2→3 transition (m) |
| `UW` | Width of the upper hinge (m) |
| `UW2` | Width of the lower hinge (m) |
| `USmax` | Maximum shortening Smax (mm/yr) |

Constraint enforced: β > θ > ω (geometrically required).

#### Fixed parameters

| Parameter | Description |
|-----------|-------------|
| `Y_faille` | Along-profile position of the surface fault trace (m) |
| `Z_faille` | Elevation of the surface fault trace (m) |
| `Ymin` | Model domain start — also used to filter InSAR data (m) |
| `Ymax` | Model domain end — also used to filter InSAR data (m) |
| `di` | Number of material points discretizing the deforming layer |
| `n_tot` | Number of incremental shortening steps (1 = single step) |

#### Data paths

| Variable | Description |
|----------|-------------|
| `coupe` | Profile shapefile filename |
| `chemin_coupe` | Path to the profile shapefile |
| `insar_vertical` | Vertical velocity raster (GeoTIFF, mm/yr) |
| `insar_horizontal` | Horizontal shortening raster (GeoTIFF, mm/yr) |
| `chemin_insar` | Path to InSAR rasters |
| `mnt` | DEM raster filename |
| `mnt_err` | DEM uncertainty raster — **optional**, defaults to σ = 1 if omitted |
| `chemin_mnt` | Path to DEM rasters |

---

## Kinematic model

`kinematic.py` implements a tri-ramp fault-bend fold:

- **Three ramps** at dip angles β, θ, ω connected by two curved hinges of widths W and W2
- **Four axial surfaces** partition the hanging wall into deformation zones
- Material points are advected following kink-band kinematics
- Output: vertical uplift and horizontal shortening at the surface, interpolated to InSAR positions

The likelihood assumes independent Gaussian noise: σ = 10 mm (vertical) and σ = 50 mm (horizontal).

### Output files

All figures are saved as PDF to `output_dir` (configurable in the input file).

| File | Description |
|------|-------------|
| `trace.pdf` | MCMC chain evolution per parameter |
| `posterior.pdf` | Posterior histograms |
| `corner.pdf` | Pairwise parameter correlations (hexbin) |
| `forest_angles.pdf` | 95% HDI intervals for β, θ, ω, Smax |
| `forest_geometry.pdf` | 95% HDI intervals for Y_r2, Y_r3, W, W2 |
| `model_fit.pdf` | Predicted vs. observed displacement profiles + posterior fault geometry |
| `traces/<param>.txt` | Raw posterior samples per parameter (all chains concatenated, one value per line) |

---

## Step 3 — Posterior histogram plots (`plot_histo.py`)

Reads the trace text files saved by `optimize_kinematic.py` and produces publication-ready histogram figures with mean and 95% HDI for each parameter.

**Usage:**
```bash
python3 fold/python/plot_histo.py <traces_dir> [output_dir]
```

Example:
```bash
python3 fold/python/plot_histo.py work/fold/output/traces/ work/fold/output/
```

**Output:**

| File | Description |
|------|-------------|
| `posterior_histo.pdf` | 4×2 grid — one subplot per parameter with histogram, mean, 95% HDI |
| `posterior_dip_angles.pdf` | Dip angles β, θ, ω overlaid for direct comparison |

The seismic catalogue supports mixed date formats (USGS, Sun2012, Zha2013). Missing RMS values default to 1 km.

---

## Reference geometry

Kinematic framework follows Suppe (1983) and Mitra (1990), adapted for MCMC-based uncertainty quantification with InSAR observations.

## License

MIT
