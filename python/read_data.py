import re
import rasterio
import rasterio.windows
import numpy as np
import pandas as pd
from datetime import datetime
from pyproj import Transformer


def _parse_seismic_time(time_str):
    """Parse date/time strings from mixed seismic catalogue formats.

    Handles three formats found in the catalogue:
      - USGS  : "2004-5-4-5:5:0.3"   → YYYY-M-D-H:M:S.f
      - Sun2012: "030417-004839"       → YYMMDD-HHMMSS  (2000 + YY)
      - Zha2013: "17/04/2003"          → DD/MM/YYYY

    Returns decimal year (float), e.g. 2004.34.
    """
    s = str(time_str).strip()

    # USGS: YYYY-M-D-H:M:S[.f]  (seconds may have a leading space, e.g. ": 5.2")
    m = re.match(r'^(\d{4})-(\d{1,2})-(\d{1,2})-(\d{1,2}):(\d{1,2}):\s*([\d.]+)', s)
    if m:
        y, mo, d, h, mi, sec = m.groups()
        dt = datetime(int(y), int(mo), int(d), int(h), int(mi), int(float(sec)))
        return dt.year + (dt.timetuple().tm_yday - 1) / 365.25

    # Sun2012: YYMMDD-HHMMSS
    m = re.match(r'^(\d{2})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})$', s)
    if m:
        yy, mo, d, h, mi, sec = m.groups()
        dt = datetime(2000 + int(yy), int(mo), int(d), int(h), int(mi), int(sec))
        return dt.year + (dt.timetuple().tm_yday - 1) / 365.25

    # Zha2013: DD/MM/YYYY
    m = re.match(r'^(\d{2})/(\d{2})/(\d{4})$', s)
    if m:
        d, mo, y = m.groups()
        dt = datetime(int(y), int(mo), int(d))
        return dt.year + (dt.timetuple().tm_yday - 1) / 365.25

    # Fallback: let pandas try (ISO8601 / standard formats)
    try:
        ts = pd.Timestamp(s)
        return ts.year + (ts.day_of_year - 1) / 365.25
    except Exception:
        return float('nan')

'==================================================================='
'                               INSAR                               '
'==================================================================='
class Insar:

    def __init__(self, filename, wdir, profile):
        self.path = wdir + filename
        self.raster = rasterio.open(self.path)
        self.profile = profile

    def projection_insar(self, width):

        band = self.raster.read(1)
        rows, cols = band.shape

        abscisses_insar = []
        velocities = []

        for row in range(rows):
            for col in range(cols):
                value = band[row, col]
                # Skip invalid pixels before computing coordinates (faster)
                if np.isnan(value) or not np.isfinite(value) or value >= 1e10:
                    continue
                x, y = self.raster.xy(row, col)
                proj = self.profile.get_projection((x, y), width)
                if proj is not None:
                    xpp, ypp = proj
                    abscisses_insar.append(xpp)
                    velocities.append(value)

        return np.array(abscisses_insar), np.array(velocities)

    def insar_statistics(self, width, nbins=100):
        """
        Computes median and standard deviation of InSAR velocities along the profile.
        Ignores NaN, inf, -inf, and zero values.
        """
        abscisses_insar, velocities = self.projection_insar(width)
        xmin = 0
        xmax = self.profile.l

        bins = np.linspace(xmin, xmax, nbins)
        digitized = np.digitize(abscisses_insar, bins)

        bin_centers = []
        median_vel = []
        std_vel = []

        for i in range(1, len(bins)):
            mask = digitized == i
            if np.any(mask):
                bin_centers.append((bins[i] + bins[i - 1]) / 2)
                median_vel.append(np.median(velocities[mask]))
                std_vel.append(np.std(velocities[mask]))

        return np.array(bin_centers), np.array(median_vel), np.array(std_vel)

'==================================================================='
'                               MNT                                 '
'==================================================================='
class MNT:

    def __init__(self, filename, filename_err, wdir):
        self.path = wdir + filename
        self.raster = rasterio.open(self.path)
        # Error raster is optional — if None, elevations_err returns sigma=1
        self.raster_err = None
        if filename_err is not None:
            self.path_err = wdir + filename_err
            self.raster_err = rasterio.open(self.path_err)

    def elevations(self, coord):
        elevations = []
        for pt in coord:
            lig, col = self.raster.index(pt[0], pt[1])
            window = rasterio.windows.Window(col, lig, 1, 1)
            try:
                elevation = self.raster.read(1, window=window)[0, 0]
            except IndexError:
                elevations.append(np.nan)
                continue
            elevations.append(elevation)
        return elevations

    def elevations_err(self, coord):
        if self.raster_err is None:
            return np.ones(len(coord))  # default sigma = 1
        elevations_err = []
        for pt in coord:
            lig, col = self.raster_err.index(pt[0], pt[1])
            window = rasterio.windows.Window(col, lig, 1, 1)
            try:
                elevation = self.raster_err.read(1, window=window)[0, 0]
            except IndexError:
                elevations_err.append(np.nan)
                continue
            elevations_err.append(elevation)
        return elevations_err

'==================================================================='
'                               Strata                              '
'==================================================================='

'==================================================================='
'                               Seismic                             '
'==================================================================='
class Seismic:

    def __init__(self, filename, wdir, profile):
        """
        :param filename:
        :param wdir:
        """
        self.path = wdir + filename
        self.profile = profile
        self.data = pd.read_csv(self.path, sep=r",")  # comma separator

        # Coordinate conversion: geographic (WGS84) → UTM
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32647", always_xy=True)

        east, north = transformer.transform(
            self.data["longitude"].values,
            self.data["latitude"].values)

        self.data["longitude"] = east
        self.data["latitude"] = north

    def projection_seismic(self, width_seismic, default_depth_error_km=2.0):
        """Project seismic catalogue onto profile.

        Returns
        -------
        abscisses : list of float
            Along-profile positions (same CRS as profile).
        depths : list of float
            Hypocentral depths (km).
        magnitudes : list of float
            Moment magnitudes.
        z_errors : list of float
            Depth uncertainty in metres, from 'depthError' column (km × 1000).
            Falls back to default_depth_error_km × 1000 when not available.
        times : list of float
            Decimal years (e.g. 2008.5) for colormap scaling.
        """
        # Parse time column — handles USGS, Sun2012 and Zha2013 formats
        decimal_years = np.array([_parse_seismic_time(t) for t in self.data["time"]])

        has_z_err = "depthError" in self.data.columns
        default_z_m = default_depth_error_km * 1000.

        abscisses  = []
        depths     = []
        magnitudes = []
        z_errors   = []
        times      = []

        for i in range(len(self.data)):
            point = (self.data["longitude"][i], self.data["latitude"][i])
            proj  = self.profile.get_projection(point, width_seismic)
            if proj is not None:
                xpp, ypp = proj
                abscisses.append(xpp)
                depths.append(self.data["depth"][i])
                magnitudes.append(self.data["mag"][i])
                z_raw = self.data["depthError"][i] if has_z_err else np.nan
                z_errors.append(float(z_raw) * 1000 if pd.notna(z_raw) else default_z_m)
                times.append(decimal_years[i])

        return abscisses, depths, magnitudes, z_errors, times
