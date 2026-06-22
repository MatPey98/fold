import rasterio
import rasterio.windows
import numpy as np
import pandas as pd
from pyproj import Transformer

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

    def projection_seismic(self, width_seismic):

        abscisses = []
        depths = []
        magnitudes = []
        rms_list = []  # List to store RMS values

        for i in range(len(self.data)):

            point = (self.data["longitude"][i], self.data["latitude"][i])

            proj = self.profile.get_projection(point, width_seismic)

            if proj is not None:
                xpp, ypp = proj
                abscisses.append(xpp)
                depths.append(self.data["depth"][i])
                magnitudes.append(self.data["mag"][i])
                rms_list.append(self.data["rms"][i] * 1000)  # Append RMS value

        return abscisses, depths, magnitudes, rms_list
