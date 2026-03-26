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
                x, y = self.raster.xy(row, col)
                proj = self.profile.get_projection((x, y), width)

                if proj is not None:
                    xpp, ypp = proj
                    value = band[row, col]

                    if not np.isnan(value) and np.isfinite(value):
                        abscisses_insar.append(xpp)
                        velocities.append(value)

        return np.array(abscisses_insar), np.array(velocities)

    def insar_statistics(self, width, nbins=100):
        """
        Calcule la médiane et l'écart-type des vitesses InSAR le long du profil.
        Ignore les valeurs nulles, NaN, inf et -inf.
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
        self.path_err = wdir + filename_err
        self.raster = rasterio.open(self.path)
        self.raster_err = rasterio.open(self.path_err)

    def elevations(self, coord):
        elevations = []
        for pt in coord:
            lig, col = self.raster.index(pt[0], pt[1])
            window = rasterio.windows.Window(col, lig, 1, 1)
            try:
                elevation = self.raster.read(1, window=window)[0, 0]
            except IndexError:
                elevations.append("NaN")
                continue
            elevations.append(elevation)
        return elevations

    def elevations_err(self, coord):
        elevations_err = []
        for pt in coord:
            lig, col = self.raster_err.index(pt[0], pt[1])
            window = rasterio.windows.Window(col, lig, 1, 1)
            try:
                elevation = self.raster_err.read(1, window=window)[0, 0]
            except IndexError:
                elevations_err.append("NaN")
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
        self.data = pd.read_csv(self.path, sep=r",") # si virgule
        # self.data = self.data.dropna(subset=["latitude", "longitude", "depth", "mag"])

        ### coordinates change :
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32647",always_xy=True)

        east, north = transformer.transform(
            self.data["longitude"].values,
            self.data["latitude"].values)

        self.data["longitude"] = east
        self.data["latitude"] = north


    def projection_seismic(self, width_seismic):

        abscisses = []
        depths = []
        magnitudes = []
        rms_list = []  # Nouvelle liste pour stocker les valeurs RMS

        for i in range(len(self.data)):

            point = (self.data["longitude"][i], self.data["latitude"][i])

            proj = self.profile.get_projection(point, width_seismic)

            if proj is not None:
                xpp, ypp = proj
                abscisses.append(xpp)
                depths.append(self.data["depth"][i])
                magnitudes.append(self.data["mag"][i])
                rms_list.append(self.data["rms"][i]*1000)  # Ajoutez la valeur RMS

        return abscisses, depths, magnitudes, rms_list