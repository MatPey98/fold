import rasterio
import rasterio.windows
import numpy as np


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

                    if not np.isnan(value):
                        abscisses_insar.append(xpp)
                        velocities.append(value)

        return np.array(abscisses_insar), np.array(velocities)


    def insar_statistics(self, width, nbins=100):
        """
        Calcule la médiane et l'écart-type des vitesses InSAR le long du profil.
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
