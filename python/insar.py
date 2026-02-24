import rasterio
import rasterio.windows


class Insar:

    def __init__(self, filename, wdir):
        self.path = wdir + filename
        self.raster = rasterio.open(self.path)

    def projection_insar(self):
        
        band = self.raster.read(1)
        rows, cols = band.shape

        abscisses_insar = []
        velocities = []

        for row in range(rows):
            for col in range(cols):

                x, y = self.raster.xy(row, col)
                proj = profile.get_projection((x, y), width)

                if proj is not None:

                    xpp, ypp = proj
                    value = band[row, col]

                    if not np.isnan(value):
                        abscisses_insar.append(xpp)
                        velocities.append(value)
        
        return abscisses_insar, velocities


