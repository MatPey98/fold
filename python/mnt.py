import matplotlib.pyplot as plt
import rasterio
import rasterio.windows


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

    def print(self, abs, elevations):
        plt.plot(abs, elevations, "k")
