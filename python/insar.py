import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.windows


class Insar:
    def __init__(self, filename, wdir):
        self.path = wdir + filename
        self.raster = rasterio.open(self.path)

    def velocities(self, coord):
        velocities = []
        for pt in coord:
            lig, col = self.raster.index(pt[0], pt[1])
            window = rasterio.windows.Window(col, lig, 1, 1)
            velocity = self.raster.read(1, window=window)[0, 0]
            velocities.append(velocity)
        return velocities

    def smooth_profile(self, abs, v):
        last = 0
        new_abs = []
        new_v = [v[0]]
        for i in range(1, len(v)):
            if v[i - 1] != v[i]:
                list = abs[last : i]
                new_abs.append(np.median(list))
                new_v.append(v[i])
                last = i
        list = abs[last : len(abs)]
        new_abs.append(np.median(list))
        return new_abs, new_v

    def print(self, abs, elevations):
        plt.plot(abs, elevations, "r")