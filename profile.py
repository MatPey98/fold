import math
import geopandas


class Profile:
    def __init__(self, filename, wdir):
        self.path = wdir + filename

    def get_extreme_points(self):
        file = geopandas.read_file(self.path)
        polyline = file.geometry.iloc[0]
        self.start_point = polyline.coords[0]
        self.end_point = polyline.coords[-1]

    def get_azimuth(self):
        dx = self.end_point[0] - self.start_point[0]
        dy = self.end_point[1] - self.start_point[1]
        azimut_rad = math.atan2(dx, dy)
        azimut_deg = math.degrees(azimut_rad)
        if azimut_deg < 0:
            azimut_deg += 360
        self.azimuth = azimut_deg

    def linspace(self, n):
        points = []
        x1, y1 = self.start_point
        x2, y2 = self.end_point
        for i in range(n + 1):
            points.append([x1 + i * (x2 - x1) / n, y1 + i * (y2 - y1) / n])
        self.points = points