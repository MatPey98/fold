import math
import geopandas
import numpy as np

class Profile:
    def __init__(self, filename, wdir, w):
        self.path = wdir + filename
        self.w=w*1e3

        self.get_extreme_points()
        self.get_azimuth()
        self.referential_change()
        self.get_distance_profil()


    def get_extreme_points(self):
        """
        :return: Le point de départ du tracé de du profil et sa terminaison (il doit être orienté sud-nord).
        """
        file = geopandas.read_file(self.path)
        polyline = file.geometry.iloc[0]

        self.start_point = polyline.coords[0]
        self.end_point = polyline.coords[-1]


    def get_azimuth(self):
        """
        :return: l'azimut du profile
        """
        dx = self.end_point[0] - self.start_point[0]
        dy = self.end_point[1] - self.start_point[1]

        azimut_rad = math.atan2(dx, dy)
        azimut_deg = math.degrees(azimut_rad)

        if azimut_deg < 0:
            azimut_deg += 360

        self.azimuth = azimut_deg


    def referential_change(self):
        """
        changement de référentiel en prenant x : le long du profil, et y : perpendiculaire à ce profil.
        :return:
        """
        theta = math.radians(self.azimuth)

        self.s = np.array([np.sin(theta), np.cos(theta)])
        self.n = np.array([np.cos(theta), -np.sin(theta)])


    def get_distance_profil(self):
        """

        :return:
        """
        self.l = np.sqrt(
            (self.end_point[0] - self.start_point[0]) ** 2 + (self.end_point[1] - self.start_point[1]) ** 2)


    def linspace(self, n):
        """
        :param n: Nombre de points
        :return: n + 1 points régulièrement espacés le long du profil.
        """
        points = []
        x1, y1 = self.start_point
        x2, y2 = self.end_point

        for i in range(n + 1):
            points.append([x1 + i * (x2 - x1) / n, y1 + i * (y2 - y1) / n])

        self.points = points
        self.abscisse = np.linspace(0, self.l, n + 1)


    def get_projection(self, point, width):
        """
        Projection d'un point dans le référentiel du profil
        """
        x0, y0 = self.start_point
        x, y = point

        # vecteur relatif
        dx = x - x0
        dy = y - y0

        # projection dans la base locale
        self.xpp = dx * self.s[0] + dy * self.s[1] # dans l'allongement du profil
        self.ypp = dx * self.n[0] + dy * self.n[1] # Perpendiculaire au profil

        if self.xpp > self.l or self.xpp < 0 or abs(self.ypp) > width/2:
            return None
        else :
            return self.xpp, self.ypp


    def get_projection_all(self, point):
        """
        rotation des points dans le référentiel du profil sans tri
        """
        x0, y0 = self.start_point
        x, y = point

        # vecteur relatif
        dx = x - x0
        dy = y - y0

        # projection dans la base locale
        self.xpp = dx * self.s[0] + dy * self.s[1] # dans l'allongement du profil
        self.ypp = dx * self.n[0] + dy * self.n[1] # Perpendiculaire au profil

        return self.xpp, self.ypp
