import math
import os

import geopandas
import matplotlib.pyplot as plt
import numpy as np


class Dip:
    def __init__(self, wdir):
        self.wdir = wdir
        items = os.listdir(self.wdir)
        self.items = [item for item in items if item.endswith(".shp")]

    def load_points(self, filename, mnt):
        points = []
        file = geopandas.read_file(self.wdir + filename)
        for pt in file.geometry:
            # points.append([float(pt.x), float(pt.y)])
            points.append([pt.x, pt.y])
        for i in range(len(points)):
            # points[i].append(float(mnt.elevations([points[i]])[0]))
            points[i].append(mnt.elevations([points[i]])[0])
        self.points = points

    def fit_plane(self):
        points = np.array(self.points)
        A = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
        B = points[:, 2]
        coefficients = np.linalg.lstsq(A, B, rcond=None)[0]
        return coefficients

    def calculate_dip(self, a, b, c):
        norm_n = np.sqrt(a ** 2 + b ** 2 + c ** 2)
        cos_theta = abs(c) / norm_n
        theta_rad = np.arccos(cos_theta)
        theta_deg = np.degrees(theta_rad)
        self.dip = theta_deg

    def calculate_azimuth(self, a, b):
        azimut_rad = math.atan2(-a, -b)
        azimut_deg = math.degrees(azimut_rad)
        if azimut_deg < 0:
            azimut_deg += 360
        self.azimuth = azimut_deg

    def projeter_pendage(self, profile):
        phi = abs(profile.azimuth - self.azimuth) % 360
        if phi > 180:
            phi = 360 - phi
        pendage_reel_rad = np.radians(self.dip)
        phi_rad = np.radians(phi)
        pendage_projete_rad = np.arctan(np.tan(pendage_reel_rad) * np.cos(phi_rad))
        pendage_projete_deg = np.degrees(pendage_projete_rad)
        if pendage_projete_deg > 0:
            pendage_projete_deg = -pendage_projete_deg
        elif pendage_projete_deg < 0:
            pendage_projete_deg = 180 - pendage_projete_deg
        self.dip = pendage_projete_deg

    def point_median(self):
        points_array = np.array(self.points)
        # points_array = points_array.astype(float) 
        # print("points_array[:, 0]",points_array[:, 0])
        median_x = np.median(points_array[:, 0])
        median_y = np.median(points_array[:, 1])
        self.x, self.y = median_x, median_y

    def find_intersection(self, profile):
        if abs(self.dip) < 5:
            m1 = 1 / np.tan(np.radians(profile.azimuth + 90))
            m2 = 1 / np.tan(np.radians(profile.azimuth))
            x = (m1 * self.x - self.y - m2 * profile.start_point[0] + profile.start_point[1]) / (m1 - m2)
            y = m1 * (x - self.x) + self.y
        else:
            m1 = 1 / np.tan(np.radians(self.azimuth + 90))
            m2 = 1 / np.tan(np.radians(profile.azimuth))
            x = (m1 * self.x - self.y - m2 * profile.start_point[0] + profile.start_point[1]) / (m1 - m2)
            y = m1 * (x - self.x) + self.y
        self.intersect = [[x, y]]

    def print(self, l, x, y):
        if self.azimuth >= 0:
            dx = l * np.cos(np.radians(self.dip))
            dy = l * np.sin(np.radians(self.dip))
            x2 = x + dx
            y2 = y + dy
            plt.plot([x, x2], [y, y2], "k")

    def print_all(self, topodata, profile, l, x0):
        for filename in self.items:
            self.load_points(filename, topodata)
            self.point_median()
            a, b, c = self.fit_plane()
            self.calculate_dip(a, b, -1)
            self.calculate_azimuth(a, b)
            self.projeter_pendage(profile)
            self.find_intersection(profile)
            x = np.sqrt(self.intersect[0][0] ** 2 + self.intersect[0][1] ** 2) - x0
            y = topodata.elevations(self.intersect)
            if y[0] != "NaN":
                self.print(l, x, y)
