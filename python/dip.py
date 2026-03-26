import math
import os
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lst
import scipy.optimize as opt


class Dip:
    """
    Lis les points géographiques depuis un fichier SIG
    Récupère les altitudes depuis le MNT
    Ajuste un plan 3D sur les points (x, y, z)
    Puis calcul le pendage de ce plan
    """ 
    def __init__(self, wdir):
        """
        Liste tous les fichiers d'un dossier et récupère les fichiers "shp"
        :param wdir: répertoire de travail
        """
        self.wdir = wdir
        items = os.listdir(self.wdir)
        self.items = [item for item in items if item.endswith(".shp")]


    def load_points(self, filename, mnt):
        """
        Charge les points sur le mnt
        :param filename: Nom du fichier contenant les points pour le pendage
        :param mnt: Modèle d'élévation Pléiades
        :return: Les points (x, y, z)
        """
        file = geopandas.read_file(self.wdir + filename)

        points = [] 
        sigma = []

        for pt in file.geometry:
            x, y = pt.x, pt.y
            z = mnt.elevations([[x,y]])[0]
            err = mnt.elevations_err([[x,y]])[0]

            points.append([x,y,z])
            sigma.append(err)

        self.points = points
        self.sigma = sigma


    def fit_plane(self):
        """
        Méthode des moindres carrées pondéré par l'erreur de triangulation du MNT pour retrouver les coefficients du plan
        :return: les coefficients de l'équation du plan c = ax + by - z
        """
        ## pris dans PyGdalSAR, invers_disp2coef.py lignes 1191 à 1259
        points = np.array(self.points)
        sigma = np.array(self.sigma)
        
        A = np.c_[points[:, 0], points[:, 1], - np.ones(points.shape[0])]
        b = points[:, 2]

        W = np.diag(1.0 / sigma) # poids pour la pondération
        coefficients = np.linalg.lstsq(W @ A, W @ b, rcond=None)[0]

        try:
            varx = np.linalg.pinv(A.T @ A)
            res2 = np.sum((b - A @ coefficients) ** 2)
            scale = 1. / (A.shape[0] - A.shape[1])
            sigmam = np.sqrt(scale * res2 * np.diag(varx)) # écart type estimé sur a, b, c
        except np.linalg.LinAlgError:
            sigmam = np.full(A.shape[1], np.nan)

        return coefficients, sigmam


    def propagate_uncertainties(self, coefficients, sigmam, n=1000):
        """
        Propagation Monte-Carlo des incertitudes
        """

        a0, b0, c0 = coefficients
        sigma_a, sigma_b, sigma_c = sigmam

        # Tirages aléatoires
        a_samples = np.random.normal(a0, sigma_a, n)
        b_samples = np.random.normal(b0, sigma_b, n)

        return a_samples, b_samples


    def calculate_dip(self, a, b, c):
        """
        Calcul le pendage par rapport à la norme "n" du plan et l'horizontale
        """
        norm_n = np.sqrt(a ** 2 + b ** 2 + c ** 2)
        cos_theta = c / norm_n
        theta_rad = np.arccos(cos_theta)
        theta_deg = np.degrees(theta_rad)
        if theta_deg > 90: # Corriger les pendages au dessus de 90°, éviter les inversions de sens
            theta_deg = 180 - theta_deg
        print(f'dip : {theta_deg}')
        self.dip = theta_deg
        return theta_deg


    def calculate_azimuth(self, a, b):
        """
        Calcul de l'azimut du plan
        """
        azimut_rad = math.atan2(-a, -b) # Valeur strike entre 0 et 90°
        azimut_deg = math.degrees(azimut_rad)
        if azimut_deg < 0: # Ajoute les valeurs de strike entre 270 et 360° -> complète les valeurs manquantes de strike
            azimut_deg += 360
        self.azimuth = azimut_deg
        return azimut_deg


    def projeter_pendage(self, profile):
        """
        :param profile: profile tracé sur un GIS
        :return: Le pendage projeté sur ce profile
        """
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
        """

        :return:
        """
        points_array = np.array(self.points)
        median_x = np.median(points_array[:, 0])
        median_y = np.median(points_array[:, 1])
        self.x, self.y = median_x, median_y


    def find_intersection(self, profile):
        """

        :param profile:
        :return:
        """
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

# ======================================================================================================================
# Plot strata
# ======================================================================================================================

    def print(self, length_dip, x, y, ax):
        """
        plot strata 
        :param length_dip:
        :param x:
        :param y:
        :return:
        """
        if self.azimuth >= 0:
            dx = length_dip * np.cos(np.radians(self.dip))
            dy = length_dip * np.sin(np.radians(self.dip))
            x2 = x - dx
            y2 = y + dy
            ax.plot([x, x2], [y, y2], color='blue', zorder=10)


    def print_all(self, topodata, profile, length_dip, ax):
        """
        Pour effectuer la projection des pendages
        :param topodata:
        :param profile:
        :param length_dip:
        :param x0:
        :return:
        """
        for filename in self.items:
            self.load_points(filename, topodata)
            self.point_median()
            coefficients, sigmam = self.fit_plane()
            a, b, c = coefficients
            
            # pendage moyen
            self.calculate_dip(a, b, -1)
            self.calculate_azimuth(a, b)
            self.projeter_pendage(profile)
            self.find_intersection(profile)
            proj = profile.get_projection_all(self.intersect[0])
            x = np.max(profile.abscisse) - proj[0]
            y = topodata.elevations(self.intersect)
            if y[0] != "NaN":
                self.print(length_dip, x, y, ax)
            
            ### tout refaire pour la méthode monte carlo pour pas écraser les calculs avec self (refaire de nouvelles méthodes?)
            a_samples, b_samples = self.propagate_uncertainties(coefficients, sigmam, n=20)

            for a_i, b_i in zip(a_samples, b_samples):

                # calcul sans modifier dip
                norm = np.sqrt(a_i**2 + b_i**2 + 1)
                dip = np.degrees(np.arccos(1 / norm))
                if dip > 90:
                    dip = 180 - dip

                az = np.degrees(np.arctan2(-a_i, -b_i))
                if az < 0:
                    az += 360

                # projection
                phi = abs(profile.azimuth - az) % 360
                if phi > 180:
                    phi = 360 - phi

                dip_rad = np.radians(dip)
                phi_rad = np.radians(phi)
                dip_proj = np.degrees(
                    np.arctan(np.tan(dip_rad) * np.cos(phi_rad))
                )

                if dip_proj > 0:
                    dip_proj = -dip_proj
                elif dip_proj < 0:
                    dip_proj = 180 - dip_proj

                # intersection simplifiée
                m1 = 1 / np.tan(np.radians(az + 90))
                m2 = 1 / np.tan(np.radians(profile.azimuth))

                x_int = (m1 * self.x - self.y
                        - m2 * profile.start_point[0]
                        + profile.start_point[1]) / (m1 - m2)

                y_int = m1 * (x_int - self.x) + self.y

                intersect = [[x_int, y_int]]

                proj = profile.get_projection_all(intersect[0])
                x_mc = np.max(profile.abscisse) - proj[0]
                y_mc = topodata.elevations(intersect)

                if y_mc[0] != "NaN":

                    dx = length_dip * np.cos(np.radians(dip_proj))
                    dy = length_dip * np.sin(np.radians(dip_proj))

                    x2 = x_mc - dx
                    y2 = y_mc + dy

                    ax.plot([x_mc, x2], [y_mc, y2], color='lightblue', alpha=0.5)

# ======================================================================================================================
# Plot fault
# ======================================================================================================================

    def print_fault(self, length_dip, x, y, ax):
        """
        plot fault
        :param length_dip:
        :param x:
        :param y:
        :return:
        """
        if self.azimuth >= 0:
            dx = length_dip * np.cos(np.radians(self.dip))
            dy = length_dip * np.sin(np.radians(self.dip))
            x2 = x - dx
            y2 = y + dy
            ax.plot([x, x2], [y, y2], color='red')


    def print_all_fault(self, topodata, profile, length_dip, ax):
        """
        Pour effectuer la projection des pendages
        :param topodata:
        :param profile:
        :param length_dip:
        :param x0:
        :return:
        """
        for filename in self.items:
            self.load_points(filename, topodata)
            self.point_median()
            coefficients, sigmam = self.fit_plane()
            a, b, c = coefficients
            
            # pendage moyen
            self.calculate_dip(a, b, -1)
            self.calculate_azimuth(a, b)
            self.projeter_pendage(profile)
            self.find_intersection(profile)
            proj = profile.get_projection_all(self.intersect[0])
            x = np.max(profile.abscisse) - proj[0]
            y = topodata.elevations(self.intersect)
            if y[0] != "NaN":
                self.print_fault(length_dip, x, y, ax)
            
            ### tout refaire pour la méthode monte carlo pour pas écraser les calculs avec self (refaire de nouvelles méthodes?)
            a_samples, b_samples = self.propagate_uncertainties(coefficients, sigmam, n=20)

            for a_i, b_i in zip(a_samples, b_samples):

                # calcul sans modifier dip
                norm = np.sqrt(a_i**2 + b_i**2 + 1)
                dip = np.degrees(np.arccos(1 / norm))
                if dip > 90:
                    dip = 180 - dip

                az = np.degrees(np.arctan2(-a_i, -b_i))
                if az < 0:
                    az += 360

                # projection
                phi = abs(profile.azimuth - az) % 360
                if phi > 180:
                    phi = 360 - phi

                dip_rad = np.radians(dip)
                phi_rad = np.radians(phi)
                dip_proj = np.degrees(
                    np.arctan(np.tan(dip_rad) * np.cos(phi_rad))
                )

                if dip_proj > 0:
                    dip_proj = -dip_proj
                elif dip_proj < 0:
                    dip_proj = 180 - dip_proj

                # intersection simplifiée
                m1 = 1 / np.tan(np.radians(az + 90))
                m2 = 1 / np.tan(np.radians(profile.azimuth))

                x_int = (m1 * self.x - self.y
                        - m2 * profile.start_point[0]
                        + profile.start_point[1]) / (m1 - m2)

                y_int = m1 * (x_int - self.x) + self.y

                intersect = [[x_int, y_int]]

                proj = profile.get_projection_all(intersect[0])
                x_mc = np.max(profile.abscisse) - proj[0]
                y_mc = topodata.elevations(intersect)

                if y_mc[0] != "NaN":

                    dx = length_dip * np.cos(np.radians(dip_proj))
                    dy = length_dip * np.sin(np.radians(dip_proj))

                    x2 = x_mc - dx
                    y2 = y_mc + dy

                    ax.plot([x_mc, x2], [y_mc, y2], color='red', alpha=0.5)