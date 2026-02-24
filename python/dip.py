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
        Méthode des moindres carrées afin de fitter un plan au nuage de points
        :return: les coefficients de l'équation du plan c = ax + by - z
        """
        # points = np.array(self.points)
        # A = np.c_[points[:, 0], points[:, 1], - np.ones(points.shape[0])]
        # B = points[:, 2]
        # coefficients = np.linalg.lstsq(A, B, rcond=None)[0]
        # return coefficients

        points = np.array(self.points)
        sigma = np.array(self.sigma)
        # sigma = 1    

        G = np.c_[points[:, 0], points[:, 1], - np.ones(points.shape[0])]
        
        data = points[:, 2]

        x0 = lst.lstsq(G, data, rcond=None)[0]

        _func = lambda x: np.sum(((np.dot(G,x)-data)/sigma)**2)
        _fprime = lambda x: 2*np.dot(G.T/sigma, (np.dot(G,x)-data)/sigma)

        coefficients = opt.fmin_slsqp(_func,x0,fprime=_fprime,iter=2000,full_output=True,iprint=0,acc=1.e-9)[0]

        ### Covariance
        Gw = G / sigma[:,None]
        Cm = np.linalg.inv(Gw.T @ Gw)

        self.G = G
        self.Cm = Cm

        return coefficients


    def calculate_dip(self, a, b, c):
        """
        Calcul le pendage par rapport à la norme "n" du plan et l'horizontale
        :param a: coefficient a (est) de l'équation de plan
        :param b: coefficient b (nord) de l'équation de plan
        :param c: coefficient c de l'équation de plan
        """
        norm_n = np.sqrt(a ** 2 + b ** 2 + c ** 2)
        cos_theta = c / norm_n
        theta_rad = np.arccos(cos_theta)
        theta_deg = np.degrees(theta_rad)
        if theta_deg > 90: # Corriger les pendages au dessus de 90°, éviter les inversions de sens
            theta_deg = 180 - theta_deg
        print(f'dip : {theta_deg}')
        self.dip = theta_deg


    def calculate_azimuth(self, a, b):
        """
        Calcul de l'azimut du plan
        :param a: coefficient a de l'équation de plan
        :param b: coefficient b de l'équation de plan
        """
        azimut_rad = math.atan2(-a, -b) # Valeur strike entre 0 et 90°
        azimut_deg = math.degrees(azimut_rad)
        if azimut_deg < 0: # Ajoute les valeurs de strike entre 270 et 360° -> complète les valeurs manquantes de strike
            azimut_deg += 360
        self.azimuth = azimut_deg

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


    def monte_carlo(self, coefficients, n=20):
        samples = np.random.multivariate_normal(coefficients, self.Cm, n)
        return samples 


    def plot_uncertainties(self, samples, profile, topodata, length_dip, x0):
        
        original_dip = self.dip
        original_az = self.azimuth

        for a,b,c in samples:
            self.calculate_dip(a,b,-1)
            self.calculate_azimuth(a,b)
            self.projeter_pendage(profile)
            self.find_intersection(profile)

            proj = profile.get_projection_all(self.intersect[0])
            x = proj[0]
            y = topodata.elevations(self.intersect)

            if y[0] != "NaN":
                
                dx = length_dip * np.cos(np.radians(self.dip))
                dy = length_dip * np.sin(np.radians(self.dip))

                x2 = x + dx
                y2 = y + dy
                plt.plot([x, x2], [y, y2], color='red', alpha=0.15)
        
        self.dip = original_dip
        self.azimuth = original_az


    def print(self, length_dip, x, y):
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
            x2 = x + dx
            y2 = y + dy
            plt.plot([x, x2], [y, y2], color='blue', zorder=10)


    def print_all(self, topodata, profile, length_dip, x0):
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
            a, b, c = self.fit_plane()
            self.calculate_dip(a, b, -1)
            self.calculate_azimuth(a, b)
            self.projeter_pendage(profile)
            self.find_intersection(profile)
            #x = np.sqrt(self.intersect[0][0] ** 2 + self.intersect[0][1] ** 2) - x0
            proj = profile.get_projection_all(self.intersect[0])
            x = proj[0]
            y = topodata.elevations(self.intersect)
            if y[0] != "NaN":
                self.print(length_dip, x, y)
            
            samples = self.monte_carlo([a,b,c], n=20)
            self.plot_uncertainties(samples, profile, topodata, length_dip, x0)


    def print_fault(self, length_dip, x, y):
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
            x2 = x + dx
            y2 = y + dy
            plt.plot([x, x2], [y, y2], color='red')

    def print_all_fault(self, topodata, profile, length_dip, x0):
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
            a, b, c = self.fit_plane()
            self.calculate_dip(a, b, -1)
            self.calculate_azimuth(a, b)
            self.projeter_pendage(profile)
            self.find_intersection(profile)
            #x = np.sqrt(self.intersect[0][0] ** 2 + self.intersect[0][1] ** 2) - x0
            proj = profile.get_projection_all(self.intersect[0])
            x = proj[0]
            y = topodata.elevations(self.intersect)
            if y[0] != "NaN":
                self.print_fault(length_dip, x, y)