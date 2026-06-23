import math
import os
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lst
import scipy.optimize as opt


class Dip:
    """
    Reads geographic points from a GIS shapefile, retrieves elevations from a DEM,
    fits a 3D plane to the points (x, y, z), and computes the dip and azimuth
    of that plane. Supports Monte Carlo uncertainty propagation.
    """

    def __init__(self, wdir):
        """
        Lists all shapefiles in a directory.

        Parameters
        ----------
        wdir : str
            Working directory containing the shapefiles.
        """
        self.wdir = wdir
        items = os.listdir(self.wdir)
        self.items = [item for item in items if item.endswith(".shp")]

    def load_points(self, filename, mnt):
        """
        Loads points from a shapefile and retrieves their elevations from a DEM.

        Parameters
        ----------
        filename : str
            Shapefile name (inside wdir).
        mnt : MNT
            DEM object used to query elevation and elevation error.
        """
        file = geopandas.read_file(self.wdir + filename)

        points = []
        sigma  = []

        for pt in file.geometry:
            x, y = pt.x, pt.y
            z   = mnt.elevations([[x, y]])[0]
            err = mnt.elevations_err([[x, y]])[0]
            points.append([x, y, z])
            sigma.append(err)

        self.points = points
        self.sigma  = sigma

    def fit_plane(self):
        """
        Weighted least-squares fit of a plane through the loaded points.
        Weights are the inverse of DEM triangulation errors.

        Returns
        -------
        coefficients : ndarray, shape (3,)
            Plane coefficients [a, b, c] such that z = a*x + b*y + c.
        sigmam : ndarray, shape (3,)
            Estimated standard deviations on [a, b, c].
        """
        points = np.array(self.points)
        sigma  = np.array(self.sigma, dtype=float)
        sigma  = np.clip(sigma, 1e-6, None)  # Avoid division by zero

        A = np.c_[points[:, 0], points[:, 1], -np.ones(len(points))]
        b = points[:, 2]

        W            = np.diag(1.0 / sigma)
        coefficients = np.linalg.lstsq(W @ A, W @ b, rcond=None)[0]

        try:
            varx   = np.linalg.pinv(A.T @ A)
            res2   = np.sum((b - A @ coefficients) ** 2)
            scale  = 1.0 / (A.shape[0] - A.shape[1])
            sigmam = np.sqrt(scale * res2 * np.diag(varx))
        except np.linalg.LinAlgError:
            sigmam = np.full(A.shape[1], np.nan)

        return coefficients, sigmam

    def propagate_uncertainties(self, coefficients, sigmam, n=1000):
        """
        Monte Carlo uncertainty propagation on plane coefficients.

        Parameters
        ----------
        coefficients : array-like, shape (3,)
        sigmam : array-like, shape (3,)
        n : int
            Number of Monte Carlo draws.

        Returns
        -------
        a_samples, b_samples : ndarray
        """
        a0, b0, c0       = coefficients
        sigma_a, sigma_b, sigma_c = sigmam
        a_samples = np.random.normal(a0, sigma_a, n)
        b_samples = np.random.normal(b0, sigma_b, n)
        return a_samples, b_samples

    def calculate_dip(self, a, b, c):
        """
        Computes the dip angle of the plane relative to horizontal.

        Parameters
        ----------
        a, b, c : float
            Plane coefficients (z = a*x + b*y + c, note c = -1 in our convention).
        """
        norm_n    = np.sqrt(a**2 + b**2 + c**2)
        theta_deg = np.degrees(np.arccos(c / norm_n))
        if theta_deg > 90:
            theta_deg = 180 - theta_deg  # Avoid dip > 90° (sign flip)
        print(f"dip: {theta_deg:.1f}°")
        self.dip = theta_deg
        return theta_deg

    def calculate_azimuth(self, a, b):
        """
        Computes the strike azimuth of the plane (0–360°).
        """
        azimuth_deg = math.degrees(math.atan2(-a, -b))
        if azimuth_deg < 0:
            azimuth_deg += 360
        print(f"strike: {azimuth_deg:.1f}°")
        self.azimuth = azimuth_deg
        return azimuth_deg

    def project_dip(self, profile):
        """
        Projects the true dip onto the cross-section profile direction.

        Parameters
        ----------
        profile : Profile
        """
        phi = abs(profile.azimuth - self.azimuth) % 360
        if phi > 180:
            phi = 360 - phi

        projected = np.degrees(
            np.arctan(np.tan(np.radians(self.dip)) * np.cos(np.radians(phi)))
        )
        # Sign convention: dip toward the section is negative
        if projected > 0:
            projected = -projected
        elif projected < 0:
            projected = 180 - projected

        self.dip = projected

    def compute_median_point(self):
        """
        Sets self.x, self.y to the median coordinates of the loaded points.
        """
        pts = np.array(self.points)
        self.x = np.median(pts[:, 0])
        self.y = np.median(pts[:, 1])

    def find_intersection(self, profile):
        """
        Finds the intersection of the plane strike with the cross-section profile line.
        """
        if abs(self.dip) < 5:
            m1 = 1 / np.tan(np.radians(profile.azimuth + 90))
            m2 = 1 / np.tan(np.radians(profile.azimuth))
        else:
            m1 = 1 / np.tan(np.radians(self.azimuth + 90))
            m2 = 1 / np.tan(np.radians(profile.azimuth))

        x = (m1 * self.x - self.y - m2 * profile.start_point[0] + profile.start_point[1]) / (m1 - m2)
        y = m1 * (x - self.x) + self.y
        self.intersect = [[x, y]]

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _plot_dip_segment(self, length, x, y, ax, color, alpha=1.0):
        """Draws a single dip segment on the cross-section."""
        if self.azimuth >= 0:
            dx = length * np.cos(np.radians(self.dip))
            dy = length * np.sin(np.radians(self.dip))
            ax.plot([x, x - dx], [y, y + dy], color=color, alpha=alpha, zorder=10)

    def _plot_all(self, topodata, profile, length, ax,
                  mean_color, mc_color, mc_alpha=0.5):
        """
        Common logic for plotting mean dip + Monte Carlo uncertainty for all
        shapefiles in wdir. Used by plot_all_strata and plot_all_fault.
        """
        for filename in self.items:
            # --- Mean dip ---
            self.load_points(filename, topodata)
            self.compute_median_point()
            coefficients, sigmam = self.fit_plane()
            a, b, c = coefficients

            self.calculate_dip(a, b, -1)
            self.calculate_azimuth(a, b)
            self.project_dip(profile)
            self.find_intersection(profile)

            proj = profile.get_projection_all(self.intersect[0])
            x    = np.max(profile.abscisse) - proj[0]
            y    = topodata.elevations(self.intersect)

            if not np.isnan(float(y[0])):
                self._plot_dip_segment(length, x, y, ax, color=mean_color)

            # --- Monte Carlo uncertainty ---
            a_samples, b_samples = self.propagate_uncertainties(coefficients, sigmam, n=20)

            for a_i, b_i in zip(a_samples, b_samples):
                norm    = np.sqrt(a_i**2 + b_i**2 + 1)
                dip_i   = np.degrees(np.arccos(1 / norm))
                if dip_i > 90:
                    dip_i = 180 - dip_i

                az_i = np.degrees(np.arctan2(-a_i, -b_i))
                if az_i < 0:
                    az_i += 360

                phi = abs(profile.azimuth - az_i) % 360
                if phi > 180:
                    phi = 360 - phi

                dip_proj = np.degrees(
                    np.arctan(np.tan(np.radians(dip_i)) * np.cos(np.radians(phi)))
                )
                if dip_proj > 0:
                    dip_proj = -dip_proj
                elif dip_proj < 0:
                    dip_proj = 180 - dip_proj

                m1    = 1 / np.tan(np.radians(az_i + 90))
                m2    = 1 / np.tan(np.radians(profile.azimuth))
                x_int = (m1 * self.x - self.y
                         - m2 * profile.start_point[0]
                         + profile.start_point[1]) / (m1 - m2)
                y_int = m1 * (x_int - self.x) + self.y

                proj_mc = profile.get_projection_all([x_int, y_int])
                x_mc    = np.max(profile.abscisse) - proj_mc[0]
                y_mc    = topodata.elevations([[x_int, y_int]])

                if not np.isnan(float(y_mc[0])):
                    dx = length * np.cos(np.radians(dip_proj))
                    dy = length * np.sin(np.radians(dip_proj))
                    ax.plot([x_mc, x_mc - dx], [y_mc, y_mc + dy],
                            color=mc_color, alpha=mc_alpha)

    # =========================================================================
    # Public plotting methods
    # =========================================================================

    def plot_all_strata(self, topodata, profile, length, ax):
        """
        Projects and plots all strata dip measurements onto the cross-section,
        including Monte Carlo uncertainty realizations.
        """
        self._plot_all(topodata, profile, length, ax,
                       mean_color='blue', mc_color='lightblue')

    def plot_all_fault(self, topodata, profile, length, ax):
        """
        Projects and plots all fault dip measurements onto the cross-section,
        including Monte Carlo uncertainty realizations.
        """
        self._plot_all(topodata, profile, length, ax,
                       mean_color='red', mc_color='red', mc_alpha=0.5)
