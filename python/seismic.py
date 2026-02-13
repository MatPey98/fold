# -*- coding:utf-8 -*-
__projet__ = "Pendages"
__nom_fichier__ = "seismic"
__author__ = "Mathieu Peyrache"
__date__ = "février 2026"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Transformer

class Seismic:
    def __init__(self, filename, wdir, profile):
        """

        :param filename:
        :param wdir:
        """
        self.path = wdir + filename
        self.profile = profile
        # self.data = pd.read_csv(self.path, sep=r",", header=None) # Si espace 
        # self.data = pd.read_csv(self.path, sep=r",", header=None) # si virgule
        self.data = pd.read_csv(self.path, sep=r"\t") # Si tabulation 
        # self.data.columns = ["date","magnitude","x","y","depth"]
        self.data.columns = ["date", "y", "x", "depth", "magnitude"]
        

    def projection_seismic(self, width_seismic):

        abscisses = []
        depths = []
        magnitudes = []

        for i in range(len(self.data)):

            point = (self.data["x"][i], self.data["y"][i])

            proj = self.profile.get_projection(point, width_seismic)

            if proj is not None:
                xpp, ypp = proj
                abscisses.append(xpp)
                depths.append(self.data["depth"][i])
                magnitudes.append(self.data["magnitude"][i])

        return abscisses, depths, magnitudes


    def print_seismic(self, abs_seismic, prof_seismic, mag):
        plt.scatter(
            abs_seismic,
            -np.array(prof_seismic),  # profondeur vers le bas
            c=mag,
            cmap="hot"
        )

