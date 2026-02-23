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
        self.data = pd.read_csv(self.path, sep=r",") # si virgule

        ### coordinates change :
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32647",always_xy=True)

        east, north = transformer.transform(
            self.data["longitude"].values, 
            self.data["latitude"].values)

        self.data["longitude"] = east
        self.data["latitude"] = north


    def projection_seismic(self, width_seismic):

        abscisses = []
        depths = []
        magnitudes = []

        for i in range(len(self.data)):

            point = (self.data["longitude"][i], self.data["latitude"][i])

            proj = self.profile.get_projection(point, width_seismic)

            if proj is not None:
                xpp, ypp = proj
                abscisses.append(xpp)
                depths.append(self.data["depth"][i])
                magnitudes.append(self.data["mag"][i])

        return abscisses, depths, magnitudes



