#!/usr/bin/env python3
# -*- coding:utf-8 -*-
__projet__ = "Fold"
__nom_fichier__ = "bayes_fold_inversion"
__author__ = "Mathieu Peyrache"
__date__ = "mars 2026"

import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from cinematic import compute_fault_and_axial_surfaces

class BayesFoldInversion:
    """
    Classe pour effectuer une inversion bayésienne des paramètres de faille
    en utilisant les données InSAR.
    """

    def __init__(self, x_obs, y_obs, y_err, params_template):
        """
        Initialise l'inversion bayésienne.

        Args:
            x_obs (np.ndarray): Abscisses des données InSAR observées.
            y_obs (np.ndarray): Vitesses InSAR observées.
            y_err (np.ndarray): Incertitudes sur les vitesses InSAR.
            params_template (dict): Dictionnaire de paramètres initiaux pour le modèle.
        """
        self.x_obs = x_obs
        self.y_obs = y_obs
        self.y_err = y_err
        self.params_template = params_template

    def likelihood(self, theta):
        """
        Calcule la log-vraisemblance des paramètres theta.

        Args:
            theta (list): Liste des paramètres à tester.

        Returns:
            float: Log-vraisemblance.
        """
        params = self.params_template.copy()
        params["beta"] = theta[0]
        params["teta"] = theta[1]
        params["omega"] = theta[2]
        params["W"] = theta[3]
        params["W2"] = theta[4]
        params["Ypref"] = theta[5]
        params["Zdec"] = theta[6]
        params["sigma"] = theta[7]
        params["Smax"] = theta[8]

        params["Ymax"] = theta[10]
        params["Ymax2"] = theta[11]

        results = compute_fault_and_axial_surfaces(params)
        y_model = results["Z_save"]

