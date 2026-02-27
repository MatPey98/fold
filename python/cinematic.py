import numpy as np
from scipy.optimize import fsolve

def compute_fault_and_axial_surfaces(params):
    # ============================================================================
    # PARAMETRES (modifiables via le dictionnaire params)
    # ============================================================================
    beta = np.deg2rad(params.get("beta", -45))  # Angle négatif pour un pendage nord
    teta = np.deg2rad(params.get("teta", -20))
    omega = np.deg2rad(params.get("omega", -5))
    sigma = np.deg2rad(params.get("sigma", -3.2))

    alpha = teta + ((beta - teta) / 2)
    coef = -1 / alpha

    alpha2 = omega + ((teta - omega) / 2)
    coef2 = -1 / alpha2

    Zdec = params.get("Zdec", 3476.9)  # Position verticale d'émergence
    ldec = params.get("ldec", 100)
    Ypref = params.get("Ypref", 1207.99)  # Position horizontale d'émergence
    W = params.get("W", 700)
    W2 = params.get("W2", 300)
    Ymin = params.get("Ymin", 0)
    Ymax = params.get("Ymax", 10000)
    n_strata = params.get("n_strata", 20)
    Zhaut = params.get("Zhaut", 1800)
    Ymax2 = params.get("Ymax2", 14000)
    di = params.get("di", 1000)
    dj = int((Ymax2 - Ymin - 1) / di)
    ite_s = params.get("ite_s", 1)

    debut = np.pi
    fin = 2 * np.pi
    angle = np.linspace(debut, fin, 10000)

    # ============================================================================
    # DEFINITION DES CHARNIERES
    # ============================================================================
    aramp1 = np.tan(beta)
    bramp1 = 3307 - (aramp1 * 20000)
    aramp2 = np.tan(teta)
    bramp2 = Zdec

    def ramp1(x):
        return aramp1 * x + bramp1

    def ramp2(x):
        return aramp2 * x + bramp2

    def inter_ramp(x):
        return ramp1(x) - ramp2(x)

    Ycrois = fsolve(inter_ramp, 4593000)[0]
    Zcrois = ramp1(Ycrois)

    Rc = (W / 2) / np.sin((beta - teta) / 2)
    Rc2 = (W2 / 2) / np.sin((teta - omega) / 2)

    hypo = (W / 2) / np.cos((beta - teta) / 2)
    Ych1 = Ycrois + hypo * np.cos(beta)
    Zch1 = Zcrois + hypo * np.sin(beta)
    Ych2 = Ycrois - hypo * np.cos(teta)
    Zch2 = Zcrois - hypo * np.sin(teta)
    Ych3 = Ypref
    Zpref = ramp2(Ypref)
    Zch3 = Zpref

    Ycentre = Ych1 - Rc * np.sin(beta)
    Zcentre = Zch1 + Rc * np.cos(beta)
    Ycentre2 = Ych3 - Rc2 * np.sin(teta)
    Zcentre2 = Zch3 + Rc2 * np.cos(teta)

    Ych4 = Ycentre2 + Rc2 * np.sin(omega)
    Zch4 = Zcentre2 - Rc2 * np.cos(omega)

    aramp3 = np.tan(omega)
    bramp3 = Zch4 - (aramp3 * Ych4)

    def ramp3(x):
        return aramp3 * x + bramp3

    abis = coef
    bbis = Zcentre - (abis * Ycentre)

    def biss(x):
        return abis * x + bbis

    abis2 = coef2
    bbis2 = Zcentre2 - (abis2 * Ycentre2)

    def biss2(x):
        return abis2 * x + bbis2

    def Cercle(x):
        return Zcentre - np.sqrt(np.maximum(0, -(x - Ycentre)**2 + Rc**2))

    def Cercle2(x):
        return Zcentre2 - np.sqrt(np.maximum(0, -(x - Ycentre2)**2 + Rc2**2))

    # ============================================================================
    # CALCUL DES COORDONNEES DE LA FAILLE
    # ============================================================================
    Yfaille = np.arange(Ymin, Ymax + 1, 1)
    taille = len(Yfaille)
    Zfaille = np.zeros(taille)

    for l in range(taille):
        if Yfaille[l] < Ych4:
            Zfaille[l] = ramp3(Yfaille[l])
        elif Yfaille[l] >= Ych4 and Yfaille[l] < Ych3:
            Zfaille[l] = Cercle2(Yfaille[l])
        elif Yfaille[l] < Ych2 and Yfaille[l] >= Ych3:
            Zfaille[l] = ramp2(Yfaille[l])
        elif Yfaille[l] >= Ych2 and Yfaille[l] < Ych1:
            Zfaille[l] = Cercle(Yfaille[l])
        elif Yfaille[l] >= Ych1:
            Zfaille[l] = ramp1(Yfaille[l])

    # ============================================================================
    # DEFINITION DES SURFACES AXIALES
    # ============================================================================
    b_surf1 = Zch1 - (Ych1 * coef)
    b_surf2 = Zch2 - (Ych2 * coef)
    b_surf3 = Zch3 - (Ych3 * coef2)
    b_surf4 = Zch4 - (Ych4 * coef2)

    def Asurf1(x):
        return coef * x + b_surf1

    def Asurf2(x):
        return coef * x + b_surf2

    def Asurf3(x):
        return coef2 * x + b_surf3

    def Asurf4(x):
        return coef2 * x + b_surf4

    Y_asurf1 = Yfaille
    Y_asurf2 = Yfaille
    Y_asurf3 = Yfaille
    Y_asurf4 = Yfaille
    Z_asurf1 = Asurf1(Y_asurf1)
    Z_asurf2 = Asurf2(Y_asurf2)
    Z_asurf3 = Asurf3(Y_asurf3)
    Z_asurf4 = Asurf4(Y_asurf4)

    # ============================================================================
    # LIMITATION DE LA TAILLE DES SURFACES AXIALES
    # ============================================================================
    def filter_z(Z, Y, min_z=-2000, max_z=4000):
        indices = np.where((Z > min_z) & (Z < max_z))
        return Y[indices], Z[indices]

    Y_asurf1, Z_asurf1 = filter_z(Z_asurf1, Y_asurf1)
    Y_asurf2, Z_asurf2 = filter_z(Z_asurf2, Y_asurf2)
    Y_asurf3, Z_asurf3 = filter_z(Z_asurf3, Y_asurf3)
    Y_asurf4, Z_asurf4 = filter_z(Z_asurf4, Y_asurf4)

    # ============================================================================
    # DEFOMRATION DES STRATES (logique MATLAB reproduite)
    # ============================================================================
    Smax = 30
    n_tot = 1
    deltaS = Smax / n_tot

    # Utilisation des données topo et InSAR passées en paramètres
    Y_topo = params.get("Y_topo", np.linspace(0, 20000, 1000))
    Z_topo = params.get("Z_topo", np.zeros(1000) + 3307)
    Y_insar = params.get("Y_insar", np.linspace(0, 20000, 1000))
    Z_insar = params.get("Z_insar", np.zeros(1000))

    G_Y0 = np.linspace(0, 20000, di + 1)
    G_Z0 = np.zeros(di + 1) + 3307

    Y_save = np.zeros_like(G_Y0)
    Z_save = np.zeros_like(G_Z0)

    deltaZ_char2 = Rc * np.cos(teta)
    deltaY_char2 = Rc * np.sin(teta)

    S = deltaS
    for n in range(1, n_tot + 1):
        Y = G_Y0.copy()
        Z = G_Z0.copy()
        a_traj = np.tan(teta)
        b_traj = Z - (Y * a_traj)
        Ypoint = (b_traj - b_surf2) / (coef - a_traj)
        Zpoint = Asurf2(Ypoint)
        C_Y = Ypoint - (Rc * np.sin(teta))
        C_Z = Zpoint + (Rc * np.cos(teta))
        Zpoint2 = Zpoint + (Rc * np.cos(teta)) - (Rc * np.cos(beta))
        Ypoint2 = Ypoint - (Rc * np.sin(teta)) + (Rc * np.sin(beta))

        for l in range(di + 1):
            hyp3 = 0  # Initialisation par défaut

            # Zone 1: Z[l] < Asurf4(Y[l])
            if Z[l] < Asurf4(Y[l]):
                a_temp = np.tan(omega)
                b_temp = Z[l] - (a_temp * Y[l])
                Yp = (b_temp - b_surf4) / (coef2 - a_temp)
                Zp = (coef2 * Yp) + b_surf4
                hypo = (Yp - Y[l]) / np.cos(omega)

                if S < hypo:
                    Y[l] = Y[l] + (S * np.cos(omega))
                    Z[l] = Z[l] + (S * np.sin(omega))
                elif S > hypo and S < (hypo + (Rc2 * abs(teta - omega))):  # Utilisation de abs()
                    phi = (S - hypo) / Rc2
                    hypote = 2 * Rc2 * np.sin(phi / 2)
                    jela = (phi / 2) + omega
                    deltaZ = hypote * np.sin(jela)
                    deltaY = hypote * np.cos(jela)
                    Y[l] = Y[l] + deltaY + (hypo * np.cos(omega))
                    Z[l] = Z[l] + deltaZ + (hypo * np.sin(omega))
                elif S > (hypo + (Rc2 * abs(teta - omega))) and S < (hypo + (Rc2 * abs(teta - omega)) + hyp3):  # Utilisation de abs()
                    c = S - hypo - (Rc2 * abs(teta - omega))  # Utilisation de abs()
                    hypote = 2 * Rc2 * np.sin(abs(teta - omega) / 2)  # Utilisation de abs()
                    jela = (abs(teta - omega) / 2) + omega  # Utilisation de abs()
                    deltaZ = hypote * np.sin(jela)
                    deltaY = hypote * np.cos(jela)
                    Y[l] = Y[l] + deltaY + (hypo * np.cos(omega)) + (c * np.cos(teta))
                    Z[l] = Z[l] + deltaZ + (hypo * np.sin(omega)) + (c * np.sin(teta))
                elif S > (hypo + (Rc2 * abs(teta - omega)) + hyp3) and S < (hypo + (Rc2 * abs(teta - omega)) + hyp3 + (Rc * abs(beta - teta))):  # Utilisation de abs()
                    c = S - hypo - (Rc2 * abs(teta - omega)) - hyp3  # Utilisation de abs()
                    hypote = 2 * Rc2 * np.sin(abs(teta - omega) / 2)  # Utilisation de abs()
                    jela = (abs(teta - omega) / 2) + omega  # Utilisation de abs()
                    deltaZ1 = hypote * np.sin(jela)
                    deltaY1 = hypote * np.cos(jela)
                    hypote2 = 2 * Rc * np.sin(abs(c / (2 * Rc)))  # Utilisation de abs()
                    jela2 = (abs(c / (2 * Rc))) + teta  # Utilisation de abs()
                    deltaZ2 = hypote2 * np.sin(jela2)
                    deltaY2 = hypote2 * np.cos(jela2)
                    Y[l] = Y[l] + (hypo * np.cos(omega)) + deltaY1 + (hyp3 * np.cos(teta)) + deltaY2
                    Z[l] = Z[l] + (hypo * np.sin(omega)) + deltaZ1 + (hyp3 * np.sin(teta)) + deltaZ2
                elif S > (hypo + (Rc2 * abs(teta - omega)) + hyp3 + (Rc * abs(beta - teta))):  # Utilisation de abs()
                    c = S - hypo - (Rc2 * abs(teta - omega)) - hyp3 - (Rc * abs(beta - teta))  # Utilisation de abs()
                    hypote = 2 * Rc2 * np.sin(abs(teta - omega) / 2)  # Utilisation de abs()
                    jela = (abs(teta - omega) / 2) + omega  # Utilisation de abs()
                    deltaZ1 = hypote * np.sin(jela)
                    deltaY1 = hypote * np.cos(jela)
                    hypote2 = 2 * Rc * np.sin(abs(beta - teta) / 2)  # Utilisation de abs()
                    jela2 = (abs(beta - teta) / 2) + teta  # Utilisation de abs()
                    deltaZ2 = hypote2 * np.sin(jela2)
                    deltaY2 = hypote2 * np.cos(jela2)
                    Y[l] = Y[l] + (hypo * np.cos(omega)) + deltaY1 + (hyp3 * np.cos(teta)) + deltaY2 + (c * np.cos(beta))
                    Z[l] = Z[l] + (hypo * np.sin(omega)) + deltaZ1 + (hyp3 * np.sin(teta)) + deltaZ2 + (c * np.sin(beta))

            # ZONE DANS LA PREMIERE CHARNIERE
            elif Z[l] > Asurf4(Y[l]) and Z[l] < Asurf3(Y[l]):
                Zc = Z[l] - Rc2 * np.sin(angle)
                Yc = Y[l] - Rc2 * np.cos(angle)
                test2 = Yc * abis2 + bbis2
                indice = np.argmin(np.abs(Zc - test2))
                sol_Y = Yc[indice]
                sol_Z = Zc[indice]
                dey1 = np.abs(sol_Y - Y[l])
                dez1 = np.abs(sol_Z - Z[l])
                Rc2test2 = np.sqrt(dez1**2 + dey1**2)
                phi3 = np.arctan2(dey1, dez1)
                if Y[l] < sol_Y:
                    dist = Rc2 * (abs(teta) + phi3)  # Utilisation de abs()
                else:
                    dist = Rc2 * (abs(teta) - phi3)  # Utilisation de abs()

                deltaZ1 = Rc2 * np.cos(teta)
                deltaY1 = Rc2 * np.sin(teta)
                if Y[l] < sol_Y:
                    Yinta = Y[l] + dey1 + deltaY1
                    Zinta = Z[l] - deltaZ1 + dez1
                else:
                    Yinta = Y[l] - dey1 + deltaY1
                    Zinta = Z[l] - deltaZ1 + dez1

                b_ramp_int = Zinta - aramp2 * Yinta
                Yinta2 = (b_ramp_int - b_surf2) / (coef - aramp2)
                Zinta2 = aramp2 * Yinta2 + b_ramp_int
                hyp3 = np.sqrt((Zinta2 - Zinta)**2 + (Yinta2 - Yinta)**2)  # Calcul de hyp3
                dist_int = dist + hyp3
                dist_tot = dist + hyp3 + Rc * abs(beta - teta)  # Utilisation de abs()

                if S < dist:
                    if omega < 0:
                        phi = S / Rc2
                        rota = np.arccos((Y[l] - sol_Y) / Rc2)
                        Y[l] = sol_Y + Rc2 * np.cos(-rota + phi)
                        Z[l] = sol_Z + Rc2 * np.sin(-rota + phi)
                    else:
                        phi1 = S / Rc2
                        deltaZ1 = Rc2 * np.cos(phi1 + phi3)
                        deltaY1 = Rc2 * np.sin(phi1 + phi3)
                        Y[l] = Y[l] - dey1 + deltaY1
                        Z[l] = Z[l] - deltaZ1 + dez1
                elif S > dist and S < dist_int:
                    Y[l] = Yinta + ((S - dist) * np.cos(teta))
                    Z[l] = Zinta + ((S - dist) * np.sin(teta))
                elif S > dist and S < dist_tot:
                    c = S - dist - hyp3
                    phi4 = c / Rc
                    deltaZ = Rc * np.cos(teta)
                    deltaY = Rc * np.sin(teta)
                    deltaZ1 = Rc * np.cos(teta + phi4)
                    deltaY1 = Rc * np.sin(teta + phi4)
                    Y[l] = Yinta2 - deltaY + deltaY1
                    Z[l] = Zinta2 + deltaZ - deltaZ1
                elif S > dist_tot:
                    c = S - dist - hyp3 - Rc * abs(beta - teta)  # Utilisation de abs()
                    deltaZ = Rc * np.cos(teta)
                    deltaY = Rc * np.sin(teta)
                    deltaZ1 = Rc * np.cos(beta)
                    deltaY1 = Rc * np.sin(beta)
                    Zinta3 = Zinta2 + deltaZ - deltaZ1
                    Yinta3 = Yinta2 - deltaY + deltaY1
                    Y[l] = Yinta3 + c * np.cos(beta)
                    Z[l] = Zinta3 + c * np.sin(beta)

            # ZONE ENTRE LES DEUX CHARNIERES
            elif Z[l] > Asurf3(Y[l]) and Z[l] < Asurf2(Y[l]):
                Hyp = np.sqrt(((Ypoint[l] - Y[l])**2) + ((Zpoint[l] - Z[l])**2))
                if S < Hyp:
                    Y[l] = Y[l] + (S * np.cos(teta))
                    Z[l] = Z[l] + (S * np.sin(teta))
                elif S > Hyp and S < (Rc * abs(beta - teta) + Hyp):  # Utilisation de abs()
                    phi = (S - Hyp) / Rc
                    deltaZ1 = np.abs(Rc * np.cos(teta + phi))
                    deltaY1 = np.abs(Rc * np.sin(teta + phi))
                    Y[l] = Ypoint[l] - deltaY_char2 + deltaY1
                    Z[l] = Zpoint[l] - deltaZ1 + deltaZ_char2
                elif S > (Rc * abs(beta - teta) + Hyp):  # Utilisation de abs()
                    c = S - (Rc * abs(beta - teta) + Hyp)  # Utilisation de abs()
                    deltaZ = Rc * np.cos(teta + beta)
                    deltaY = Rc * np.sin(teta + beta)
                    Y[l] = Ypoint2[l] + (c * np.cos(beta))
                    Z[l] = Zpoint2[l] + (c * np.sin(beta))

            # ZONE DANS LA PREMIERE CHARNIERE
            elif Z[l] > Asurf2(Y[l]) and Z[l] < Asurf1(Y[l]):
                Zc = Z[l] - Rc * np.sin(angle)
                Yc = Y[l] - Rc * np.cos(angle)
                test = Yc * abis + bbis
                indice = np.argmin(np.abs(Zc - test))
                sol_Y = Yc[indice]
                sol_Z = Zc[indice]
                Cint_Y = sol_Y
                Cint_Z = sol_Z
                dez = np.abs(Cint_Z - Z[l])
                dey = np.abs(Cint_Y - Y[l])
                Rctest = np.sqrt(dez**2 + dey**2)
                phi3 = np.arctan2(dey, dez)
                hypo = Rc * (abs(beta) - phi3)  # Utilisation de abs()

                if S < hypo:
                    phi1 = S / Rc
                    deltaZ1 = Rc * np.cos(phi1 + phi3)
                    deltaY1 = Rc * np.sin(phi1 + phi3)
                    Y[l] = Y[l] - dey + deltaY1
                    Z[l] = Z[l] - deltaZ1 + dez
                elif S > hypo:
                    phi2 = np.arccos(dez / Rc)
                    deltaZ1 = Rc * np.cos(beta)
                    deltaY1 = Rc * np.sin(beta)
                    Yint = Y[l] - dey + deltaY1
                    Zint = Z[l] - deltaZ1 + dez
                    Y[l] = Yint + ((S - hypo) * np.cos(beta))
                    Z[l] = Zint + ((S - hypo) * np.sin(beta))

            # ZONE APRES LA PREMIERE CHARNIERE
            elif Z[l] > Asurf1(Y[l]):
                Y[l] = Y[l] + (S * np.cos(beta))
                Z[l] = Z[l] + (S * np.sin(beta))

        if n == 1:
            Y_save = Y.copy()
            Z_save = Z.copy()
        S += deltaS

    # ============================================================================
    # Retourne uniquement les données à plotter
    # ============================================================================
    return {
        "Yfaille": Yfaille,
        "Zfaille": Zfaille,
        "Y_asurf1": Y_asurf1,
        "Z_asurf1": Z_asurf1,
        "Y_asurf2": Y_asurf2,
        "Z_asurf2": Z_asurf2,
        "Y_asurf3": Y_asurf3,
        "Z_asurf3": Z_asurf3,
        "Y_asurf4": Y_asurf4,
        "Z_asurf4": Z_asurf4,
        "Y_save": Y_save,
        "Z_save": Z_save,
        "Y_topo": Y_topo,
        "Z_topo": Z_topo,
        "Y_insar": Y_insar,
        "Z_insar": Z_insar,
    }
