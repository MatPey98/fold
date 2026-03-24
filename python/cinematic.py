import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def compute_fault_and_axial_surfaces(params):
    # ============================================================================
    # PARAMETRES SETTING (identique au MATLAB)
    # ============================================================================
    beta = np.deg2rad(params.get("beta",60))
    teta = np.deg2rad(params.get("teta", 50))
    omega = np.deg2rad(params.get("omega",30))
    # sigma = np.deg2rad(-3.2)

    alpha = teta + ((beta - teta) / 2)
    coef = -1 / alpha

    alpha2 = omega + ((teta - omega) / 2)
    coef2 = -1 / alpha2

    # CoordonnÃ©es de la faille en surface
    Y_faille = params.get("Y_faille", 18400)  # en m
    Z_faille = params.get("Z_faille", 3400)  # en m

    # DÃ©finition de la premiÃ¨re rampe
    aramp1 = np.tan(beta)
    bramp1 = Z_faille - (aramp1 * Y_faille)

    def ramp1(x):
        return aramp1 * x + bramp1

    # Point d'inflexion vers la 2Ã¨me rampe
    Y_r2 = params.get("Y_r2", 20000)
    Z_r2 = ramp1(Y_r2)

    # DÃ©finition de la 2Ã¨me rampe
    aramp2 = np.tan(teta)
    bramp2 = Z_r2 - (aramp2 * Y_r2)

    def ramp2(x):
        return aramp2 * x + bramp2

    # Point d'inflexion vers la 3Ã¨me rampe
    Y_r3 = params.get("Y_r3", 10000)
    Z_r3 = ramp2(Y_r3)

    # DÃ©finition de la 3Ã¨me rampe
    aramp3 = np.tan(omega)
    bramp3 = Z_r3 - (aramp3 * Y_r3)

    def ramp3(x):
        return aramp3 * x + bramp3

    # Largeur des charniÃ¨res
    W = params.get("W", 7000)
    W2 = params.get("W2", 3000)

    # Ã‰tendue de la zone d'Ã©tude
    Ymin = params.get("Ymin", -3000)
    Ymax = params.get("Ymax", 50000)

    # Nombre de strates Ã  dÃ©former
    n_strata = params.get("n_strata", 20)
    Zhaut = params.get("Zhaut", 1800)

    # Nombre de points par strate au dÃ©part
    Ymax2 = params.get("Ymax2", 14000)
    di = params.get("di", 2000)
    dj = int((Ymax2 - Ymin - 1) / di)

    # Nombre d'itÃ©rations de raccourcissement
    ite_s = 1

    debut = np.pi
    fin = 2 * np.pi
    angle = np.linspace(debut, fin, 10000)

    # ============================================================================
    # DEFINITION DES CHARNIERES
    # ============================================================================
    def inter_ramp(x):
        return ramp1(x) - ramp2(x)

    Ycrois = fsolve(inter_ramp, 4593000)[0]
    Zcrois = ramp1(Ycrois)

    # Calcul des rayons de courbure
    Rc = (W / 2) / np.sin((beta - teta) / 2)
    Rc2 = (W2 / 2) / np.sin((teta - omega) / 2)

    # Points de sortie de charniÃ¨re sur la rampe au nord
    hypo = (W / 2) / np.cos((beta - teta) / 2)
    Ych1 = Y_r2 + hypo * np.cos(beta)
    Zch1 = Z_r2 + hypo * np.sin(beta)
    Ych2 = Y_r2 - hypo * np.cos(teta)
    Zch2 = Z_r2 - hypo * np.sin(teta)

    hypo2 = (W2 / 2) / np.cos((teta - omega) / 2)
    Ych3 = Y_r3 + hypo2 * np.cos(teta)
    Zch3 = Z_r3 + hypo2 * np.sin(teta)
    Ych4 = Y_r3 - hypo2 * np.cos(omega)
    Zch4 = Z_r3 - hypo2 * np.sin(omega)

    # Centre des cercles
    Ycentre = Ych1 - Rc * np.sin(beta)
    Zcentre = Zch1 + Rc * np.cos(beta)
    Ycentre2 = Ych3 - Rc2 * np.sin(teta)
    Zcentre2 = Zch3 + Rc2 * np.cos(teta)

    # Droite oÃ¹ se trouveront l'ensemble des centres de rotation
    abis = coef
    bbis = Zcentre - (abis * Ycentre)

    def biss(x):
        return abis * x + bbis

    abis2 = coef2
    bbis2 = Zcentre2 - (abis2 * Ycentre2)

    def biss2(x):
        return abis2 * x + bbis2

    # Ã‰quation des cercles
    def Cercle(x):
        return Zcentre - np.sqrt(np.maximum(0, -(x - Ycentre)**2 + Rc**2))

    def Cercle2(x):
        return Zcentre2 - np.sqrt(np.maximum(0, -(x - Ycentre2)**2 + Rc2**2))

    # DÃ©finition des coordonnÃ©es Y de la faille
    Yfaille = np.arange(Ymin, Ymax + 1, 1)
    taille = len(Yfaille)
    Zfaille = np.zeros(taille)

    # Calcul des coordonnÃ©es Z de la faille
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

    # Limitation de la taille des surfaces axiales pour reprÃ©sentation graphique
    def filter_z(Z, Y, min_z=-2000, max_z=4000):
        indices = np.where((Z > min_z) & (Z < max_z))
        return Y[indices], Z[indices]

    Y_asurf1, Z_asurf1 = filter_z(Z_asurf1, Y_asurf1)
    Y_asurf2, Z_asurf2 = filter_z(Z_asurf2, Y_asurf2)
    Y_asurf3, Z_asurf3 = filter_z(Z_asurf3, Y_asurf3)
    Y_asurf4, Z_asurf4 = filter_z(Z_asurf4, Y_asurf4)

    # ============================================================================
    # LECTURE DES DONNEES TOPO ET INSAR
    # ============================================================================
    Y_topo = params.get("Y-topo", np.zeros(1000) + 3307)
    Z_topo = params.get("Z_topo", np.linspace(0, 20000, 1000))
    Y_insar = params.get("Y_insar", np.linspace(0, 20000, 1000))
    Z_insar = params.get("Z_insar", np.zeros(1000))

    # Y_insar = np.max(Y_insar) - Y_insar
    # Y_topo = np.max(Y_topo) - Y_topo

    # ============================================================================
    # DEFOMRATION DES STRATES
    # ============================================================================
    Smax = params.get("Smax", 30)
    n_tot = params.get("n_tot", 1)
    deltaS = Smax / n_tot

    G_Y0 = np.linspace(0, 30000, di + 1)
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
            # Zone 1: Z(l) < Asurf4(Y(l))
            if Z[l] < Asurf4(Y[l]):
                a_temp = np.tan(omega)
                b_temp = Z[l] - (a_temp * Y[l])
                Yp = (b_temp - b_surf4) / (coef2 - a_temp)
                Zp = (coef2 * Yp) + b_surf4
                hypo = (Yp - Y[l]) / np.cos(omega)

                if S < hypo:
                    Y[l] = Y[l] + (S * np.cos(omega))
                    Z[l] = Z[l] + (S * np.sin(omega))
                elif S > hypo and S < (hypo + (Rc2 * (teta - omega))):
                    phi = (S - hypo) / Rc2
                    hypote = 2 * Rc2 * np.sin(phi / 2)
                    jela = (phi / 2) + omega
                    deltaZ = hypote * np.sin(jela)
                    deltaY = hypote * np.cos(jela)
                    Y[l] = Y[l] + deltaY + (hypo * np.cos(omega))
                    Z[l] = Z[l] + deltaZ + (hypo * np.sin(omega))
                elif S > (hypo + (Rc2 * (teta - omega))) and S < (hypo + (Rc2 * (teta - omega)) + hyp3):
                    c = S - hypo - (Rc2 * (teta - omega))
                    hypote = 2 * Rc2 * np.sin((teta - omega) / 2)
                    jela = ((teta - omega) / 2) + omega
                    deltaZ = hypote * np.sin(jela)
                    deltaY = hypote * np.cos(jela)
                    Y[l] = Y[l] + deltaY + (hypo * np.cos(omega)) + (c * np.cos(teta))
                    Z[l] = Z[l] + deltaZ + (hypo * np.sin(omega)) + (c * np.sin(teta))
                elif S > (hypo + (Rc2 * (teta - omega)) + hyp3) and S < (hypo + (Rc2 * (teta - omega)) + hyp3 + (Rc * (beta - teta))):
                    c = S - hypo - (Rc2 * (teta - omega)) - hyp3
                    hypote = 2 * Rc2 * np.sin((teta - omega) / 2)
                    jela = ((teta - omega) / 2) + omega
                    deltaZ1 = hypote * np.sin(jela)
                    deltaY1 = hypote * np.cos(jela)
                    hypote2 = 2 * Rc * np.sin(c / (2 * Rc))
                    jela2 = (c / (2 * Rc)) + teta
                    deltaZ2 = hypote2 * np.sin(jela2)
                    deltaY2 = hypote2 * np.cos(jela2)
                    Y[l] = Y[l] + (hypo * np.cos(omega)) + deltaY1 + (hyp3 * np.cos(teta)) + deltaY2
                    Z[l] = Z[l] + (hypo * np.sin(omega)) + deltaZ1 + (hyp3 * np.sin(teta)) + deltaZ2
                elif S > (hypo + (Rc2 * (teta - omega)) + hyp3 + (Rc * (beta - teta))):
                    c = S - hypo - (Rc2 * (teta - omega)) - hyp3 - (Rc * (beta - teta))
                    hypote = 2 * Rc2 * np.sin((teta - omega) / 2)
                    jela = ((teta - omega) / 2) + omega
                    deltaZ1 = hypote * np.sin(jela)
                    deltaY1 = hypote * np.cos(jela)
                    hypote2 = 2 * Rc * np.sin((beta - teta) / 2)
                    jela2 = ((beta - teta) / 2) + teta
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
                    dist = Rc2 * (teta + phi3)
                else:
                    dist = Rc2 * (teta - phi3)

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
                hyp3 = np.sqrt((Zinta2 - Zinta)**2 + (Yinta2 - Yinta)**2)
                dist_int = dist + hyp3
                dist_tot = dist + hyp3 + Rc * (beta - teta)

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
                    c = S - dist - hyp3 - Rc * (beta - teta)
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
                elif S > Hyp and S < (Rc * (beta - teta) + Hyp):
                    phi = (S - Hyp) / Rc
                    deltaZ1 = np.abs(Rc * np.cos(teta + phi))
                    deltaY1 = np.abs(Rc * np.sin(teta + phi))
                    Y[l] = Ypoint[l] - deltaY_char2 + deltaY1
                    Z[l] = Zpoint[l] - deltaZ1 + deltaZ_char2
                elif S > (Rc * (beta - teta) + Hyp):
                    c = S - (Rc * (beta - teta) + Hyp)
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
                hypo = Rc * (beta - phi3)

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
        "n_tot": n_tot,
        "Ych1": Ych1,  # Point d'intersection 1
        "Zch1": Zch1,
        "Ych2": Ych2,  # Point d'intersection 2
        "Zch2": Zch2,
        "Ych3": Ych3,  # Point d'intersection 3
        "Zch3": Zch3,
        "Ych4": Ych4,  # Point d'intersection 4
        "Zch4": Zch4,
        "Ycrois": Ycrois,  # Point de croisement entre ramp1 et ramp2
        "Zcrois": Zcrois,
        "Y_r2": Y_r2,
        "Z_r2": Z_r2,
        "Y_r3": Y_r3,
        "Z_r3": Z_r3,
    }

if __name__ == "__main__":
    params = {
  # 1ere rampe
    "beta": 65,  
    "Y_faille": 18440,
    "Z_faille": 3400,

  # 2ème rampe
    "teta": 55,
    "Y_r2": 17000,

  # 3ème rampe
    "omega": 30,  # Troisième segment presque horizontal
    "Y_r3": 6500,

  # ZOne d'étude
    "Ymin": -3000,  # Début du calcul de la faille
    "Ymax": 18440,

  # Largeur des charnières
    "W": 7000,     # Largeur de la première charnière
    "W2": 3000,    # Largeur de la deuxième charnière
    
    "Zhaut": 1800, 
    "n_strata": 20, # Nombre de strates
    "di": 2000, # Pas de discrétisation
    "ite_s": 1, # Nombre d'itérations

    # "Y_insar": abscisses_insar_verti,  # Vos données InSAR
    # "Z_insar": velocities_verti,  # Vos données InSAR
    "Ymax2":14000,
    "Smax": 5, # raccourcissement
    "n_tot": 1
    }

    results = compute_fault_and_axial_surfaces(params)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Graphique supÃ©rieur : DÃ©formation et Topo
    ax1.plot(results["Y_topo"], results["Z_topo"], '-k', label='Topo')  # Ã‰lÃ©vation Ã  gauche
    ax1.set_ylabel('élévation (m)', color='k')
    ax1.set_ylim([3200, 4400])
    ax1.tick_params(axis='y', labelcolor='k')
    ax1.grid(True)

    ax1b = ax1.twinx()  # CrÃ©er un deuxiÃ¨me axe Y
    ax1b.plot(results["Y_insar"], results["Z_insar"], '-r', label='InSAR')  # DÃ©formation mesurÃ©e Ã  droite
    ax1b.plot(results["Y_save"], results["Z_save"] - 3307, '-b', label='Déformation calculée')  # DÃ©formation calculÃ©e en bleu
    ax1b.set_ylabel('Déformation (m)', color='r')
    ax1b.set_ylim([-5, 20])
    ax1b.tick_params(axis='y', labelcolor='r')
    ax1b.legend(loc='upper right')

    # Graphique infÃ©rieur : Profil gÃ©nÃ©ral
    ax2.plot(results["Y_asurf1"], results["Z_asurf1"], '-k', label='Surface axiale 1')
    ax2.plot(results["Y_asurf2"], results["Z_asurf2"], '-k', label='Surface axiale 2')
    ax2.plot(results["Y_asurf3"], results["Z_asurf3"], '-k', label='Surface axiale 3')
    ax2.plot(results["Y_asurf4"], results["Z_asurf4"], '-k', label='Surface axiale 4')
    ax2.plot(results["Yfaille"], results["Zfaille"], '-b', label='Faille')
    ax2.plot(results["Y_topo"], results["Z_topo"], '-k', label='Topo')
    ax2.plot(results["Yfaille"], results["Zfaille"], 'og', label='Faille Surface')
    ax2.plot(results["Y_r2"], results["Z_r2"], 'or', label='Point R2')
    ax2.plot(results["Y_r3"], results["Z_r3"], 'or', label='Point R3')
    ax2.plot(results["Ych1"], results["Zch1"], 'ok', label='Charnière 1')
    ax2.plot(results["Ych2"], results["Zch2"], 'ok', label='Charnière 2')
    ax2.plot(results["Ych3"], results["Zch3"], 'ok', label='Charnière 3')
    ax2.plot(results["Ych4"], results["Zch4"], 'ok', label='Charnière 4')
    ax2.set_xlabel('Distance horizontale (m)')
    ax2.set_ylabel('Profondeur (m)')
    ax2.axis('equal')
    ax2.grid(True)
    ax2.legend()
    ax2.set_xlim([0, 35000])
    ax2.set_ylim([-16000, 5000])

    plt.tight_layout()
    plt.show()