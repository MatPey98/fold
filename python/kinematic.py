# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def compute_fault_and_axial_surfaces(params, Y_topo, Z_topo, Y_insar, Z_insar):
    """
    Compute fault geometry, axial surfaces and surface deformation
    for a trishear/kink-band kinematic model.

    Parameters
    ----------
    params : dict
        Model parameters (see input file).
    Y_topo, Z_topo : array-like
        Along-profile topography coordinates.
    Y_insar, Z_insar : array-like
        Along-profile InSAR velocities.

    Returns
    -------
    dict with fault geometry, axial surfaces, deformed strata, and
    projected topo/InSAR.
    """
    # ============================================================================
    # PARAMETERS
    # ============================================================================
    beta  = np.deg2rad(params["beta"])
    teta  = np.deg2rad(params["teta"])
    omega = np.deg2rad(params["omega"])

    # Bisector angles for axial surfaces
    alpha  = teta  + (beta  - teta)  / 2
    alpha2 = omega + (teta  - omega) / 2
    coef   = -1 / alpha
    coef2  = -1 / alpha2

    # Fault surface point
    Y_faille = params["Y_faille"]
    Z_faille = params["Z_faille"]

    # Ramp equations  (y = a*x + b)
    aramp1 = np.tan(beta);  bramp1 = Z_faille - aramp1 * Y_faille
    ramp1  = lambda x: aramp1 * x + bramp1

    Y_r2   = params["Y_r2"];  Z_r2 = ramp1(Y_r2)
    aramp2 = np.tan(teta);  bramp2 = Z_r2 - aramp2 * Y_r2
    ramp2  = lambda x: aramp2 * x + bramp2

    Y_r3   = params["Y_r3"];  Z_r3 = ramp2(Y_r3)
    aramp3 = np.tan(omega); bramp3 = Z_r3 - aramp3 * Y_r3
    ramp3  = lambda x: aramp3 * x + bramp3

    # Hinge widths and study extent
    W    = params["W"]
    W2   = params["W2"]
    Ymin = params["Ymin"]
    Ymax = params["Ymax"]

    # Shortening
    Smax  = params["Smax"]
    n_tot = params.get("n_tot", 1)
    di    = params.get("di", 4000)
    deltaS = Smax / n_tot

    # ============================================================================
    # HINGE GEOMETRY
    # ============================================================================
    # Ramp intersection (analytical)
    Ycrois = (bramp2 - bramp1) / (aramp1 - aramp2)
    Zcrois = ramp1(Ycrois)

    # Curvature radii
    Rc  = (W  / 2) / np.sin((beta - teta) / 2)
    Rc2 = (W2 / 2) / np.sin((teta - omega) / 2)

    # Hinge exit points — ramp 1/2 transition
    hypo  = (W  / 2) / np.cos((beta - teta) / 2)
    Ych1  =  Y_r2 + hypo * np.cos(beta);   Zch1 = Z_r2 + hypo * np.sin(beta)
    Ych2  =  Y_r2 - hypo * np.cos(teta);   Zch2 = Z_r2 - hypo * np.sin(teta)

    # Hinge exit points — ramp 2/3 transition
    hypo2 = (W2 / 2) / np.cos((teta - omega) / 2)
    Ych3  =  Y_r3 + hypo2 * np.cos(teta);  Zch3 = Z_r3 + hypo2 * np.sin(teta)
    Ych4  =  Y_r3 - hypo2 * np.cos(omega); Zch4 = Z_r3 - hypo2 * np.sin(omega)

    # Circle centres
    Ycentre  = Ych1 - Rc  * np.sin(beta);  Zcentre  = Zch1 + Rc  * np.cos(beta)
    Ycentre2 = Ych3 - Rc2 * np.sin(teta);  Zcentre2 = Zch3 + Rc2 * np.cos(teta)

    # Rotation-centre bisector lines (coefficients only — no function needed)
    abis  = coef;  bbis  = Zcentre  - abis  * Ycentre
    abis2 = coef2; bbis2 = Zcentre2 - abis2 * Ycentre2

    # Circle equations
    Cercle  = lambda x: Zcentre  - np.sqrt(np.maximum(0, Rc **2 - (x - Ycentre) **2))
    Cercle2 = lambda x: Zcentre2 - np.sqrt(np.maximum(0, Rc2**2 - (x - Ycentre2)**2))

    # ============================================================================
    # FAULT TRACE  (vectorized)
    # ============================================================================
    Yfaille = np.arange(Ymin, Ymax + 1, dtype=float)
    Zfaille = np.where(
        Yfaille < Ych4, ramp3(Yfaille),
        np.where((Yfaille < Ych3), Cercle2(Yfaille),
        np.where((Yfaille < Ych2), ramp2(Yfaille),
        np.where((Yfaille < Ych1), Cercle(Yfaille),
                                   ramp1(Yfaille)))))

    # ============================================================================
    # AXIAL SURFACES
    # ============================================================================
    b_surf1 = Zch1 - Ych1 * coef
    b_surf2 = Zch2 - Ych2 * coef
    b_surf3 = Zch3 - Ych3 * coef2
    b_surf4 = Zch4 - Ych4 * coef2

    Asurf1 = lambda x: coef  * x + b_surf1   # Northern axial surface
    Asurf2 = lambda x: coef  * x + b_surf2   # Northern inner axial surface
    Asurf3 = lambda x: coef2 * x + b_surf3   # Southern inner axial surface
    Asurf4 = lambda x: coef2 * x + b_surf4   # Southern axial surface

    def _clip_surface(Z, Y, zmin=-2000, zmax=4000):
        idx = (Z > zmin) & (Z < zmax)
        return Y[idx], Z[idx]

    Y_asurf1, Z_asurf1 = _clip_surface(Asurf1(Yfaille), Yfaille)
    Y_asurf2, Z_asurf2 = _clip_surface(Asurf2(Yfaille), Yfaille)
    Y_asurf3, Z_asurf3 = _clip_surface(Asurf3(Yfaille), Yfaille)
    Y_asurf4, Z_asurf4 = _clip_surface(Asurf4(Yfaille), Yfaille)

    # ============================================================================
    # PROJECT TOPO AND INSAR
    # ============================================================================
    Y_insar = np.max(Y_insar) - Y_insar
    Y_topo  = np.max(Y_topo)  - Y_topo

    # ============================================================================
    # STRATA DEFORMATION
    # ============================================================================
    angle = np.linspace(np.pi, 2 * np.pi, 10000)

    G_Y0      = np.linspace(0, 30000, di + 1)
    G_Z0      = np.full(di + 1, 3307.0)
    Y_initial = G_Y0.copy()

    deltaZ_char2 = Rc * np.cos(teta)
    deltaY_char2 = Rc * np.sin(teta)

    hyp3 = 0.0
    S = deltaS

    for n in range(1, n_tot + 1):
        Y = G_Y0.copy()
        Z = G_Z0.copy()

        a_traj  = np.tan(teta)
        b_traj  = Z - Y * a_traj
        Ypoint  = (b_traj - b_surf2) / (coef - a_traj)
        Zpoint  = Asurf2(Ypoint)
        Ypoint2 = Ypoint - Rc * np.sin(teta) + Rc * np.sin(beta)
        Zpoint2 = Zpoint + Rc * np.cos(teta) - Rc * np.cos(beta)

        for l in range(di + 1):

            # ── Zone 1: below southern axial surface ──────────────────────────
            if Z[l] < Asurf4(Y[l]):
                a_temp = np.tan(omega)
                b_temp = Z[l] - a_temp * Y[l]
                Yp   = (b_temp - b_surf4) / (coef2 - a_temp)
                hypo = (Yp - Y[l]) / np.cos(omega)

                if S < hypo:
                    Y[l] += S * np.cos(omega)
                    Z[l] += S * np.sin(omega)
                elif S < hypo + Rc2 * (teta - omega):
                    phi    = (S - hypo) / Rc2
                    hypote = 2 * Rc2 * np.sin(phi / 2)
                    jela   = phi / 2 + omega
                    Y[l] += hypote * np.cos(jela) + hypo * np.cos(omega)
                    Z[l] += hypote * np.sin(jela) + hypo * np.sin(omega)
                elif S < hypo + Rc2 * (teta - omega) + hyp3:
                    c      = S - hypo - Rc2 * (teta - omega)
                    hypote = 2 * Rc2 * np.sin((teta - omega) / 2)
                    jela   = (teta - omega) / 2 + omega
                    Y[l] += hypote * np.cos(jela) + hypo * np.cos(omega) + c * np.cos(teta)
                    Z[l] += hypote * np.sin(jela) + hypo * np.sin(omega) + c * np.sin(teta)
                elif S < hypo + Rc2 * (teta - omega) + hyp3 + Rc * (beta - teta):
                    c       = S - hypo - Rc2 * (teta - omega) - hyp3
                    hypote  = 2 * Rc2 * np.sin((teta - omega) / 2)
                    jela    = (teta - omega) / 2 + omega
                    hypote2 = 2 * Rc * np.sin(c / (2 * Rc))
                    jela2   = c / (2 * Rc) + teta
                    Y[l] += hypo*np.cos(omega) + hypote*np.cos(jela) + hyp3*np.cos(teta) + hypote2*np.cos(jela2)
                    Z[l] += hypo*np.sin(omega) + hypote*np.sin(jela) + hyp3*np.sin(teta) + hypote2*np.sin(jela2)
                else:
                    c       = S - hypo - Rc2 * (teta - omega) - hyp3 - Rc * (beta - teta)
                    hypote  = 2 * Rc2 * np.sin((teta - omega) / 2)
                    jela    = (teta - omega) / 2 + omega
                    hypote2 = 2 * Rc * np.sin((beta - teta) / 2)
                    jela2   = (beta - teta) / 2 + teta
                    Y[l] += hypo*np.cos(omega) + hypote*np.cos(jela) + hyp3*np.cos(teta) + hypote2*np.cos(jela2) + c*np.cos(beta)
                    Z[l] += hypo*np.sin(omega) + hypote*np.sin(jela) + hyp3*np.sin(teta) + hypote2*np.sin(jela2) + c*np.sin(beta)

            # ── Zone 2: in southern hinge ─────────────────────────────────────
            elif Z[l] < Asurf3(Y[l]):
                Zc   = Z[l] - Rc2 * np.sin(angle)
                Yc   = Y[l] - Rc2 * np.cos(angle)
                idx  = np.argmin(np.abs(Zc - (Yc * abis2 + bbis2)))
                sol_Y, sol_Z = Yc[idx], Zc[idx]
                dey1 = np.abs(sol_Y - Y[l])
                dez1 = np.abs(sol_Z - Z[l])
                phi3 = np.arctan2(dey1, dez1)
                dist = Rc2 * (teta + phi3) if Y[l] < sol_Y else Rc2 * (teta - phi3)

                dZ1 = Rc2 * np.cos(teta); dY1 = Rc2 * np.sin(teta)
                if Y[l] < sol_Y:
                    Yinta = Y[l] + dey1 + dY1;  Zinta = Z[l] - dZ1 + dez1
                else:
                    Yinta = Y[l] - dey1 + dY1;  Zinta = Z[l] - dZ1 + dez1

                b_ramp_int = Zinta - aramp2 * Yinta
                Yinta2 = (b_ramp_int - b_surf2) / (coef - aramp2)
                Zinta2 = aramp2 * Yinta2 + b_ramp_int
                hyp3   = np.sqrt((Zinta2 - Zinta)**2 + (Yinta2 - Yinta)**2)
                dist_int = dist + hyp3
                dist_tot = dist + hyp3 + Rc * (beta - teta)

                if S < dist:
                    if omega < 0:
                        phi  = S / Rc2
                        rota = np.arccos((Y[l] - sol_Y) / Rc2)
                        Y[l] = sol_Y + Rc2 * np.cos(-rota + phi)
                        Z[l] = sol_Z + Rc2 * np.sin(-rota + phi)
                    else:
                        phi1 = S / Rc2
                        Y[l] = Y[l] - dey1 + Rc2 * np.sin(phi1 + phi3)
                        Z[l] = Z[l] - Rc2 * np.cos(phi1 + phi3) + dez1
                elif S < dist_int:
                    Y[l] = Yinta + (S - dist) * np.cos(teta)
                    Z[l] = Zinta + (S - dist) * np.sin(teta)
                elif S < dist_tot:
                    c    = S - dist - hyp3
                    phi4 = c / Rc
                    Y[l] = Yinta2 - Rc*np.sin(teta) + Rc*np.sin(teta + phi4)
                    Z[l] = Zinta2 + Rc*np.cos(teta) - Rc*np.cos(teta + phi4)
                else:
                    c    = S - dist - hyp3 - Rc * (beta - teta)
                    Y[l] = Yinta2 - Rc*np.sin(teta) + Rc*np.sin(beta) + c*np.cos(beta)
                    Z[l] = Zinta2 + Rc*np.cos(teta) - Rc*np.cos(beta) + c*np.sin(beta)

            # ── Zone 3: between the two hinges ───────────────────────────────
            elif Z[l] < Asurf2(Y[l]):
                Hyp = np.hypot(Ypoint[l] - Y[l], Zpoint[l] - Z[l])
                if S < Hyp:
                    Y[l] += S * np.cos(teta)
                    Z[l] += S * np.sin(teta)
                elif S < Hyp + Rc * (beta - teta):
                    phi  = (S - Hyp) / Rc
                    Y[l] = Ypoint[l] - deltaY_char2 + np.abs(Rc * np.sin(teta + phi))
                    Z[l] = Zpoint[l] - np.abs(Rc * np.cos(teta + phi)) + deltaZ_char2
                else:
                    c    = S - Hyp - Rc * (beta - teta)
                    Y[l] = Ypoint2[l] + c * np.cos(beta)
                    Z[l] = Zpoint2[l] + c * np.sin(beta)

            # ── Zone 4: in northern hinge ─────────────────────────────────────
            elif Z[l] < Asurf1(Y[l]):
                Zc   = Z[l] - Rc * np.sin(angle)
                Yc   = Y[l] - Rc * np.cos(angle)
                idx  = np.argmin(np.abs(Zc - (Yc * abis + bbis)))
                sol_Y, sol_Z = Yc[idx], Zc[idx]
                dez  = np.abs(sol_Z - Z[l])
                dey  = np.abs(sol_Y - Y[l])
                phi3 = np.arctan2(dey, dez)
                hypo = Rc * (beta - phi3)

                if S < hypo:
                    phi1 = S / Rc
                    Y[l] = Y[l] - dey + Rc * np.sin(phi1 + phi3)
                    Z[l] = Z[l] - Rc * np.cos(phi1 + phi3) + dez
                else:
                    Yint = Y[l] - dey + Rc * np.sin(beta)
                    Zint = Z[l] - Rc * np.cos(beta) + dez
                    Y[l] = Yint + (S - hypo) * np.cos(beta)
                    Z[l] = Zint + (S - hypo) * np.sin(beta)

            # ── Zone 5: above northern axial surface ──────────────────────────
            else:
                Y[l] += S * np.cos(beta)
                Z[l] += S * np.sin(beta)

        if n == 1:
            mask               = Y < Y_faille
            Y_save             = Y[mask]
            Z_save             = Z[mask]
            horizontal_shortening = Y_initial[mask] - Y_save

        S += deltaS

    return {
        # Fault trace
        "Yfaille": Yfaille, "Zfaille": Zfaille,
        # Axial surfaces
        "Y_asurf1": Y_asurf1, "Z_asurf1": Z_asurf1,
        "Y_asurf2": Y_asurf2, "Z_asurf2": Z_asurf2,
        "Y_asurf3": Y_asurf3, "Z_asurf3": Z_asurf3,
        "Y_asurf4": Y_asurf4, "Z_asurf4": Z_asurf4,
        # Deformed strata
        "Y_save": Y_save, "Z_save": Z_save,
        "horizontal_shortening": horizontal_shortening,
        # Topo and InSAR (projected)
        "Y_topo": Y_topo, "Z_topo": Z_topo,
        "Y_insar": Y_insar, "Z_insar": Z_insar,
        # Shortening
        "Smax": Smax, "n_tot": n_tot,
        # Hinge points
        "Ych1": Ych1, "Zch1": Zch1,
        "Ych2": Ych2, "Zch2": Zch2,
        "Ych3": Ych3, "Zch3": Zch3,
        "Ych4": Ych4, "Zch4": Zch4,
        # Ramp inflection points
        "Ycrois": Ycrois, "Zcrois": Zcrois,
        "Y_r2": Y_r2, "Z_r2": Z_r2,
        "Y_r3": Y_r3, "Z_r3": Z_r3,
    }


if __name__ == "__main__":
    params = {
        "beta": 32,       # Steep fault segment
        "Y_faille": 29862,
        "Z_faille": 3438,
        "teta": 31,       # Intermediate segment
        "Y_r2": 26000,
        "omega": 30,      # Shallow segment
        "Y_r3": 13000,
        "Ymin": 6000,
        "Ymax": 29000,
        "W": 5000,        # Width of the first hinge
        "W2": 3000,       # Width of the second hinge
        "n_tot": 1,
        "di": 4000,
        "Smax": 30,
    }

    results = compute_fault_and_axial_surfaces(params,
                  Y_topo=np.linspace(6000, 29000, 1000),
                  Z_topo=np.linspace(1800, 3438, 1000),
                  Y_insar=np.linspace(6000, 29000, 500),
                  Z_insar=np.zeros(500))

    fig = plt.figure(layout="constrained", figsize=(8, 8))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[1, 2])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax1b = ax1.twinx()

    ax1.plot(results["Y_topo"], results["Z_topo"], color='black', linewidth=1)
    ax1b.plot(results["Y_insar"], results["Z_insar"], linewidth=1, alpha=0.7, label="InSAR")
    ax1b.plot(results["Y_save"], results["Z_save"] - 3307, '-b', label='Computed deformation')
    ax1b.set_ylabel("Velocity", color='r')

    ax2.plot(results["Y_topo"], results["Z_topo"], color="black")
    ax2.set_xlabel('Horizontal distance (m)')
    ax2.set_ylabel('Depth (m)')
    ax2.plot(results["Yfaille"],  results["Zfaille"],  '-r',  label='Fault')
    ax2.plot(results["Y_asurf1"], results["Z_asurf1"], '-k',  label='Axial surface 1')
    ax2.plot(results["Y_asurf2"], results["Z_asurf2"], '-k',  label='Axial surface 2')
    ax2.plot(results["Y_asurf3"], results["Z_asurf3"], '-k',  label='Axial surface 3')
    ax2.plot(results["Y_asurf4"], results["Z_asurf4"], '-k',  label='Axial surface 4')
    ax2.scatter(results["Ych1"], results["Zch1"], color='green',  s=50, label='Hinge 1')
    ax2.scatter(results["Ych2"], results["Zch2"], color='blue',   s=50, label='Hinge 2')
    ax2.scatter(results["Ych3"], results["Zch3"], color='orange', s=50, label='Hinge 3')
    ax2.scatter(results["Ych4"], results["Zch4"], color='purple', s=50, label='Hinge 4')
    ax2.scatter(results["Ycrois"], results["Zcrois"], color='cyan', s=50, label='ramp1-ramp2 intersection')
    ax2.plot(results["Y_r2"], results["Z_r2"], 'or', label='R2 point')
    ax2.plot(results["Y_r3"], results["Z_r3"], 'or', label='R3 point')
    ax2.legend(loc="upper right")
    ax2.grid(True)
    ax2.axis("equal")
    plt.show()
