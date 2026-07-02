# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def compute_fault_and_axial_surfaces(params, Y_insar, Z_insar):
    """
    Compute fault geometry, axial surfaces and surface deformation
    for a trishear/kink-band kinematic model.

    Parameters
    ----------
    params : dict
        Model parameters (see input file).
    Y_insar, Z_insar : array-like
        Along-profile InSAR velocities.

    Returns
    -------
    dict with fault geometry, axial surfaces, deformed strata, and
    projected InSAR.
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
    Y_fault = params["Y_fault"]
    Z_fault = params["Z_fault"]

    # Ramp equations  (y = a*x + b)
    aramp1 = np.tan(beta);  bramp1 = Z_fault - aramp1 * Y_fault
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
    Smax   = params["Smax"]
    di     = params.get("di", 500)
    deltaS = Smax

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
    Y_fault_trace = np.arange(Ymin, Ymax + 1, dtype=float)
    Z_fault_trace = np.where(
        Y_fault_trace < Ych4, ramp3(Y_fault_trace),
        np.where((Y_fault_trace < Ych3), Cercle2(Y_fault_trace),
        np.where((Y_fault_trace < Ych2), ramp2(Y_fault_trace),
        np.where((Y_fault_trace < Ych1), Cercle(Y_fault_trace),
                                   ramp1(Y_fault_trace)))))

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

    # Evaluate axial surfaces over the full fault-trace domain; the plot axes
    # limits [Ymin, Ymax] control what is actually visible.
    Y_asurf1, Z_asurf1 = Y_fault_trace, Asurf1(Y_fault_trace)
    Y_asurf2, Z_asurf2 = Y_fault_trace, Asurf2(Y_fault_trace)
    Y_asurf3, Z_asurf3 = Y_fault_trace, Asurf3(Y_fault_trace)
    Y_asurf4, Z_asurf4 = Y_fault_trace, Asurf4(Y_fault_trace)

    # ============================================================================
    # SURFACE DEFORMATION  (fully vectorized — no Python for loop)
    # ============================================================================
    G_Y0      = np.linspace(0, Y_fault, di + 1)
    G_Z0      = np.full(di + 1, float(Z_fault))
    Y_initial = G_Y0.copy()

    S = deltaS
    Y = G_Y0.copy()
    Z = G_Z0.copy()   # all points start at Z = Z_fault

    # Precompute scalar trig
    cos_b = np.cos(beta);  sin_b = np.sin(beta)
    cos_t = np.cos(teta);  sin_t = np.sin(teta)
    cos_o = np.cos(omega); sin_o = np.sin(omega)

    # Arc-length and chord constants (computed once, reused across zones)
    dTO = Rc2 * (teta - omega)         # arc through hinge 2
    dTB = Rc  * (beta - teta)          # arc through hinge 1
    deltaZ_char2 = Rc * cos_t
    deltaY_char2 = Rc * sin_t
    # Full-arc chord vectors (constants for zones 1 & 2)
    _hypote_to = 2 * Rc2 * np.sin((teta - omega) / 2)
    _jela_to   = (teta - omega) / 2 + omega
    _cos_jto   = np.cos(_jela_to); _sin_jto = np.sin(_jela_to)
    _hypote_tb = 2 * Rc  * np.sin((beta - teta) / 2)
    _jela_tb   = (beta - teta) / 2 + teta
    _cos_jtb   = np.cos(_jela_tb); _sin_jtb = np.sin(_jela_tb)

    # ── Zone membership ─────────────────────────────────────────────────────
    # All points start at Z = Z_fault (const), so Z[l] < Asurf_i(Y[l]) reduces
    # to a Y threshold (coef < 0 → inequality flips on division).
    thresh4 = (Z_fault - b_surf4) / coef2   # Zone 1 / 2 boundary
    thresh3 = (Z_fault - b_surf3) / coef2   # Zone 2 / 3 boundary
    thresh2 = (Z_fault - b_surf2) / coef    # Zone 3 / 4 boundary
    thresh1 = (Z_fault - b_surf1) / coef    # Zone 4 / 5 boundary

    m1 = Y < thresh4
    m2 = ~m1 & (Y < thresh3)
    m3 = ~m1 & ~m2 & (Y < thresh2)
    m4 = ~m1 & ~m2 & ~m3 & (Y < thresh1)
    m5 = ~m1 & ~m2 & ~m3 & ~m4

    # ── Helper: analytical circle–bisector intersection ─────────────────────
    def _circle_bisect(Yp, Zp, Rc_h, a_bis, b_bis):
        """
        For each point (Yp[i], Zp[i]), find where the circle of radius Rc_h
        centred at that point intersects Z = a_bis*Y + b_bis.
        Returns the root with angle θ ∈ [π, 2π]  (sin θ ≤ 0), which matches
        the original np.argmin scan over angle = linspace(π, 2π, 10000).
        """
        K   = (Zp - a_bis * Yp - b_bis) / Rc_h
        R   = np.sqrt(1.0 + a_bis ** 2)
        phi = np.arctan(a_bis)                           # scalar
        psi = np.arcsin(np.clip(K / R, -1.0, 1.0))      # ∈ [-π/2, π/2]
        th1 = (phi + psi)           % (2 * np.pi)
        th2 = (phi + np.pi - psi)   % (2 * np.pi)
        theta = np.where(np.sin(th1) <= 0, th1, th2)    # pick root in [π, 2π]
        return Yp - Rc_h * np.cos(theta), Zp - Rc_h * np.sin(theta)

    # ── Zone 5: above Asurf1 → ramp1 (β) ────────────────────────────────────
    Y[m5] += S * cos_b
    Z[m5] += S * sin_b

    # ── Zone 4: northern hinge ───────────────────────────────────────────────
    if m4.any():
        Y4 = Y[m4]; Z4 = Z[m4]
        sol_Y4, sol_Z4 = _circle_bisect(Y4, Z4, Rc, abis, bbis)
        dey4   = np.abs(sol_Y4 - Y4)
        dez4   = np.abs(sol_Z4 - Z4)
        phi3_4 = np.arctan2(dey4, dez4)
        hypo4  = Rc * (beta - phi3_4)

        s4a    = S < hypo4
        phi1_4 = np.where(s4a, S / Rc, 0.0)
        c4     = np.where(~s4a, S - hypo4, 0.0)

        Y[m4] = np.where(s4a,
                         Y4 - dey4 + Rc * np.sin(phi1_4 + phi3_4),
                         Y4 - dey4 + Rc * sin_b + c4 * cos_b)
        Z[m4] = np.where(s4a,
                         Z4 - Rc * np.cos(phi1_4 + phi3_4) + dez4,
                         Z4 - Rc * np.cos(beta) + dez4 + c4 * sin_b)

    # ── Zone 3: between hinges → ramp2 (θ) ──────────────────────────────────
    if m3.any():
        Y3 = Y[m3]; Z3 = Z[m3]
        b_t3  = Z3 - Y3 * np.tan(teta)
        Yp3   = (b_t3 - b_surf2) / (coef - np.tan(teta))
        Zp3   = coef * Yp3 + b_surf2
        Yp2_3 = Yp3 - deltaY_char2 + Rc * sin_b
        Zp2_3 = Zp3 + deltaZ_char2 - Rc * cos_b
        Hyp3  = np.hypot(Yp3 - Y3, Zp3 - Z3)

        s3a   = S < Hyp3
        s3b   = ~s3a & (S < Hyp3 + dTB)
        s3c   = ~s3a & ~s3b
        phi3b = np.where(s3b, (S - Hyp3) / Rc, 0.0)
        c3c   = np.where(s3c, S - Hyp3 - dTB, 0.0)

        Y[m3] = np.where(s3a, Y3 + S * cos_t,
                np.where(s3b, Yp3 - deltaY_char2 + np.abs(Rc * np.sin(teta + phi3b)),
                              Yp2_3 + c3c * cos_b))
        Z[m3] = np.where(s3a, Z3 + S * sin_t,
                np.where(s3b, Zp3 - np.abs(Rc * np.cos(teta + phi3b)) + deltaZ_char2,
                              Zp2_3 + c3c * sin_b))

    # ── Zone 2: southern hinge ───────────────────────────────────────────────
    if m2.any():
        Y2 = Y[m2]; Z2 = Z[m2]
        sol_Y2, sol_Z2 = _circle_bisect(Y2, Z2, Rc2, abis2, bbis2)
        dey2   = np.abs(sol_Y2 - Y2)
        dez2   = np.abs(sol_Z2 - Z2)
        phi3_2 = np.arctan2(dey2, dez2)
        dist2  = np.where(Y2 < sol_Y2,
                          Rc2 * (teta + phi3_2),
                          Rc2 * (teta - phi3_2))

        dY1_2  = Rc2 * sin_t;  dZ1_2 = Rc2 * cos_t
        Yinta  = np.where(Y2 < sol_Y2, Y2 + dey2 + dY1_2, Y2 - dey2 + dY1_2)
        Zinta  = Z2 - dZ1_2 + dez2

        b_ri   = Zinta - aramp2 * Yinta
        Yinta2 = (b_ri - b_surf2) / (coef - aramp2)
        Zinta2 = aramp2 * Yinta2 + b_ri
        hyp3_2 = np.hypot(Zinta2 - Zinta, Yinta2 - Yinta)

        dist_int2 = dist2 + hyp3_2
        dist_tot2 = dist2 + hyp3_2 + dTB

        s2a = S < dist2
        s2b = ~s2a & (S < dist_int2)
        s2c = ~s2a & ~s2b & (S < dist_tot2)
        s2d = ~s2a & ~s2b & ~s2c

        # omega > 0 always (lower prior bound = 1°) → skip the omega < 0 branch
        phi1_2 = np.where(s2a, S / Rc2, 0.0)
        c2b    = np.where(s2b, S - dist2, 0.0)
        c2c    = np.where(s2c, S - dist_int2, 0.0)
        phi4_2 = c2c / Rc
        c2d    = np.where(s2d, S - dist_int2 - dTB, 0.0)

        Y[m2] = np.where(s2a, Y2 - dey2 + Rc2 * np.sin(phi1_2 + phi3_2),
                np.where(s2b, Yinta  + c2b * cos_t,
                np.where(s2c, Yinta2 - Rc * sin_t + Rc * np.sin(teta + phi4_2),
                              Yinta2 - Rc * sin_t + Rc * sin_b + c2d * cos_b)))
        Z[m2] = np.where(s2a, Z2 - Rc2 * np.cos(phi1_2 + phi3_2) + dez2,
                np.where(s2b, Zinta  + c2b * sin_t,
                np.where(s2c, Zinta2 + Rc * cos_t - Rc * np.cos(teta + phi4_2),
                              Zinta2 + Rc * cos_t - Rc * cos_b + c2d * sin_b)))

    # ── Zone 1: below Asurf4 → ω ramp, hinge2, hinge1, β ramp ──────────────
    # hyp3 = 0 when zone 1 is processed (zone 1 Y < zone 2 Y in the point order),
    # so the intermediate ramp2 segment is absent and one branch is dead code.
    if m1.any():
        Y1 = Y[m1]; Z1 = Z[m1]
        b_t1  = Z1 - np.tan(omega) * Y1
        Yp1   = (b_t1 - b_surf4) / (coef2 - np.tan(omega))
        hypo1 = (Yp1 - Y1) / cos_o

        s1a = S < hypo1
        s1b = ~s1a & (S < hypo1 + dTO)
        s1d = ~s1a & ~s1b & (S < hypo1 + dTO + dTB)
        s1e = ~s1a & ~s1b & ~s1d

        phi_1b     = np.where(s1b, (S - hypo1) / Rc2, 0.0)
        hypote_1b  = 2 * Rc2 * np.sin(phi_1b / 2)
        jela_1b    = phi_1b / 2 + omega

        c1d        = np.where(s1d, S - hypo1 - dTO, 0.0)
        hypote2_1d = 2 * Rc  * np.sin(c1d / (2 * Rc))
        jela2_1d   = c1d / (2 * Rc) + teta

        c1e = np.where(s1e, S - hypo1 - dTO - dTB, 0.0)

        dY_base = hypo1 * cos_o
        dZ_base = hypo1 * sin_o

        Y[m1] = np.where(s1a, Y1 + S * cos_o,
                np.where(s1b, Y1 + dY_base + hypote_1b * np.cos(jela_1b),
                np.where(s1d, Y1 + dY_base + _hypote_to * _cos_jto
                                            + hypote2_1d * np.cos(jela2_1d),
                              Y1 + dY_base + _hypote_to * _cos_jto
                                           + _hypote_tb * _cos_jtb + c1e * cos_b)))
        Z[m1] = np.where(s1a, Z1 + S * sin_o,
                np.where(s1b, Z1 + dZ_base + hypote_1b * np.sin(jela_1b),
                np.where(s1d, Z1 + dZ_base + _hypote_to * _sin_jto
                                            + hypote2_1d * np.sin(jela2_1d),
                              Z1 + dZ_base + _hypote_to * _sin_jto
                                           + _hypote_tb * _sin_jtb + c1e * sin_b)))

    mask               = Y < Y_fault
    Y_def             = Y[mask]
    Z_def             = Z[mask]
    horizontal_def = Y_initial[mask] - Y_def # calcul du racourcissement: Y_apres_def - Y_ini

    return {
        # Fault trace
        "Y_fault_trace": Y_fault_trace, "Z_fault_trace": Z_fault_trace,
        # Axial surfaces
        "Y_asurf1": Y_asurf1, "Z_asurf1": Z_asurf1,
        "Y_asurf2": Y_asurf2, "Z_asurf2": Z_asurf2,
        "Y_asurf3": Y_asurf3, "Z_asurf3": Z_asurf3,
        "Y_asurf4": Y_asurf4, "Z_asurf4": Z_asurf4,
        # Deformed strata
        "Y_def": Y_def, "Z_def": Z_def,
        "horizontal_def": horizontal_def,
        # Shortening
        "Smax": Smax,
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
        "Y_fault": 29862,
        "Z_fault": 3438,
        "teta": 31,       # Intermediate segment
        "Y_r2": 26000,
        "omega": 30,      # Shallow segment
        "Y_r3": 13000,
        "Ymin": 6000,
        "Ymax": 29000,
        "W": 5000,        # Width of the first hinge
        "W2": 3000,       # Width of the second hinge
        "di": 4000,
        "Smax": 30,
    }

    results = compute_fault_and_axial_surfaces(params,
                  Y_insar=np.linspace(6000, 29000, 500),
                  Z_insar=np.zeros(500))

