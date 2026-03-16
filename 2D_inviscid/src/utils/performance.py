import numpy as np
import os

import matplotlib
from matplotlib.path import Path
matplotlib.use('Qt5Agg')  # or 'Qt6Agg' if you have PyQt6
import matplotlib.pyplot as plt


def dedupe_te_node(x, y, tol=1e-12):
    if np.isclose(x[0], x[-1], atol=tol) and np.isclose(y[0], y[-1], atol=tol):
        x = x[:-1].copy(); y = y[:-1].copy()
    return x, y


def build_tangential_matrix_isolated(xmid, ymid, ds, cosine, sine):
    """
    Return T (m x m) so that PV tangential velocity = (T @ gamma).
    Uses free-space 2D vortex kernel (no pitch t).
    NOTE: This T is the *principal-value* operator -> diagonal = 0.0.
          Add ±0.5*gamma later to get one-sided surface velocities.
    """
    m = len(xmid)
    T = np.zeros((m, m))
    tiny = 1e-14

    for i in range(m):
        tx_i = cosine[i]   # tangent unit vector components at target i
        ty_i = sine[i]
        xi, yi = xmid[i], ymid[i]

        for j in range(m):
            if i == j:
                continue
            dx = xmid[j] - xi
            dy = ymid[j] - yi
            r2 = dx*dx + dy*dy + tiny

            # 2D point-vortex induced velocity (per unit circulation), then × ds_j:
            # u =  (γ/2π) *  dy / r^2
            # v = -(γ/2π) *  dx / r^2
            u =  (0.5/np.pi) * (dy / r2)
            v = -(0.5/np.pi) * (dx / r2)

            # project onto *target* panel's tangent
            T[i, j] = (u*tx_i + v*ty_i) * ds[j]

        # principal value: no self term here
        T[i, i] = 0.0

    return T


def airfoil_performance(geom: dict, flow: dict, coup, gamma_b, 
                         plot_cp=False, plot_save=True):

    # --- unpack geometry ---
    xmid, ymid   = geom["xmid"], geom["ymid"]
    x, y         = geom["x"]   , geom["y"]
    ds           = geom["ds"]
    slope        = geom["slope"]
    sine         = geom["sine"]
    cosine       = geom["cosine"]
    beta         = geom["beta"]
    beta_deg     = np.rad2deg(beta)
    chord        = geom["chord"]

    # --- unpack flow ---
    U_1 = float(flow["U"])
    V_1 = float(flow["V"])
    W_1 = float(flow["W"])
    alpha_1 = flow["alpha_1"]
    rho = flow["rho"]
    
    # --- airfoil surface pressure ---
    # Induced tangential velocity from the bound sheet
    T = build_tangential_matrix_isolated(xmid, ymid, ds, cosine, sine)
    Vs = W_1 * np.cos(alpha_1 - slope) + T @ gamma_b  # shape (m,)

    Cp = 1 - (Vs / W_1)**2

    # --- forces (x & y direction) ---
    q_inf = 0.5 * rho * (W_1**2) # dynamic pressure [Pa]
    
    # xy-axis (parallel, normal) vector
    nx, ny = sine, -cosine

    # force per panel
    scalar = - q_inf * Cp * ds
    dF_x = scalar * nx # Fx (x-direction)
    dF_y = scalar * ny # Fy (y-direction)
    dF   = np.column_stack((dF_x, dF_y)) # (Fx, Fy)

    # total force
    F_x, F_y = dF.sum(axis=0)
    F_xy_vec = np.array([F_x, F_y])

    # --- forces (flow -parallel & -normal direction) ---
    # normal & parallel vectors
    e_p = np.array([ np.cos(alpha_1), np.sin(alpha_1)]) # chord-parallel vector
    e_n = np.array([-np.sin(alpha_1), np.cos(alpha_1)]) # chord-normal vector

    # total force
    F_f_parallel = float(F_xy_vec @ e_p) # force along chord (downstream is +ve)
    F_f_normal   = float(F_xy_vec @ e_n) # force normal to chord (up is +ve)
    F_flow_vec = np.array([F_f_parallel, F_f_normal])

    # force coefficient
    forcecoeff_flow = F_flow_vec / (q_inf * chord) # (Cd_chord, Cl_chord)

    # circulation
    circulation_flow = F_f_normal / (rho * W_1) # in airfoil frame axis

    # --- forces (airfoil -parallel & -normal direction) ---
    # normal & parallel vectors
    n_p = np.array([ np.cos(beta), np.sin(beta)]) # chord-parallel vector
    n_n = np.array([-np.sin(beta), np.cos(beta)]) # chord-normal vector

    # total force
    F_a_parallel = float(F_xy_vec @ n_p) # force along chord (downstream is +ve)
    F_a_normal   = float(F_xy_vec @ n_n) # force normal to chord (up is +ve)
    F_airfoil_vec = np.array([F_a_parallel, F_a_normal])

    # force coefficient
    forcecoeff_airfoil = F_airfoil_vec / (q_inf * chord) # (Cd_chord, Cl_chord)

    # circulation
    circulation_airfoil = F_a_normal / (rho * W_1) # in airfoil frame axis

    if plot_cp:
        # --- TE-clean plotting: drop the two TE corner nodes and draw a short bridge ---
        ile = int(np.argmin(xmid))                # LE index

        x_u, Cp_u = xmid[1:ile+1],  Cp[1:ile+1] # TE(upper)->LE
        x_l, Cp_l = xmid[ile:-1] , Cp[ile:-1]   # LE->TE(lower)

        plt.figure(figsize=(8,6))
        plt.plot(x_u, Cp_u, 'b-', linewidth=1.5, label="Upper surface")
        plt.plot(x_l, Cp_l, 'r-', linewidth=1.5, label="Lower surface")

        # short “bridge” at TE to visually connect the last two points
        x_te = xmid[0]
        plt.plot([x_te, x_te], [Cp_u[0], Cp_l[-1]], 'b-', linewidth=1.5)

        # plot limit
        plt.ylim([min(Cp_u.min(), Cp_l.min()), 1.0])

        plt.gca().invert_yaxis()
        plt.xlabel('X (normalised by chord)')
        plt.ylabel("Pressure Coefficient ($C_P$)", fontsize=14)
        plt.title(
             "Pressure Coefficient around aerofoil for\n"
            fr"freestream flow angle $\alpha_{{\infty}}$ = {np.rad2deg(alpha_1):.1f}°" "\n"
            fr"airfoil tilt angle $\beta$ = {beta_deg:.1f}° (+ve = LE points up)",
            fontsize=12
        )
        plt.legend(loc='best')
        plt.minorticks_on()  # enable minor ticks
        plt.grid(True, which='both', linestyle=':', linewidth=0.5)

        if plot_save:
            # --- create a folder if it doesn't exist ---
            out_dir = "figures/Cp"
            os.makedirs(out_dir, exist_ok=True)

            # --- save to file ---
            filename = f"Cp_alpha{np.rad2deg(alpha_1):.1f}_beta{beta_deg:.1f}.png"
            filepath = os.path.join(out_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight') # high-res, cropped
            plt.close()

    # ======================================
    ## Results
    # ======================================
    results = dict()
    results["vorticity"]           = gamma_b
    results["circulation_flow"]    = circulation_flow
    results["circulation_airfoil"] = circulation_airfoil
    results["force_flow"]          = F_flow_vec
    results["force_airfoil"]       = F_airfoil_vec
    results["forcecoeff_flow"]     = forcecoeff_flow
    results["forcecoeff_airfoil"]  = forcecoeff_airfoil
    results["Cp"]          = Cp
    results["V_surface"]   = Vs
    results["U_in"]        = U_1
    results["V_in"]        = V_1
    results["alpha_in"]    = np.rad2deg(alpha_1)
    results["W_in"]        = W_1

    airfoil = dict()
    airfoil["tilt_angle"]     = beta_deg
    airfoil["chord"]          = chord
    airfoil["airfoil_coords"] = np.column_stack((xmid, ymid))

    return results, airfoil


def display_airfoil_performance(results, flow, airfoil):

    # --- unpack values ---
    Re = flow["Re"]

    Fx   , Fy    = results["force_flow"][0]     , results["force_flow"][1]
    Cd_Fx, Cl_Fy = results["forcecoeff_flow"][0], results["forcecoeff_flow"][1]

    drag_airfoil, lift_airfoil = results["force_airfoil"][0]     , results["force_airfoil"][1]
    Cd_airfoil  , Cl_airfoil   = results["forcecoeff_airfoil"][0], results["forcecoeff_airfoil"][1]

    print(f"\n-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --")
    print("Results : AIRFOIL\n" \
          "Solver  : Vortex Panel Method\n" \
          "Flow    : Inviscid\n" \
          f"Reynolds number  = {Re/1.0e6:.2f} (x10^6)\n"
    )
    

    print(
        f"U_in  = {results['U_in']:2.1f} m/s  |  "
        f"V_in  = {results['V_in']:2.1f} m/s  |  "
        f"W_in  = {results['W_in']:2.1f} m/s  |  "
        f"alpha_in  = {results['alpha_in']:2.2f} deg"
    )

    print(f"Airfoil tilt angle         = {airfoil['tilt_angle']:.1f} deg (+ve means upwards tilt)")
    print(f"Effective angle-of-attack  = {results['alpha_in']+airfoil['tilt_angle']:.1f} deg (+ve means upwards tilt)\n")

    print("(\nFreestream frame)")
    print(f"Fy               = {Fy:.0f} N | {Fy/9.81:.0f} kg (force in y-direction) ")
    print(f"Fx               = {Fx:.0f} N | {Fx/9.81:.0f} kg (force in x-direction)")
    print(f"Lift coefficient = {Cl_Fy:.2f}")
    print(f"Drag coefficient = {Cd_Fx:.2f}")
    print(f"Circulation      = {results['circulation_flow']:.2f} m^2/s")
    
    print("(\nAirfoil frame)")
    print(f"Lift             = {lift_airfoil:.0f} N | {lift_airfoil/9.81:.0f} kg ")
    print(f"Drag             = {drag_airfoil:.0f} N | {drag_airfoil/9.81:.0f} kg ")
    print(f"Lift coefficient = {Cl_airfoil:.2f}")
    print(f"Drag coefficient = {Cd_airfoil:.2f}")
    print(f"L/D ratio        = {lift_airfoil/drag_airfoil:.2f}")
    print(f"Circulation      = {results['circulation_airfoil']:.2f} m^2/s")

    print(f"-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --\n")

