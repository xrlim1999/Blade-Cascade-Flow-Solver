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

    # --- vorticity and circulation ---
    # orientation: +1 = CCW, -1 = CW
    circulation = np.sum(gamma_b * ds)

    # --- Lift and Lift Coefficient (per dr) ---
    CL   = 2.0 * circulation / (U_1 * chord)
    lift = CL * (0.5 * rho * (U_1**2) * chord)

    # --- Surface Velocity ---
    T = build_tangential_matrix_isolated(xmid, ymid, ds, cosine, sine)

    # Induced tangential velocity from the bound sheet
    Vs = W_1* np.cos(alpha_1 - slope) + T @ gamma_b  # shape (m,)
    
    # ======================================
    ## Aerofoil surface pressure
    # ======================================
    Cp = 1 - (Vs / U_1)**2

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

        plt.gca().invert_yaxis()
        plt.xlabel('X (normalised by chord)')
        plt.ylabel("Pressure Coefficient (Cp)", fontsize=14)
        plt.title(f"Pressure Coefficient around aerofoil for\n{np.rad2deg(alpha_1):.0f}deg AoA & {beta:.0f}deg downwards aerofoil tilt", fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, which='both', linestyle='--', alpha=0.6)

        if plot_save:
            # --- create a folder if it doesn't exist ---
            out_dir = "figures/Cp"
            os.makedirs(out_dir, exist_ok=True)

            # --- save to file ---
            filename = f"Cp_alpha{np.rad2deg(alpha_1):.0f}_beta{beta_deg:.0f}.png"
            filepath = os.path.join(out_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight') # high-res, cropped
            plt.close()

    # ===============================================
    ## Blade force in axial and tangential directions
    # ===============================================
    lift_y = lift * np.cos(beta)
    lift_x = lift * np.sin(beta)

    # ======================================
    ## Results
    # ======================================
    results = dict()
    results["vorticity"]   = gamma_b
    results["circulation"] = circulation
    results["CL"]          = CL
    results["Lift"]        = lift
    results["lift_axial"]  = lift_x ; results["lift_tan"] = lift_y
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


def display_airfoil_performance(results, airfoil):

    print(f"\n-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --")
    print("Results : AIRFOIL\n" \
          "Solver  : Vortex Panel Method\n" \
          "Flow    : Inviscid\n")


    print(
        f"U_in  = {results['U_in']:2.1f} m/s  |  "
        f"V_in  = {results['V_in']:2.1f} m/s  |  "
        f"W_in  = {results['W_in']:2.1f} m/s  |  "
        f"alpha_in  = {results['alpha_in']:2.2f} deg"
    )

    print(f"Airfoil tilt angle         = {airfoil['tilt_angle']:.1f} deg (-ve means downwards tilt)")
    print(f"Effective angle-of-attack  = {results['alpha_in']+airfoil['tilt_angle']:.1f} deg (-ve means downwards tilt)\n")
    
    print(f"Circulation      = {results['circulation']:.2f} m^2/s")
    print(f"Lift coefficient = {results['CL']:.2f}")
    print(f"Lift             = {results['Lift']:.0f} N | {results['Lift']/9.81:.0f} kg ")
    print(f"Fy               = {results['lift_tan']:.0f} N | {results['lift_tan']/9.81:.0f} kg (force in tangential direction | torque force) ")
    print(f"Fx               = {results['lift_axial']:.0f} N | {results['lift_axial']/9.81:.0f} kg (force in axial direction | +ve = +x direction)")

    print(f"-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --\n")

