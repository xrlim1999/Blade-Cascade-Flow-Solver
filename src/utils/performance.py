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

def build_tangential_matrix_cascade(t, ds, sine, cosine, slope, xmid, ymid):
    """
    Return T (m x m) such that Vt = U∞*sin(alpha - slope) + T @ gamma.
    Uses the same periodic kernel as your coupling matrix, but projects onto the
    TARGET panel's tangent instead of normal.
    """
    m = xmid.shape[0]
    T = np.zeros((m, m))

    slope = np.unwrap(slope)
    ds_min = float(np.min(ds))
    core   = 0.2 * ds_min
    core_over_t_sq = (2.0*np.pi*core / t)**2
    tiny = 1e-14

    for i in range(m):
        tx_i = cosine[i]
        ty_i = sine[i]

        for j in range(m):
            if j == i:
                continue

            dx = xmid[j] - xmid[i]
            dy = ymid[j] - ymid[i]

            X = 2.0*np.pi*dx / t
            Y = 2.0*np.pi*dy / t

            denom = np.cosh(X) - np.cos(Y)
            denom = max(denom, 0.5*core_over_t_sq, tiny)

            # Lewis periodic kernel components (same as in your coup)
            u =  (0.5 / t) * (np.sin(Y)  / denom)
            v = -(0.5 / t) * (np.sinh(X) / denom)

            # Project the induced velocity onto the *target* panel's tangent at i
            # Panel-i unit tangent components:
            # tx_i = -sine[i]   # because tangent is +90° from normal (cos,sin) used in your code
            # ty_i =  cosine[i]
            # (If your tangent definition is tx=dx/S, ty=dy/S with CCW, use that instead:
            # tx_i, ty_i = something you already have. The key is: project onto TANGENT of i.)

            T[i, j] = (u * tx_i + v * ty_i) * ds[j]

    # Principal value self-term: +0.5*gamma_i
    # This is the standard contribution of the sheet to its own tangential velocity jump.
    for i in range(m):
        T[i, i] = 0.5

    return T

def bladerow_performance(geom, flow, gamma_b, 
                         plot_cp=False, plot_save=True):

    # --- unpack geometry ---
    xmid, ymid   = geom["xmid"], geom["ymid"]
    ds           = geom["ds"]
    sine, cosine = geom["sine"], geom["cosine"]
    slope        = geom["slope"]
    beta_deg     = geom["beta"]
    t            = geom["pitch"]
    r            = geom["r"]
    chord        = geom["chord"]

    # --- unpack flow ---
    U_1 = float(flow["U"])
    V_1 = float(flow["V"])
    W_1 = np.sqrt(U_1**2 + V_1**2)
    Omega = flow.get("Omega", 0.0)
    alpha_1 = flow["alpha_1"]
    rho = flow["rho"]

    # --- vorticity and circulation ---
    vorticity = gamma_b
    circulation = np.sum(vorticity * ds)

    # --- tangential change ---
    delta_Vy = circulation / t

    # --- absolute (stationary frame) velocities and angles ---
    V_2 = float(V_1 + delta_Vy)
    alpha1 = alpha_1
    alpha2 = np.arctan2(V_2, U_1)

    # --- rotating-frame angles (relative) ---
    alpha1_rel = np.arctan2(V_1 - Omega*r, U_1)   # inlet, relative frame  :contentReference[oaicite:6]{index=6}
    alpha2_rel = np.arctan2(V_2 - Omega*r, U_1)   # exit,  relative frame
    
    # ======================================
    ## Lift Force (per dr)
    # ======================================
    CL   = 2 * circulation / (U_1 * chord)
    lift = CL * (0.5 * rho * (U_1**2) * chord)

    # ======================================
    ## Aerofoil surface velocity
    # ======================================
    # # Build tangential influence once
    T = build_tangential_matrix_cascade(t, ds, sine, cosine, slope, xmid, ymid)

    Vs = ( U_1*np.cos(slope) + V_1*np.sin(slope) ) + T @ vorticity
    
    # ======================================
    ## Aerofoil surface pressure
    # ======================================
    Cp = 1 - (Vs / U_1)**2

    if plot_cp:
        # --- TE-clean plotting: drop the two TE corner nodes and draw a short bridge ---
        ile = int(np.argmin(xmid))                # LE index

        x_u, Cp_u = xmid[1:ile+1],  Cp[1:ile+1]   # TE(upper)->LE
        x_l, Cp_l = xmid[ile:-1], Cp[ile:-1]  # LE->TE(lower)

        plt.figure(figsize=(8,6))
        plt.plot(x_u, Cp_u, 'b-', linewidth=1.5, label="Upper surface")
        plt.plot(x_l, Cp_l, 'r-', linewidth=1.5, label="Lower surface")

        # short “bridge” at TE to visually connect the last two points
        x_te = xmid[0]
        plt.plot([x_te, x_te], [Cp_u[0], Cp_l[-1]], 'b-', linewidth=1.5)

        plt.gca().invert_yaxis()
        plt.xlabel('X (normalised by chord)')
        plt.ylabel("Pressure Coefficient (Cp)", fontsize=14)
        plt.title(f"Pressure Coefficient around aerofoil for\n{np.rad2deg(alpha_1):.0f}deg AoA & {np.rad2deg(beta_deg):.0f}deg downwards aerofoil tilt", fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, which='both', linestyle='--', alpha=0.6)

        if plot_save:
            # --- create a folder if it doesn't exist ---
            out_dir = "figures"
            os.makedirs(out_dir, exist_ok=True)

            # --- save to file ---
            filename = f"Cp_alpha{np.rad2deg(alpha_1):.0f}_beta{np.rad2deg(beta_deg):.0f}.png"
            filepath = os.path.join(out_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight') # high-res, cropped
            plt.close()

    # ===============================================
    ## Blade force in axial and tangential directions
    # ===============================================
    lift_axial = lift * np.sin(beta_deg)
    lift_tan   = lift * np.cos(beta_deg)

    # ======================================
    ## Results
    # ======================================
    # ------ Save Results ------
    results = dict()
    results["vorticity"]   = vorticity
    results["circulation"] = circulation
    results["CL"]          = CL
    results["Lift"]        = lift
    results["lift_axial"]  = lift_axial ; results["lift_tan"] = lift_tan
    results["Cp"]          = Cp
    results["V_surface"]   = Vs
    results["U_in"]        = U_1 ; results["U_out"] = U_1
    results["V_in"]        = V_1 ; results["V_out"] = V_2
    results["alpha_in"]    = np.rad2deg(alpha1) ; results["alpha_out"] = np.rad2deg(alpha2)
    results["alpha_in_rel"] = np.rad2deg(alpha1_rel) ; results["alpha_out_rel"] = np.rad2deg(alpha2_rel)
    results["W_in"]        = W_1 ; results["W_out"] = np.sqrt(U_1**2 + V_2**2)

    blade = dict()
    blade["tilt_angle"]     = np.rad2deg(beta_deg)
    blade["pitch"]          = t
    blade["chord"]          = chord
    blade["RPM"]            = Omega/(2*np.pi) * 60
    blade["airfoil_coords"] = np.column_stack((xmid, ymid))

    return results, blade

def display_cascade_performance(results, blade):

    print(f"\n========== Cascade : Results ==========\n")
    print(f"Blade tilt angle           = {blade['tilt_angle']:.1f} deg (-ve means downwards tilt)")
    print(f"Effective angle-of-attack  = {results['alpha_in']+blade['tilt_angle']:.1f} deg (-ve means downwards tilt)\n")
    
    print(f"Circulation      = {results['circulation']:.2f} m^2/s")
    print(f"Lift coefficient = {results['CL']:.2f}")
    print(f"Lift             = {results['Lift']:.0f} N | {results['Lift']/9.81:.0f} kg ")
    print(f"Fy               = {results['lift_tan']:.0f} N | {results['lift_tan']/9.81:.0f} kg (force in tangential direction | torque force) ")
    print(f"Fx               = {results['lift_axial']:.0f} N | {results['lift_axial']/9.81:.0f} kg (force in axial direction | +ve = drag)  \n")


    print(
        f"U_in  = {results['U_in']:2.1f} m/s  |  "
        f"V_in  = {results['V_in']:2.1f} m/s  |  "
        f"W_in  = {results['W_in']:2.1f} m/s  |  "
        f"alpha_in  = {results['alpha_in']:2.2f} deg"
    )

    print(
        f"U_out = {results['U_out']:2.1f} m/s  |  "
        f"V_out = {results['V_out']:2.1f} m/s  |  "
        f"W_out = {results['W_out']:2.1f} m/s  |  "
        f"alpha_out = {results['alpha_out']:2.2f} deg"
    )

    print(f"\n========== End ==========\n")

