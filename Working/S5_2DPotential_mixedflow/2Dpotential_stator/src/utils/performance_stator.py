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

def airfoil_performance_in_cascade(rho, alpha, beta, chord, t,
                                   ds, xmid, ymid, sine, cosine, slope, U, V,
                                   vorticity_U, vorticity_V, cascade_type,
                                   plot_cp=False, plot_save=True):
    # --------- Helper terms ---------
    alpha_in = alpha # upstream flow angle
    W = np.sqrt(U**2 + V**2) # absolute velocity
    vorticity_total = (U * vorticity_U) + (V * vorticity_V) 

    circulation_U = np.sum(vorticity_U * ds)
    circulation_V = np.sum(vorticity_V * ds)
    circulation   = np.sum(vorticity_total * ds)

    # ======================================
    ## Lift Force
    # ======================================
    CL          = 2 * circulation / (U * chord)
    lift        = CL * (0.5*rho*U**2*chord)

    # ======================================
    ## Aerofoil surface velocity
    # ======================================
    # # Build tangential influence once
    T = build_tangential_matrix_cascade(t, ds, sine, cosine, slope, xmid, ymid)

    # Build tangential influence once
    T = build_tangential_matrix_cascade(t, ds, sine, cosine, slope, xmid, ymid)

    # Tangential velocity at each panel (just outside on the chosen side):
    Vs = ( U*np.cos(slope) + V*np.sin(slope) ) + T @ vorticity_total

    # ======================================
    ## Aerofoil surface pressure
    # ======================================
    Cp = 1 - (Vs / U)**2

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
        plt.title(f"Pressure Coefficient around aerofoil for\n{np.rad2deg(alpha):.0f}deg AoA & {np.rad2deg(beta):.0f}deg downwards aerofoil tilt", fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, which='both', linestyle='--', alpha=0.6)

        if plot_save:
            # --- create a folder if it doesn't exist ---
            out_dir = "figures"
            os.makedirs(out_dir, exist_ok=True)

            # --- save to file ---
            filename = f"Cp_alpha{np.rad2deg(alpha):.0f}_beta{np.rad2deg(beta):.0f}.png"
            filepath = os.path.join(out_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight') # high-res, cropped
            print("here")
            plt.close()

    # ==============================
    ## Exit flow angle and velocity
    # ==============================
    # 1. angle
    a = circulation_U/(2.0*t*U)
    b = circulation_V/(2.0*t*U)
    alpha_out = np.atan( ((1-b)/(1+b))*np.tan(alpha) - (2/(1+b))*a )

    # 2. velocity
    W_out = U / np.cos(alpha_out)
    V_out = U * np.tan(alpha_out)

    # ===============================================
    ## Blade force in axial and tangential directions
    # ===============================================
    if cascade_type == "stator":
        # ===== stator row =====
        lift_axial = - lift * np.sin(alpha_in)
        lift_tan   =   lift * np.cos(alpha_in)
    elif cascade_type == "rotor":
        # ===== rotating row =====
        lift_axial = - lift * np.sin(-beta)
        lift_tan   =   lift * np.cos( beta)
    else:
        raise ValueError("Wrong cascade-type input : choose only stator or rotor")

    # ======================================
    ## Results
    # ======================================
    # ------ Save Results ------
    results = dict()
    results["vorticity"] = vorticity_total
    results["circulation"] = circulation
    results["CL"] = CL
    results["Lift"] = lift
    results["lift_axial"] = lift_axial ; results["lift_tan"] = lift_tan
    results["Cp"] = Cp
    results["V_surface"] = Vs
    results["U_in"] = U ; results["U_out"] = U
    results["V_in"] = V ; results["V_out"] = V_out
    results["alpha_in"] = np.rad2deg(alpha_in) ; results["alpha_out"] = np.rad2deg(alpha_out)
    results["W_in"] = W ; results["W_out"] = W_out

    blade = dict()
    blade["tilt_angle"] = np.rad2deg(beta)
    blade["pitch"]      = t
    blade["chord"]      = chord

    return results, blade

def display_cascade_performance(results, blade):

    print(f"\n========== Stator Cascade : Results ==========\n")
    print(f"Blade tilt angle           = {blade["tilt_angle"]:.1f} deg (-ve means downwards tilt)")
    print(f"Effective angle-of-attack  = {results["alpha_in"]+blade["tilt_angle"]:.1f} deg (-ve means downwards tilt)\n")
    
    print(f"Circulation      = {results["circulation"]:.2f} m^2/s")
    print(f"Lift coefficient = {results["CL"]:.2f}")
    print(f"Lift             = {results["Lift"]:.0f} N | {results["Lift"]/9.81:.0f} kg ")
    print(f"Fy               = {results["lift_tan"]:.0f} N | {results["lift_tan"]/9.81:.0f} kg (force in tangential direction | torque force) ")
    print(f"Fx               = {results["lift_axial"]:.0f} N | {results["lift_axial"]/9.81:.0f} kg (force in axial direction | +ve = drag)  \n")


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

