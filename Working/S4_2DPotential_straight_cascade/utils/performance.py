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
    Return T (m x m) such that Vt = U∞ * sin(alpha - slope) + T @ gamma.
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

def airfoil_performance_cascade(rho, alpha, beta, chord, vorticity, W, t,
                                ds, xmid, ymid, sine, cosine, slope,
                                plot_cp=False, plot_save=True, print_results=False):

    circulation = np.sum(vorticity * ds)
    CL = circulation / (0.5 * W * chord)
    lift = CL * (0.5*rho*W**2*chord)

    # Build tangential influence once
    T = build_tangential_matrix_cascade(t, ds, sine, cosine, slope, xmid, ymid)

    # Tangential velocity at each panel (just outside on the chosen side):
    Vt = W * np.cos(alpha - slope) + T @ vorticity

    Cp = 1.0 - (Vt / W)**2

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

    if print_results:
        print(f"========== Results ==========\n")
        print(f"Freestream velocity        = {W:.0f} m/s")
        print(f"Freestream angle-of-attack = {np.rad2deg(alpha):.1f} deg")
        print(f"Blade tilt angle           = {np.rad2deg(beta):.1f} deg (+ve means downwards tilt)")
        print(f"Effective angle-of-attack  = {np.rad2deg(alpha-beta):.1f} deg")
        print(f"Circulation                = {circulation:.2f} m^2/s")
        print(f"Lift coefficient           = {CL:.2f}")
        print(f"Lift                       = {lift:.2f} N")
        print(f"\n========== End ==========\n")

    # Save Results
    results = dict()
    results["vorticity"] = vorticity
    results["circulation"] = circulation
    results["CL"] = CL
    results["Lift"] = lift
    results["Cp"] = Cp

    return results

def calculate_exit_conditions(results_u, results_v, t, alpha, U, V):

    # exit angle
    a = results_u["circulation"]/(2.0*t*U)
    b = results_v["circulation"]/(2.0*t*U)
    alpha_exit = np.atan( ((1-b)/(1+b))*np.tan(alpha) - (2/(1+b))*a )

    # exit velocity
    W_exit = U / np.cos(alpha_exit)
    V_exit = U * np.tan(alpha_exit)

    return np.rad2deg(alpha_exit), W_exit, V_exit

def cascade_performance(results, rho, W_start, W_end, chord):

    W_inter = ((W_start+W_end)*0.5)

    results["Lift"] = results["circulation"] * rho * W_inter
    results["CL"]   = results["Lift"] / (0.5 * rho * W_inter**2 * chord)

    return results

def display_cascade_performance(results, blade):

    print("\nMulti-Aerofoil case")
    print(f"========== Results ==========\n")
    print(f"Freestream velocity        = {results["W_start"]:.0f} m/s")
    print(f"Freestream angle-of-attack = {results["alpha_start"]:.1f} deg")
    print(f"Blade tilt angle           = {blade["tilt_angle"]:.1f} deg (-ve means downwards tilt)")
    print(f"Effective angle-of-attack  = {results["alpha_start"]+blade["tilt_angle"]:.1f} deg (-ve means downwards tilt)")
    print(f"Circulation                = {results["circulation"]:.2f} m^2/s")
    print(f"Lift coefficient           = {results["CL"]:.2f}")
    print(f"Lift                       = {results["Lift"]:.2f} N")
    print(f"Flow angle after airfoil   = {results["alpha_exit"]:.2f} deg")
    print(f"\n========== End ==========\n")