# FILE: src/main.py

from .utils import AirfoilSection, CascadeSolver
import numpy as np

import matplotlib
from matplotlib.path import Path
matplotlib.use('Qt5Agg')  # or 'Qt6Agg' if you have PyQt6
import matplotlib.pyplot as plt
import os

"""
Fan Profile
"""
# --- fan cascade sections ---
n_stages = 2

# --- Cascade Type ---
cascade_type_arr = np.array(["rotor" if i % 2 == 0 else "stator" for i in range(n_stages)])

# --- fan radius ---
r_hub = 0.30 # [m]
r_tip = 0.68 # [m]

# --- number of blades ---
n_blades_rotor  = 4
n_blades_stator = 12
n_blades_arr = np.zeros(n_stages)
n_blades_arr = np.array([n_blades_rotor if ctype == "rotor" else n_blades_stator for ctype in cascade_type_arr])

# --- fan sections ---
n_sections = 20

# --- radial positions --- 
r_stations = np.linspace(r_hub, r_tip, num=(n_sections+1))

r1_arr = r_stations[:-1]
r2_arr = r_stations[1:]

r_c_arr = (r1_arr + r2_arr) / 2 # camber-line      [assumed to be midpoint of start and end radial position of section]
r_m_arr = (r1_arr + r2_arr) / 2 # aerofoil surface [assumed to be midpoint of start and end radial position of section]

dr_arr = np.zeros(n_sections)
dr_arr = r2_arr - r1_arr


"""
Flow Conditions (Upstream of fan-face)
"""
rho    = 1.225           # static density    [kg/m^2]
W0     = 100.0           # absolute velocity [m/s]
alpha0 = np.deg2rad(0.0) # freestream angle  [deg]
U0, V0 = W0*np.cos(alpha0), W0*np.sin(alpha0) # velocity in axial, tangential directions [m/s]

# --- store in arrays ---
rho_arr   = np.zeros((n_sections, n_stages+1))
W_arr     = np.zeros((n_sections, n_stages+1))
U_arr     = np.zeros((n_sections, n_stages+1))
V_arr     = np.zeros((n_sections, n_stages+1))
alpha_arr = np.zeros((n_sections, n_stages+1))

rho_arr  [:,0] = rho
W_arr    [:,0] = W0
alpha_arr[:,0] = alpha0
U_arr    [:,0] = U0
V_arr    [:,0] = V0

"""
Necessary Matrices (initialisation)
"""
# --- Blade Section Results ---
results_fan = dict()
results_fan["Effective angle-of-attack"] = np.zeros((n_sections, n_stages+1))
results_fan["CL"]                        = np.zeros((n_sections, n_stages+1))
results_fan["Lift"]                      = np.zeros((n_sections, n_stages+1))
results_fan["Circulation"]               = np.zeros((n_sections, n_stages+1))
results_fan["Fy"]                        = np.zeros((n_sections, n_stages+1))
results_fan["Fx"]                        = np.zeros((n_sections, n_stages+1))

# --- Cascade Layer Results ---
results_cascade = dict()
results_cascade["Torque"] = np.zeros((n_sections, n_stages))
results_cascade["Power"]  = np.zeros((n_sections, n_stages))
results_cascade["Drag"]   = np.zeros((n_sections, n_stages))


"""
Blade Profile
"""
# --- Blade Stagger angle ---
blade_tilt_arr      = np.zeros((n_sections, n_stages+1))
blade_tilt_arr[:,0] = np.linspace(-60.0, -10.0, n_sections)
blade_tilt_arr[:,1] = 0.0

# --- Blade Chord ---
chord_arr      = np.zeros((n_sections, n_stages))
chord_arr[:,0] = np.linspace(0.3, 0.2, n_sections)
chord_arr[:,1] = np.linspace(0.3, 0.2, n_sections)

# --- Blade Chord ---
t_ratio_arr      = np.zeros((n_sections, n_stages))
t_ratio_arr = (((2*np.pi) * r_c_arr)[:, None] / n_blades_arr[None, :]) / chord_arr

# --- RPM (for rotor only) ---
RPM_arr    = np.zeros(n_stages)
RPM_arr[0] = 2000
Omega_arr  = (RPM_arr/60) * (2*np.pi)

m = 100

"""
Solve for all cascade layers
"""
for stage in range(n_stages):

    cascade_type = cascade_type_arr[stage]

    beta_deg = np.deg2rad(blade_tilt_arr[:,stage])
    alpha_in = np.deg2rad(alpha_arr[:,stage])

    for k in range(n_sections):

        # --- Initialise Class : Geometry + solver ---
        section = AirfoilSection("src/data/aerofoil_coords.txt", m, beta_deg[k], t_ratio_arr[k,stage])
        solver  = CascadeSolver(section, cascade_type, r_c_arr[k], r_m_arr[k], r1_arr[k], r2_arr[k])

        # --- Solve ---
        results, blade = solver.solve(rho, alpha_in[k], U_arr[k,stage], V_arr[k,stage], Omega_arr[stage], plot_cp=False)

        # --- Store Output Values (for next cascade layer) ---
        U_arr[k, stage+1] = results["U_out"]
        V_arr[k, stage+1] = results["V_out"]
        W_arr[k, stage+1] = results["W_out"]
        alpha_arr[k, stage+1] = results["alpha_out"]

        # --- Save Results ---
        results_fan["Effective angle-of-attack"][k, stage] = np.rad2deg(alpha_in[k] + beta_deg[k]) # [ deg ]
        results_fan["CL"]                       [k, stage] = results["CL"]                   # [ - ]
        results_fan["Lift"]                     [k, stage] = results["Lift"]                 # [ N ]
        results_fan["Circulation"]              [k, stage] = results["circulation"]          # [ m^2/s ]
        results_fan["Fy"]                       [k, stage] = results["lift_tan"]             # [ N ]
        results_fan["Fx"]                       [k, stage] = results["lift_axial"]           # [ N ]

      ## <<< End of current blade section >>>

    """ Analysis (cascade layer) """
    # if cascade_type == "rotor":
    results_cascade["Torque"][:,stage] = results_fan["Fy"][:,stage] * r_c_arr * dr_arr         # torque = Fy[tangential force] x r[distance to rotor center] x dr[blade section length]
    results_cascade["Power"] [:,stage] = results_cascade["Torque"][:,stage] * Omega_arr[stage] # power  = torque x r[distance to rotor center]
    results_cascade["Drag"]  [:,stage] = results_fan["Fx"][:,stage] * dr_arr                   # drag   = Fx[axial force] x dr[blade section length]

    cascade_results = True
    if cascade_results:
        if cascade_type == "rotor":
            print(f"\n============ Cascade ({cascade_type}) : Results ============")
            
            print(f"{'Power':<8}: {np.sum(results_cascade["Power"]/1000*n_blades_rotor):>5.2f} kW  |")
            print(f"{'Torque':<8}: {np.sum(results_cascade["Torque"]/1000*n_blades_rotor):>5.2f} kNm |")
            print(f"{'Drag':<8}: {np.sum(results_cascade["Drag"]*n_blades_rotor):>5.0f} N   |")
            print(f"(sum_total for {n_blades_rotor} blades)")

            print(f"================== End ==================\n")

        elif cascade_type == "stator":

            print(f"\n============ Cascade ({cascade_type}) : Results ============")

            def fmt_line(label, U, V, W, alpha):
                # label left-aligned in 10 chars; numbers right-aligned in fixed widths
                return (f"{label:<4} | "
                        f"U = {U:>6.1f} m/s | "
                        f"V = {V:>6.1f} m/s | "
                        f"W = {W:>6.1f} m/s | "
                        f"alpha = {alpha:>6.2f} deg")

            # indices
            i_root, i_tip = 0, -1

            print("(blade root)")
            print(fmt_line("in" , U_arr[i_root, stage]  , V_arr[i_root, stage]  , W_arr[i_root, stage]  , alpha_arr[i_root, stage]))
            print(fmt_line("out", U_arr[i_root, stage+1], V_arr[i_root, stage+1], W_arr[i_root, stage+1], alpha_arr[i_root, stage+1]))
            print()
            print("(blade tip)")
            print(fmt_line("in" , U_arr[i_tip, stage]  , V_arr[i_tip, stage]  , W_arr[i_tip, stage]  , alpha_arr[i_tip, stage]))
            print(fmt_line("out", U_arr[i_tip, stage+1], V_arr[i_tip, stage+1], W_arr[i_tip, stage+1], alpha_arr[i_tip, stage+1]))

            print(f"================== End ==================\n")

    # <<< End of current cascade stage >>>


"""
Plot (results)
"""
plot_results = True
if plot_results:
    plt.figure(figsize=(8,6))

    # plt.plot(r_c_arr, results_fan["Lift"][:,0], 'b-', linewidth=1.5)
    plt.plot(r_c_arr, alpha_arr[:,1], 'b-', linewidth=1.5)

    plt.xlim(r_c_arr[0], r_c_arr[-1])

    plt.xlabel('Radial Stations (m)')
    plt.ylabel("Lift Coefficient (CL)", fontsize=14)
    plt.title(f"Lift Coefficient across blade span", fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)

    # --- create a folder if it doesn't exist ---
    out_dir = "figures/Stage_1"
    os.makedirs(out_dir, exist_ok=True)

    # --- save to file ---
    filename = f"CL.png"
    filepath = os.path.join(out_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight') # high-res, cropped

    plt.close()


""" ===================================================================================================================================================================== """


