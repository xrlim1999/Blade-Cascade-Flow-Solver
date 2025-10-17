# FILE: src/main.py

import numpy as np

import matplotlib
from matplotlib.path import Path
matplotlib.use('Qt5Agg')  # or 'Qt6Agg' if you have PyQt6
import matplotlib.pyplot as plt
import os

from .utils import AirfoilSection, FlowSection, CascadeSolver, create_results, create_blade, create_flow, save_results_section

"""
Blade Profile
"""
# --- fan cascade sections ---
n_stages = 1
n_sections = 20

# --- fan radius ---
r_hub = 0.25 # [m]
r_tip = 0.68 # [m]

# --- number of blades ---
n_blades_arr = np.array([12])

# --- Blade Stagger angle ---
blade_tilt_root = np.array([+20.0]) # blade tilt at root [deg]
blade_tilt_tip  = np.array([+50.0]) # blade tilt at tip  [deg]

# --- Blade Chord ---
chord_root = np.array([0.6]) * r_tip # blade tilt at root [deg]
chord_tip  = np.array([0.3]) * r_tip # blade tilt at tip  [deg]

# --- RPM ---
RPM_arr = np.array([8000])

# --- number of panels (per element) ---
m = 100

# discretise into radial elements
blade_profile = create_blade(n_sections, n_stages, r_hub, r_tip, n_blades_arr,
                             blade_tilt_root, blade_tilt_tip, chord_root, chord_tip,
                             RPM_arr, m)

"""
Flow Conditions (Upstream of entire fan)
"""
rho    = 1.225  # static density    [kg/m^2]
W0     = 100.0  # absolute velocity [m/s]
alpha0 = 0.0    # freestream angle  [deg]

flow = create_flow(n_sections, n_stages, rho, W0, alpha0)

"""
Results initialisation
"""
results_fan, results_cascade = create_results(n_sections, n_stages)

"""
Solve
"""
# --- Fourier modes ---
A_modes = np.array([1.0, 0.25, 0.1])

# --- upacking (these are same for all stages for now) ---
r_element_arr    = blade_profile["r_element"] 
dr_element_arr   = blade_profile["dr_element"]

for stage in range(n_stages):

    # --- unpack (blade profile) ---
    n_blades         = blade_profile["n_blades"][stage]
    blade_tilt_deg   = blade_profile["blade tilt"][:,stage]
    chord            = blade_profile["chord"][:,stage]
    pitch_ratio      = blade_profile["pitch ratio"][:,stage]
    Omega            = blade_profile["Omega"][stage]

    # --- unpack (flow) ---
    rho     = flow["rho"][stage]
    alpha_1 = flow["alpha"][:,stage]
    W_1     = flow["W"][:,stage]
    U_1     = flow["U"][:,stage]
    V_1     = flow["V"][:,stage]

    for k in range(n_sections):

        # --- Initialise Classes : Geometry + Flow + Solver ---
        airfoil_section = AirfoilSection("src/data/aerofoil_coords.txt", m, r_element_arr[k], blade_tilt_deg[k], pitch_ratio[k])
        flow_section    = FlowSection(alpha_1[k], U_1[k], V_1[k], Omega, rho)
        solver          = CascadeSolver(airfoil_section, flow_section)

        # --- Solve ---
        results, blade = solver.solve(A_modes, plot_cp=False, plot_save=True)

        # --- Save Results ---
        save_results_section(k, stage, results, flow, results_fan,
                             blade_tilt_deg)

    # <<< End of current blade section >>>

    """ Analysis (cascade layer) """
    # if cascade_type == "rotor":
    results_cascade["Torque"][:,stage] = -results_fan    ["Fy"]    [:,stage] * r_element_arr * dr_element_arr # torque = Fy[tangential force] x r[distance to rotor center] x dr[blade section length]
    results_cascade["Power"] [:,stage] = results_cascade["Torque"][:,stage] * Omega                          # power  = torque x r[distance to rotor center]
    results_cascade["Drag"]  [:,stage] = results_fan    ["Fx"]    [:,stage] * dr_element_arr                 # drag   = Fx[axial force] x dr[blade section length]

    # --- Print results (stage) ---
    stage_results = True
    if stage_results:
        print(f"\n============ Stage {stage+1} : Results ============")

        if Omega == 0.0:
            print("\nStage type : Stator\n")
        else:
            print("\nStage type : Rotor\n")
        
        print(f"{'Power':<8}: {np.sum(results_cascade['Power'][:,stage]/1000*n_blades):>5.2f} kW  |")
        print(f"{'Torque':<8}: {np.sum(results_cascade['Torque'][:,stage]/1000*n_blades):>5.2f} kNm |")
        print(f"{'Drag':<8}: {np.sum(results_cascade['Drag'][:,stage]*n_blades):>5.0f} N   |")
        print(f"(sumtotal for {n_blades} blades)")

        def fmt_line(label, U, V, W, alpha):
            # label left-aligned in 10 chars; numbers right-aligned in fixed widths
            return (f"{label:<4} | "
                    f"U = {U:>6.1f} m/s | "
                    f"V = {V:>6.1f} m/s | "
                    f"W = {W:>6.1f} m/s | "
                    f"alpha = {alpha:>6.2f} deg")

        # indices
        i_root, i_tip = 0, -1

        print("\n(blade root)")
        print(fmt_line("in" , flow['U'][i_root, stage]  , flow['V'][i_root, stage]  , flow['W'][i_root, stage]  , flow['alpha'][i_root, stage]))
        print(fmt_line("out", flow['U'][i_root, stage+1], flow['V'][i_root, stage+1], flow['W'][i_root, stage+1], flow['alpha'][i_root, stage+1]))
        print()
        print("(blade tip)")
        print(fmt_line("in" , flow['U'][i_tip, stage]  , flow['V'][i_tip, stage]  , flow['W'][i_tip, stage]  , flow['alpha'][i_tip, stage]))
        print(fmt_line("out", flow['U'][i_tip, stage+1], flow['V'][i_tip, stage+1], flow['W'][i_tip, stage+1], flow['alpha'][i_tip, stage+1]))

        print(f"================== End ==================\n")

# <<< End of current cascade stage >>>


"""
Plot (results)
"""
plot_results = True
if plot_results:
    plt.figure(figsize=(8,6))

    # plt.plot(r_c_arr, results_fan["Lift"][:,0], 'b-', linewidth=1.5)
    plt.plot(r_element_arr, results_fan["Effective angle-of-attack"][:,0], 'b-', linewidth=1.5)

    plt.xlim(r_element_arr[0], r_element_arr[-1])

    plt.xlabel('Radial Stations (m)')
    plt.ylabel("Effective AoA (relative)", fontsize=14)
    plt.title(f"Effective AoA (relative) across blade span", fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)

    # --- create a folder if it doesn't exist ---
    out_dir = "figures/Stage_1"
    os.makedirs(out_dir, exist_ok=True)

    # --- save to file ---
    filename = f"aoa.png"
    filepath = os.path.join(out_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight') # high-res, cropped

    plt.close()


""" ================== End Of Code =============================================================================================================================================== """


