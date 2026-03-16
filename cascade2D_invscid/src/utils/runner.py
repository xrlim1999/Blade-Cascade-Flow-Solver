import numpy as np

from .classes import AirfoilSection, FlowSection, CascadeSolver
from .results_create_save import create_results, create_blade, create_flow, save_results_section


def solve_blade_row(
    *,
    # geometry / discretization
    n_sections: int,
    r_hub: float, r_tip: float,
    n_blades: int,
    blade_tilt_root_deg: float, blade_tilt_tip_deg: float,
    chord_root: float, chord_tip: float,
    m: int = 100,
    airfoil_path: str = "src/data/aerofoil_coords.txt",

    # flow operating condition
    RPM: float,
    rho: float,
    W0: float,
    alpha0_deg: float,

    # solver options
    A_modes = np.array([1.0, 0.0, 0.0]),
    print_stage: bool = True,
    stage_num: int = 1,
):
    
    """
    Single-API call pipeline for a single blade row.
    Returns: results_fan, results_cascade, flow, blade_profile
    """
    # --- build blade & flow containers ---
    blade_profile = create_blade(
        n_sections, r_hub, r_tip, n_blades,
        blade_tilt_root_deg, blade_tilt_tip_deg, chord_root, chord_tip,
        RPM, m
    )
    flow = create_flow(n_sections, rho, W0, alpha0_deg)
    results_fan, results_cascade = create_results(n_sections)

    # --- unpack per-stage constants ---
    r_element  = blade_profile["r_element"]
    dr_element = blade_profile["dr_element"]

    # --- stage loop (single stage) ---
    n_blades        = blade_profile["n_blades"]
    blade_tilt_deg  = blade_profile["blade tilt"]
    chord           = blade_profile["chord"]
    pitch_ratio     = blade_profile["pitch ratio"]
    Omega           = blade_profile["Omega"]

    # --- flow in (initial value) ---
    rho_stage = flow["rho"][0]
    alpha_1   = flow["alpha"][:, 0]  # degrees
    U_1       = flow["U"][:, 0]
    V_1       = flow["V"][:, 0]

    # --- flow analysis along radial direction ---
    for k in range(n_sections):
        airfoil_section = AirfoilSection(
            airfoil_path, m,
            r_element[k], blade_tilt_deg[k], pitch_ratio[k]
        )
        flow_section = FlowSection(alpha_1[k], U_1[k], V_1[k], Omega, rho_stage)
        solver = CascadeSolver(airfoil_section, flow_section)

        results, blade = solver.solve(A_modes, plot_cp=False, plot_save=False)

        # persist outputs & effective AoA (relative)
        save_results_section(
            k, results, flow, results_fan,
            blade_tilt_deg
        )

    # --- blade row (stage) analysis ---
    results_cascade["Drag"]  [:, 0] =  results_fan["Fx"][:, 0] * dr_element
    results_cascade["Torque"][:, 0] = -results_fan["Fy"][:, 0] * dr_element * r_element
    results_cascade["Power"] [:, 0] =  results_cascade["Torque"][:, 0] * Omega

    if print_stage:

        print(f"\n============ Stage {stage_num} : Results ============\n")

        print(f"Stage type : Rotor at {RPM} rpm\n" if Omega != 0.0 else "Stage type : Stator\n")

        PkW = np.sum(results_cascade['Power'][:, 0] / 1000 * n_blades)
        TkNm = np.sum(results_cascade['Torque'][:, 0] / 1000 * n_blades)
        DragN = np.sum(results_cascade['Drag'][:, 0] * n_blades)
        print(f"{'Power':<8}: {PkW:>8.2f} kW  |")
        print(f"{'Torque':<8}: {TkNm:>8.2f} kNm |")
        print(f"{'Drag':<8}: {DragN:>8.0f} N   |")
        print(f"(sumtotal for {n_blades} blades)\n")

        # show root & tip in/out
        def fmt(label, U, V, W, a): return (
            f"{label:<4} | U = {U:>6.1f} m/s | V = {V:>6.1f} m/s | "
            f"W = {W:>6.1f} m/s | alpha = {a:>6.2f} deg"
        )

        i_root, i_tip = 0, -1
        print("(blade root)")
        print(fmt("in" , flow['U'][i_root, 0], flow['V'][i_root, 0], flow['W'][i_root, 0], flow['alpha'][i_root, 0]))
        print(fmt("out", flow['U'][i_root, 1], flow['V'][i_root, 1], flow['W'][i_root, 1], flow['alpha'][i_root, 1]))
        print("\n(blade tip)")
        print(fmt("in" , flow['U'][i_tip, 0], flow['V'][i_tip, 0], flow['W'][i_tip, 0], flow['alpha'][i_tip, 0]))
        print(fmt("out", flow['U'][i_tip, 1], flow['V'][i_tip, 1], flow['W'][i_tip, 1], flow['alpha'][i_tip, 1]))

        print("\n================== End ==================\n")

    return results_fan, results_cascade, flow, blade_profile