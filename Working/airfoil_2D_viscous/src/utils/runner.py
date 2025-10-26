import numpy as np

from .classes import AirfoilSection, FlowSection, AirfoilSolver
# from .results_create_save import create_results, create_blade, create_flow, save_results_section


def solve_airfoil(
    *,
    # geometry / discretization
    airfoil_tilt_deg: float, # (+ve --> upwards rotation of LE)
    m: int = 100,
    airfoil_path: str,

    # flow operating condition
    rho: float,
    W0: float,
    alpha0_deg: float,
    dynvisc: float,

    # print conditions
    plot_cp: bool,

    # visualisation conditions
    plot_airfoil: bool, airfoil_save: bool,
    plot_flow: bool, track_velocity: bool, track_particle:  bool, flow_save: bool, flowplot_params: dict
):
    
    """
    Single-API call pipeline for a single blade row.
    Returns: results_fan, results_cascade, flow, blade_profile
    """
    # --- build blade & flow containers ---
    

    # --- airfoil and flow (initialisation) ---
    airfoil_section = AirfoilSection(airfoil_path, m, airfoil_tilt_deg)
    flow_section    = FlowSection(alpha0_deg, W0, rho)

    # --- solver (initialisation) ---
    solver = AirfoilSolver(airfoil_section, flow_section)

    # --- solver ---
    solver.solve_inviscid(plot_cp=plot_cp)

    # --- print results ---
    solver.print_results()

    # --- visualisation ---
    if plot_airfoil:
        solver.plot_airfoil(plot_save=airfoil_save)

    if plot_flow:
        flowfield = solver.plot_flow(track_velocity=track_velocity, track_particle=track_particle, 
                                                    flowplot_params=flowplot_params, plot_save=flow_save)
    else:
        flowfield = None
    
    print("--- Flow solver successfully finished ---\n")

    return solver.results, solver.airfoil, flowfield


