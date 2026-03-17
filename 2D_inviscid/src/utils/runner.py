import numpy as np

from .classes import AirfoilSection, FlowSection, AirfoilSolver
# from .results_create_save import create_results, create_blade, create_flow, save_results_section


def solve_airfoil(
    *,
    # geometry / discretization
    airfoil_settings: dict,

    # flow operating condition
    operating_conditions: dict,

    # visualisation conditions
    flowplot_params: dict,

    # plot conditions
    plot_settings: dict
):
    """
    Single-API call pipeline for a single blade row.
    Returns: results_fan, results_cascade, flow, blade_profile
    """
    # --- airfoil and flow (initialisation) ---
    airfoil_section = AirfoilSection(airfoil_settings["airfoil_name"], 
                                     airfoil_settings["num_sample_points"], 
                                     airfoil_settings["tilt_angle_deg"])
                                     
    flow_section = FlowSection(operating_conditions["alpha0_deg"], 
                               operating_conditions["W0"], 
                               operating_conditions["rho"], 
                               operating_conditions["dynvisc"])

    # --- solver (initialisation) ---
    solver = AirfoilSolver(airfoil_section, flow_section)

    # --- solver ---
    solver.solve(plot_cp=plot_settings["plot_cp"], plot_save_cp=plot_settings["save_plot_cp"])

    # --- print results ---
    solver.print_results()

    # --- visualisation ---
    if plot_settings["plot_airfoil"]:
        solver.plot_airfoil(plot_save=plot_settings["save_plot_airfoil"])

    if plot_settings["plot_flow"]:
        flowfield = solver.plot_flow(track_velocity=plot_settings["track_velocity"], track_particle=plot_settings["track_particle"], 
                                                    flowplot_params=flowplot_params, plot_save=plot_settings["save_plot_flow"])
    else:
        flowfield = None
    
    print("*** Flow solver successfully finished ***\n")

    return solver.results, solver.airfoil, flowfield
