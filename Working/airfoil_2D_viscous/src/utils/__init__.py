"""
utils package initializer

This file makes `utils` a Python package and re-exports the most commonly used
functions and classes so they can be imported directly with:

    from utils import AirfoilSection, CascadeSolver, airfoil_performance_in_cascade, ...

If you want something more specific, you can still do:
    from utils.geometry import load_airfoil_xy
"""

# Solver Runner
from .runner import(
    solve_airfoil
)

# # Results initialisation
# from .results_create_save import (
#     create_results,
#     create_blade,
#     create_flow,
#     save_results_section
# )

# Geometry tools
from .geometry import (
    load_airfoil_xy,
    rearrange_airfoil,
    resample_airfoil_cosine_finite_TE,
    rotate_airfoil_about_te
)

# Panel preparation
from .panel import (
    data_preparation
)

# Influence kernels
from .martensen_kernels import (
    coupling_coefficient_bound
)

# RHS builders
from .rhs import (
    right_hand_side_airfoil
)

# 3D flow solver
from .solver_inviscid import (
    vorticity_solution_kutta
)

from .solver_viscous import (
    starting_assumption,
    momentumthickness_integral
)

# Performance analysis
from .performance import (
    airfoil_performance
)

# Visualisation
from .visualisation import(
    airfoil_visualisation,
    flow_visualisation
)

# High-level classes
from .classes import (
    AirfoilSection,
    FlowSection,
    AirfoilSolver,
)

__all__ = [
    # --- solver runner ---
    "solve_airfoil",
    # # --- results initialisation ---
    # "create_results", "create_blade", "create_flow", "save_results_section",
    # --- geometry ---
    "load_airfoil_xy", "rearrange_airfoil", "resample_airfoil_cosine_finite_TE", "rotate_airfoil_about_te",
    # --- panel ---
    "data_preparation",
    # --- kernels ---
    "coupling_coefficient_bound",
    # --- rhs ---
    "right_hand_side_airfoil",
    # --- inviscid solver ---
    "vorticity_solution_kutta",
    # --- viscous solver ---
    "starting_assumption", "momentumthickness_integral",
    # performance
    "airfoil_performance",
    # visualisation
    "airfoil_visualisation", "flow_visualisation",
    # classes
    "AirfoilSection", "FlowSection", "AirfoilSolver",
]
