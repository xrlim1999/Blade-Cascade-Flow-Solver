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
    solve_blade_row
)

# Results initialisation
from .results_create_save import (
    create_results,
    create_blade,
    create_flow,
    save_results_section
)

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
    coupling_coefficient_bound,
    coupling_coefficient_trailing
)

# RHS builders
from .rhs import (
    right_hand_side_3D
)

# 3D flow solver
from .solver import (
    solve_cascade_iterative
)

# Performance analysis
from .performance import (
    bladerow_performance,
    display_cascade_performance
)

# High-level classes
from .classes import (
    AirfoilSection,
    FlowSection,
    CascadeSolver,
)

__all__ = [
    # --- solver runner ---
    "solve_blade_row",
    # --- results initialisation ---
    "create_results", "create_blade", "create_flow", "save_results_section",
    # --- geometry ---
    "load_airfoil_xy", "rearrange_airfoil", "resample_airfoil_cosine_finite_TE", "rotate_airfoil_about_te",
    # --- panel ---
    "data_preparation",
    # --- kernels ---
    "coupling_coefficient_bound", "coupling_coefficient_trailing",
    # --- rhs ---
    "right_hand_side_3D",
    # --- solver ---
    "solve_cascade_iterative",
    # performance
    "bladerow_performance", "display_cascade_performance",
    # classes
    "AirfoilSection", "FlowSection", "CascadeSolver",
]
