"""
utils package initializer

This file makes `utils` a Python package and re-exports the most commonly used
functions and classes so they can be imported directly with:

    from utils import AirfoilSection, CascadeSolver, airfoil_performance_in_cascade, ...

If you want something more specific, you can still do:
    from utils.geometry import load_airfoil_xy
"""

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
    coupling_coefficient_cascade
)

# RHS builders
from .rhs import (
    right_hand_side_rotor_unit,
    right_hand_side_stator_unit
)

# Linear solver
from .solver import (
    vorticity_solution_kutta_rotor,
    vorticity_solution_kutta_stator
)

# Performance analysis
from .performance import (
    airfoil_performance_in_cascade_rotor,
    airfoil_performance_in_cascade_stator,
    display_cascade_performance
)

# High-level classes
from .classes import (
    AirfoilSection,
    CascadeSolver,
    # RotorBlade, StatorBlade   # (when you add them later)
)

__all__ = [
    # --- geometry ---
    "load_airfoil_xy", "rearrange_airfoil", "resample_airfoil_cosine_finite_TE", "rotate_airfoil_about_te",
    # --- panel ---
    "data_preparation",
    # --- kernels ---
    "coupling_coefficient_cascade",
    # --- rhs ---
    "right_hand_side_rotor_unit", "right_hand_side_stator_unit",
    # --- solver ---
    "vorticity_solution_kutta_rotor", "vorticity_solution_kutta_stator",
    # performance
    "airfoil_performance_in_cascade_rotor", "airfoil_performance_in_cascade_stator", "display_cascade_performance",
    # classes
    "AirfoilSection", "CascadeSolver",
]
