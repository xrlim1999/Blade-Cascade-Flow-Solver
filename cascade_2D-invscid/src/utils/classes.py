# FILE : src/utils/classes.py

import numpy as np

# import function
from .geometry import load_airfoil_xy, rearrange_airfoil, resample_airfoil_cosine_finite_TE, rotate_airfoil_about_te
from .panel import data_preparation
from .rhs import right_hand_side_3D
from .solver import solve_cascade_iterative
from .performance import bladerow_performance, display_cascade_performance


class AirfoilSection:
    """
    Represents a single 2D section of a blade (geometry + panel prep).
    """
    def __init__(self, filename: str, m: int, r_m: float, beta_deg: float, t_ratio: float):

        # Load & resample geometry
        x, y = load_airfoil_xy(filename)
        xdata, ydata, *_ , chord, _ = rearrange_airfoil(x, y)
        xnew, ynew = resample_airfoil_cosine_finite_TE(xdata, ydata, n_points=m)
        xnew, ynew = rotate_airfoil_about_te(xnew, ynew, beta_deg, pivot="mid")

        # Panel preperation
        geom = data_preparation(xnew, ynew, len(xnew))

        # Store raw geometry
        geom["x"], geom["y"] = xnew, ynew         # aerfoil nodal points
        geom["r"]        = r_m                    # current blade radial station [m]
        geom["n_panels"] = len(geom["xmid"]) # number of panels per aerfoil cross-section
        geom["chord"]    = chord                  # chord length [m]
        geom["beta"]     = np.deg2rad(beta_deg)   # blade tilt angle [deg]
        geom["pitch"]    = chord * t_ratio        # blade pitch

        self.geom = geom

class FlowSection:
    """
    Represents the flowfield parameters the blade is subjected to.
    """
    def __init__(self, alpha_in: float, U: float, V: float, Omega: float, rho: float):

        # store flow datas
        self.flow = dict()
        self.flow["U"]       = float(U)        # blade inlet U              [m/s]
        self.flow["V"]       = float(V)        # blade inlet V              [m/s]
        self.flow["Omega"]   = float(Omega)    # blade row rotational speed [rad/s]
        self.flow["alpha_1"] = float(np.deg2rad(alpha_in)) # inlet flow angle           [deg]
        self.flow["rho"]     = float(rho)      # inlet flow air density     [kg/m^3]]

class CascadeSolver:
    """
    Solves unit vorticity problems and runs full cascade calculations.
    """
    def __init__(self, Airfoil_section: AirfoilSection, Flow_section: FlowSection):

        self.airfoil_section = Airfoil_section
        self.flow_section    = Flow_section

        # --- Build RHS ---
        self.rhs = right_hand_side_3D(self.airfoil_section.geom, self.flow_section.flow)
    
    def solve(self, A_modes, plot_cp, plot_save):

        # --- define Fourier Modes ---
        A_modes = A_modes

        #  --- iteratively solve for bound vorticity ---
        gamma_b, beta_wake, self.beta_wake_history = solve_cascade_iterative(self.airfoil_section.geom, self.flow_section.flow, A_modes, self.rhs,
                                                                        max_iter=20, relax=0.5, tol_deg=1e-2)
        
        # --- compute performance of current fan stage ---
        self.results, self.blade = bladerow_performance(self.airfoil_section.geom, self.flow_section.flow, gamma_b, 
                                                        plot_cp=plot_cp, plot_save=plot_save)

        return self.results, self.blade
    
    def print_results(self):
        # Display Results
        display_cascade_performance(self.results, self.blade)

