# FILE : src/utils/classes.py

import numpy as np

# import function
from .geometry import load_airfoil_xy, rearrange_airfoil, resample_airfoil_cosine_finite_TE, rotate_airfoil_about_te
from .panel import data_preparation
from .rhs import right_hand_side_airfoil
from .solver import vorticity_solution_kutta
from .performance import airfoil_performance, display_airfoil_performance
from .visualisation import airfoil_visualisation, flow_visualisation


class AirfoilSection:
    """
    Represents a single 2D section of a blade (geometry + panel prep).
    """
    def __init__(self, filename: str, m: int, beta_deg: float):

        # convert angles to radians
        beta_rad = np.deg2rad(beta_deg)

        # Load & resample geometry
        x, y = load_airfoil_xy(filename)
        xdata, ydata, *_ , chord, _ = rearrange_airfoil(x, y)
        xnew, ynew = resample_airfoil_cosine_finite_TE(xdata, ydata, n_points=m)
        xnew, ynew = rotate_airfoil_about_te(xnew, ynew, beta_rad, pivot="mid")

        # Panel preperation
        geom = data_preparation(xnew, ynew, len(xnew))

        # Store raw geometry
        geom["x"], geom["y"] = xnew, ynew    # aerfoil nodal points
        geom["n_panels"] = len(geom["xmid"]) # number of panels per aerfoil cross-section
        geom["chord"]    = chord             # chord length [m]
        geom["beta"]     = beta_rad          # blade tilt angle [deg]

        self.geom = geom

class FlowSection:
    """
    Represents the flowfield parameters the blade is subjected to.
    """
    def __init__(self, alpha_in: float, W: float, rho: float):
        
        a    = np.deg2rad(alpha_in)
        U, V = W*np.cos(a), W*np.sin(a)

        # store flow datas
        self.flow = dict()
        self.flow["U"]       = float(U)                    # blade inlet U              [m/s]
        self.flow["V"]       = float(V)                    # blade inlet V              [m/s]
        self.flow["W"]       = float(W)                    # blade inlet absolute speed [m/s]
        self.flow["alpha_1"] = float(a) # inlet flow angle           [deg]
        self.flow["rho"]     = float(rho)                  # inlet flow air density     [kg/m^3]]

class AirfoilSolver:
    """
    Solves unit vorticity problems and runs full cascade calculations.
    """
    def __init__(self, Airfoil_section: AirfoilSection, Flow_section: FlowSection):

        self.geom = Airfoil_section.geom
        self.flow = Flow_section.flow
    
    def solve(self, plot_cp: bool, plot_save=True):

        # --- Build RHS ---
        self.rhs = right_hand_side_airfoil(self.geom, self.flow, delta=None, Ue=None)

        #  --- iteratively solve for bound vorticity ---
        coup, gamma = vorticity_solution_kutta(self.geom, self.flow, self.rhs, mode='gamma')

        # --- compute performance of current fan stage ---
        self.results, self.airfoil = airfoil_performance(self.geom, self.flow, coup, gamma, 
                                                        plot_cp=plot_cp, plot_save=plot_save)

        return self.results, self.airfoil
    
    def print_results(self):
        # Display Results
        display_airfoil_performance(self.results, self.airfoil)

    def plot_airfoil(self, plot_save=True):
        airfoil_visualisation(self.geom, plot_save=plot_save)

    def plot_flow(self, track_particle, track_velocity, flowplot_params, plot_save=True):
        
        flowfield = flow_visualisation(self.geom, self.flow, self.results["vorticity"], 
                                                      flowplot_params,
                                                      track_particle=track_particle, track_velocity=track_velocity,
                                                      plot_save=plot_save)
        
        return flowfield
