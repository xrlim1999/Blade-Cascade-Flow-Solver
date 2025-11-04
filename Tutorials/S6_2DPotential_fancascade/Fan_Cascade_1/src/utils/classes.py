# FILE : src/utils/classes.py

import numpy as np

# import function
from .geometry import load_airfoil_xy, rearrange_airfoil, resample_airfoil_cosine_finite_TE, rotate_airfoil_about_te
from .panel import data_preparation
from .martensen_kernels import coupling_coefficient_cascade
from .rhs import right_hand_side_rotor_unit, right_hand_side_stator_unit
from .solver import vorticity_solution_kutta_rotor, vorticity_solution_kutta_stator
from .performance import airfoil_performance_in_cascade_rotor, airfoil_performance_in_cascade_stator, display_cascade_performance


class AirfoilSection:
    """
    Represents a single 2D section of a blade (geometry + panel prep).
    """
    def __init__(self, filename: str, m: int, beta_deg: float, t_ratio: float = 1.0):
        # Load & resample geometry
        x, y = load_airfoil_xy(filename)
        xdata, ydata, *_ , chord, _ = rearrange_airfoil(x, y)
        xnew, ynew = resample_airfoil_cosine_finite_TE(xdata, ydata, n_points=m)
        xnew, ynew = rotate_airfoil_about_te(xnew, ynew, beta_deg, pivot="mid")

        # Store raw geometry
        self.x, self.y = xnew, ynew
        self.chord = chord
        self.beta = np.deg2rad(beta_deg)
        self.pitch = chord * t_ratio

        # Panel prep
        ds, sine, cosine, slope, xmid, ymid = data_preparation(xnew, ynew, len(xnew))
        self.ds, self.sine, self.cosine, self.slope = ds, sine, cosine, slope
        self.xmid, self.ymid = xmid, ymid
        self.n_panels = len(xmid)

    def build_coupling(self):
        return coupling_coefficient_cascade(
            self.pitch, self.ds, self.sine, self.cosine,
            self.slope, self.xmid, self.ymid, self.n_panels
        )


class CascadeSolver:
    """
    Solves unit vorticity problems and runs full cascade calculations.
    """
    def __init__(self, section: AirfoilSection, cascade_type: str, r_c: float, r_m: float, r1: float, r2:float):

        self.cascade_type = cascade_type.lower()
        if self.cascade_type != "rotor" and self.cascade_type != "stator":
            raise ValueError("variable < cascade_type > must be either <rotor> or <stator>.\n")

        self.section = section
        self.r1, self.r2 = r1, r2
        self.AVR = 1.0

        # Coupling matrix
        self.coup = section.build_coupling()
        
        if self.cascade_type == "rotor":
            # RHS for unit problems
            self.rhs_U, self.rhs_V, self.rhs_Omega = right_hand_side_rotor_unit(
                section.n_panels, section.xmid, section.ymid,
                section.ds, section.slope, section.pitch,
                r_c, r1, r2, r_m, self.AVR
            )
            # Unit vorticity solutions
            self.vort_U, self.vort_V, self.vort_Omega = vorticity_solution_kutta_rotor(
                self.coup, self.rhs_U, self.rhs_V, self.rhs_Omega, section.ds, mode="circulation"
            )
            
        elif self.cascade_type == "stator":
            # RHS for unit problems
            self.rhs_U, self.rhs_V = right_hand_side_stator_unit(
                section.n_panels, section.slope
            )
            # Unit vorticity solutions
            self.vort_U, self.vort_V = vorticity_solution_kutta_stator(
                self.coup, self.rhs_U, self.rhs_V, section.ds, mode="circulation"
            )

    # Solve for vorticity
    def solve(self, rho, alpha, U, V, Omega, plot_cp=False):
        """
        Run the full cascade for given inflow (U,V,alpha) and, for rotors, Ω.
        Returns (results, blade) like your performance functions.
        """
        if self.cascade_type == "rotor":
            results, blade = airfoil_performance_in_cascade_rotor(
                rho, alpha, self.section.beta, self.section.chord, self.section.pitch,
                self.section.ds, self.section.xmid, self.section.ymid,
                self.section.sine, self.section.cosine, self.section.slope,
                self.r1, self.r2, U, V, Omega,
                self.vort_U, self.vort_V, self.vort_Omega,
                plot_cp=plot_cp
            )

        elif self.cascade_type == "stator":
            results, blade = airfoil_performance_in_cascade_stator(
                rho, alpha, self.section.beta, self.section.chord, self.section.pitch,
                self.section.ds, self.section.xmid, self.section.ymid,
                self.section.sine, self.section.cosine, self.section.slope,
                U, V, self.vort_U, self.vort_V,
                plot_cp=plot_cp
            )

        blade["airfoil_coords"] = np.column_stack((self.section.x, self.section.y))

        return results, blade
    
    # Display Results
    def print_results(self, results, blade):
        display_cascade_performance(results, blade)
