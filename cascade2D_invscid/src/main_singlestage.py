# FILE: src/main_singlestage.py

import numpy as np

from .utils import solve_blade_row


if __name__ == "__main__":

    results_fan, results_cascade, flow, blade_profile = solve_blade_row(
        # geometry / discretization
        n_sections=20,
        r_hub=0.25, r_tip=0.68,
        n_blades=12,
        blade_tilt_root_deg=+20.0, blade_tilt_tip_deg=+50.0,
        chord_root=0.6*0.68, chord_tip=0.3*0.68,
        m=100,

        # flow operating condition
        RPM=8000,
        rho=1.225, 
        W0=100.0, alpha0_deg=0.0,
        
        # solver options
        A_modes=np.array([1.0, 0.0, 0.0]),
        airfoil_path="src/data/aerofoil_coords.txt",
        print_stage=True,
    )
