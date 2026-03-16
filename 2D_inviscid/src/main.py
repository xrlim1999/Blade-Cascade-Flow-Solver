# FILE: src/main_airfoil.py

import ast, re

from .utils import solve_airfoil

# --- import parameters for flow plotting ---
with open("plotparams_flow.txt", "r") as f:
    flowplot_params = ast.literal_eval(f.read())

with open("operating_conditions.txt", "r") as f:
    operating_conditions = ast.literal_eval(re.sub(r"#[^\n]*", "",f.read()))

if __name__ == "__main__":

    results, airfoil, flowfield = solve_airfoil(
        # geometry / discretization
        airfoil_tilt_deg = 0.0, # (+ve --> upwards rotation of LE)
        m                = 100 ,
        airfoil_path     = "data/NACA0012.txt",

        # flow operating condition
        operating_conditions = operating_conditions,

        # print conditions
        plot_cp = True,

        # visualisation conditions
        plot_airfoil = True, airfoil_save=True,
        plot_flow    = False, track_particle=True, track_velocity=True, flow_save=True, flowplot_params=flowplot_params
    )
