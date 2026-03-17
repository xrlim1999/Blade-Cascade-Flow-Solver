# FILE: src/main_airfoil.py

import ast, re

from .utils import solve_airfoil

# --- import parameters for flow plotting ---
with open("inputs/operating_conditions.txt", "r") as f:
    operating_conditions = ast.literal_eval(re.sub(r"#[^\n]*", "",f.read()))

with open("inputs/airfoil_settings.txt", "r") as f:
    airfoil_settings = ast.literal_eval(re.sub(r"#[^\n]*", "",f.read()))
airfoil_settings["airfoil_name"] = "data/" + airfoil_settings["airfoil_name"] + ".txt"

with open("inputs/flow_visual_settings.txt", "r") as f:
    flowplot_params = ast.literal_eval(re.sub(r"#[^\n]*", "",f.read()))

with open("inputs/plot_settings.txt", "r") as f:
    plot_settings = ast.literal_eval(re.sub(r"#[^\n]*", "",f.read()))

if __name__ == "__main__":

    results, airfoil, flowfield = solve_airfoil(
        # geometry / discretization
        airfoil_settings = airfoil_settings,

        # flow operating condition
        operating_conditions = operating_conditions,

        # visualisation conditions
        flowplot_params = flowplot_params,

        # plot conditions
        plot_settings = plot_settings,
    )
