import numpy as np


def create_blade(n_sections: int, r_hub: float, r_tip: float, n_blades: int,
                 blade_tilt_root:float, blade_tilt_tip: float, chord_root: float, chord_tip: float,
                 RPM: float, m: int):
    # --- radial positions ---
    r_stations = np.linspace(r_hub, r_tip, num=(n_sections+1))

    r1_arr = r_stations[:-1]
    r2_arr = r_stations[1:]

    # aerofoil surface [assumed to be midpoint of start and end radial position of section]
    r_m_arr = (r1_arr + r2_arr) / 2 

    # distance of radial element
    dr_arr = r2_arr - r1_arr

    # --- blade tilt and chord ---
    blade_tilt_arr = np.linspace(blade_tilt_root, blade_tilt_tip, n_sections)[:, None]
    chord_arr      = np.linspace(chord_root, chord_tip, n_sections)[:, None]

    # --- pitch-to-chord ratio
    t_over_c = (((2*np.pi) * r_m_arr) / float(n_blades)) / chord_arr[:, 0]
    t_ratio_arr = t_over_c[:, None]

    # --- blade rotation speed ---
    Omega = (RPM/60) * (2*np.pi)

    # --- create blade (containter) ---
    blade_profile = {
        "r_element"   : r_m_arr,        # array : float
        "dr_element"  : dr_arr,         # array : float
        "n_blades"    : n_blades,       # scalar: int
        "blade tilt"  : blade_tilt_arr, # array : float
        "chord"       : chord_arr,      # array : float
        "pitch ratio" : t_ratio_arr,    # array : float
        "RPM"         : RPM,            # scalar: float
        "Omega"       : Omega,          # scalar: float
        "n_panels"    : m               # scalar: int
    }

    return blade_profile


def create_flow(n_sections: int, rho: float, W: float, alpha: float):

    flow = dict()

    alpha_rad = np.deg2rad(alpha)
    U, V = W*np.cos(alpha_rad), W*np.sin(alpha_rad) # velocity in axial, tangential directions [m/s]

    # --- store in arrays ---
    rho_arr   = np.zeros(2)
    W_arr     = np.zeros((n_sections, 2))
    U_arr     = np.zeros((n_sections, 2))
    V_arr     = np.zeros((n_sections, 2))
    alpha_arr = np.zeros((n_sections, 2))

    rho_arr  [0]   = rho
    W_arr    [:,0] = W
    alpha_arr[:,0] = alpha
    U_arr    [:,0] = U
    V_arr    [:,0] = V

    flow = {
        "rho"   : rho_arr,
        "alpha" : alpha_arr,
        "W"     : W_arr,
        "U"     : U_arr,
        "V"     : V_arr,
    }

    return flow


def create_results(n_sections: int):
    """
    Build and return (results_fan, results_cascade) dicts with the same keys
    and shapes your current code expects.
    """
    # --- Blade Section Results ---
    results_fan = {
        "Effective angle-of-attack": np.zeros((n_sections, 1)),
        "CL"                       : np.zeros((n_sections, 1)),
        "Lift"                     : np.zeros((n_sections, 1)),
        "Circulation"              : np.zeros((n_sections, 1)),
        "Fy"                       : np.zeros((n_sections, 1)),
        "Fx"                       : np.zeros((n_sections, 1)),
    }

    # --- Cascade Layer Results ---
    results_cascade = {
        "Torque": np.zeros((n_sections, 1)),
        "Power" : np.zeros((n_sections, 1)),
        "Drag"  : np.zeros((n_sections, 1)),
    }

    return results_fan, results_cascade


def save_results_section(k, results, flow, results_fan,
                         blade_tilt_deg):

    # --- Store Output Values (for next cascade layer) ---
    flow["U"][k, 1]     = results["U_out"]
    flow["V"][k, 1]     = results["V_out"]
    flow["W"][k, 1]     = results["W_out"]
    flow["alpha"][k, 1] = results["alpha_out"]

    # --- Save Results ---
    results_fan["Effective angle-of-attack"][k, 0] = results["alpha_in_rel"] + blade_tilt_deg[k] # [ deg ]
    results_fan["CL"]                       [k, 0] = results["CL"]                  # [ - ]
    results_fan["Lift"]                     [k, 0] = results["Lift"]                # [ N ]
    results_fan["Circulation"]              [k, 0] = results["circulation"]         # [ m^2/s ]
    results_fan["Fy"]                       [k, 0] = results["lift_tan"]            # [ N ]
    results_fan["Fx"]                       [k, 0] = results["lift_axial"]          # [ N ]

    return flow, results_fan

