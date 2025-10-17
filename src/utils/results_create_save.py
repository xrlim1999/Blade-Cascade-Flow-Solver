import numpy as np


def create_blade(n_sections: int, n_stages: int, r_hub: float, r_tip: float, n_blades_arr:np.ndarray,
                 blade_tilt_root: np.ndarray, blade_tilt_tip: np.ndarray, chord_root: np.ndarray, chord_tip: np.ndarray,
                 RPM_arr: np.ndarray, m: int):
    """ 
    Radial positions
    """
    r_stations = np.linspace(r_hub, r_tip, num=(n_sections+1))

    r1_arr = r_stations[:-1]
    r2_arr = r_stations[1:]

    # aerofoil surface [assumed to be midpoint of start and end radial position of section]
    r_m_arr = (r1_arr + r2_arr) / 2 

    # distance of radial element
    dr_arr = r2_arr - r1_arr

    # --- blade tilt and chord ---
    blade_tilt_arr = np.zeros((n_sections, n_stages))
    chord_arr      = np.zeros((n_sections, n_stages))

    for i in range(n_stages):
        blade_tilt_arr[:,i] = np.linspace(blade_tilt_root[i], blade_tilt_tip[i], n_sections)
        chord_arr     [:,i] = np.linspace(chord_root[i], chord_tip[i], n_sections)

    # --- pitch-to-chord ratio
    t_ratio_arr      = np.zeros((n_sections, n_stages))
    t_ratio_arr = (((2*np.pi) * r_m_arr)[:, None] / n_blades_arr[None, :]) / chord_arr

    # --- blade rotation speed ---
    Omega_arr = (RPM_arr/60) * (2*np.pi)


    blade_profile = {
        "r_element"   : r_m_arr,
        "dr_element"  : dr_arr,
        "n_blades"    : n_blades_arr,
        "blade tilt"  : blade_tilt_arr,
        "chord"       : chord_arr,
        "pitch ratio" : t_ratio_arr,
        "RPM"         : RPM_arr,
        "Omega"       : Omega_arr,
        "n_panels"    : m
    }

    return blade_profile


def create_flow(n_sections: int, n_stages: int, rho: float, W: float, alpha: float):

    flow = dict()

    alpha_rad = np.deg2rad(alpha)
    U, V = W*np.cos(alpha_rad), W*np.sin(alpha_rad) # velocity in axial, tangential directions [m/s]

    # --- store in arrays ---
    rho_arr   = np.zeros(n_stages+1)
    W_arr     = np.zeros((n_sections, n_stages+1))
    U_arr     = np.zeros((n_sections, n_stages+1))
    V_arr     = np.zeros((n_sections, n_stages+1))
    alpha_arr = np.zeros((n_sections, n_stages+1))

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


def create_results(n_sections: int, n_stages: int):
    """
    Build and return (results_fan, results_cascade) dicts with the same keys
    and shapes your current code expects.
    """
    # --- Blade Section Results ---
    results_fan = {
        "Effective angle-of-attack": np.zeros((n_sections, n_stages)),
        "CL"                       : np.zeros((n_sections, n_stages)),
        "Lift"                     : np.zeros((n_sections, n_stages)),
        "Circulation"              : np.zeros((n_sections, n_stages)),
        "Fy"                       : np.zeros((n_sections, n_stages)),
        "Fx"                       : np.zeros((n_sections, n_stages)),
    }

    # --- Cascade Layer Results ---
    results_cascade = {
        "Torque": np.zeros((n_sections, n_stages)),
        "Power" : np.zeros((n_sections, n_stages)),
        "Drag"  : np.zeros((n_sections, n_stages)),
    }

    return results_fan, results_cascade


def save_results_section(k, stage, results, flow, results_fan,
                         blade_tilt_deg):

    # --- Store Output Values (for next cascade layer) ---
    flow["U"][k, stage+1]     = results["U_out"]
    flow["V"][k, stage+1]     = results["V_out"]
    flow["W"][k, stage+1]     = results["W_out"]
    flow["alpha"][k, stage+1] = results["alpha_out"]

    # --- Save Results ---
    results_fan["Effective angle-of-attack"][k, stage] = results["alpha_in_rel"] + blade_tilt_deg[k] # [ deg ]
    results_fan["CL"]                       [k, stage] = results["CL"]                  # [ - ]
    results_fan["Lift"]                     [k, stage] = results["Lift"]                # [ N ]
    results_fan["Circulation"]              [k, stage] = results["circulation"]         # [ m^2/s ]
    results_fan["Fy"]                       [k, stage] = results["lift_tan"]            # [ N ]
    results_fan["Fx"]                       [k, stage] = results["lift_axial"]          # [ N ]

    return flow, results_fan


