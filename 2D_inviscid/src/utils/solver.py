import numpy as np
from .martensen_kernels import coupling_coefficient_bound


def vorticity_solution_kutta(geom: dict, rhs, mode='gamma'):
    """
    Solve the vortex panel linear system with the Kutta condition enforced
    at the trailing edge.

    Builds the coupling matrix from geom, replaces the last equation with
    the Kutta condition (gamma_u + gamma_l = 0), and solves for the panel
    vorticity distribution. A condition number check is performed before
    solving to warn of potential numerical instability.

    Parameters
    ----------
    geom : dict
        Panel geometry dictionary as returned by data_preparation, containing
        panel lengths, angles, midpoints and n_panels.
    rhs : ndarray, shape (m,)
        Right-hand side vector as returned by right_hand_side_airfoil.
    mode : str, optional
        Kutta residual check mode. 'gamma' checks gamma_u + gamma_l directly.
        Any other value checks gamma_u*ds_u + gamma_l*ds_l instead.
        Default is 'gamma'.

    Returns
    -------
    coup : ndarray, shape (m, m)
        Coupling coefficient matrix.
    vorticity : ndarray, shape (m,)
        Panel vorticity distribution [m/s].

    Warns
    -----
    Prints a warning if the system condition number exceeds 1e10.
    Prints a warning if the Kutta residual exceeds 1e-3.
    """
    coup = coupling_coefficient_bound(geom)
    
    A = coup.copy()
    b = rhs.copy()

    i_u, i_l = 0, (A.shape[0]-1)

    # -------------------------------------------------------
    # Enforce Kutta Condition
    #   - replace last equation with: gamma_u + gamma_l = 0
    #   - discards the last flow-tangency equation
    # -------------------------------------------------------
    row = np.zeros_like(b)
    row[i_u] = 1.0
    row[i_l] = 1.0
    A[-1, :] = row
    b[-1]    = 0.0

    # -------------------------------------------------------
    # Condition Number Check
    #   - warns if system is ill-conditioned before solving
    #   - cond > 1e10 : risks losing ~6 digits of precision
    # -------------------------------------------------------
    cond = np.linalg.cond(A)
    if cond > 1e10:
        print(f"Warning: ill-conditioned system, cond={cond: .2e}")

    # --------------------
    # Solve for Voriticity
    # --------------------
    vorticity = np.linalg.solve(A, b)

    # ----------------------
    # Check Kutta resolution 
    # ----------------------
    if mode == 'gamma':
        kutta_res = vorticity[i_u] + vorticity[i_l]
    else:
        kutta_res = vorticity[i_u]*geom["ds"][i_u] + vorticity[i_l]*geom["ds"][i_l]

    if abs(kutta_res) >= 1e-3:   # relaxed tolerance since it's not the constraint  # <<<
        print(f"Kutta check (legacy) = {kutta_res:.3e} | TE vorticities: upper={vorticity[i_u]:.3e}, lower={vorticity[i_l]:.3e}\n")

    return coup, vorticity
