import numpy as np


def right_hand_side_airfoil(geom: dict, flow: dict):
    """
    Compute the right-hand side vector for the Martensen no-through-flow
    boundary condition.

    Evaluates the negative of the freestream velocity projected onto each
    panel's tangent direction:

        rhs[i] = -( U*cos(phi_i) + V*sin(phi_i) )

    where phi_i is the tangent angle of panel i, and U, V are the freestream
    velocity components.

    Parameters
    ----------
    geom : dict
        Panel geometry dictionary as returned by data_preparation, containing
        "cosine" and "sine" arrays of panel tangent angles, shape (m,).
    flow : dict
        Flow conditions dictionary containing scalar freestream components
        "U" and "V" [m/s].

    Returns
    -------
    rhs : ndarray, shape (m,)
        Right-hand side vector for the linear system.
    """
    return -(flow["U"] * geom["cosine"] + flow["V"] * geom["sine"])


