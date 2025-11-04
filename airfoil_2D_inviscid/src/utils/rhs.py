import numpy as np


def right_hand_side_airfoil(geom: dict, flow: dict, delta, Ue):
    """
    RHS for Martensen/Lewis/Hill boundary condition:
      rhs = -[ U*cos(phi) + (V - Omega*r)*sin(phi) ]

    Uses cosine/sine of local panel tangent stored in geom.
    """

    # --- compute blow-out velocity ---
    if delta is not None:
        num = len(delta)
        ddelta_ds = np.zeros(num)
        dUe_ds = np.zeros(num)

        # ddelta_ds[0] = 

        # for i in range(1, num-1):
            
    else:
        rhs = -(flow["U"] * geom["cosine"] + flow["V"] * geom["sine"])

    return rhs


