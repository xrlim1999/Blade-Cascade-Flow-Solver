import numpy as np


def right_hand_side_3D(geom: dict, flow: dict):
    """
    RHS for Martensen/Lewis/Hill boundary condition:
      rhs = -[ U*cos(phi) + (V - Omega*r)*sin(phi) ]

    Uses cosine/sine of local panel tangent stored in geom.
    """

    V_eff = flow["V"] - flow["Omega"] * geom["r"]  # for a stator, pass Omega = 0 RPM

    rhs = -(flow["U"] * geom["cosine"] + V_eff * geom["sine"])

    return rhs


