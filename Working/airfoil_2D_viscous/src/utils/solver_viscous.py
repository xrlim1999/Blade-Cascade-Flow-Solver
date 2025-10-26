import numpy as np


def starting_assumption(geom, flow, results, viscousflow_type: str):

    # --- unpack values ---
    xmid = geom["xmid"]
    ds = geom["ds"]
    Ue      = results["V_surface"] # velocity (inviscid case) tangent to panel geometry
    kinvisc = flow["dynvisc"] / flow["rho"] # kinematic viscosity of air around airfoil

    # --- (local) Reynolds number ---
    Re = Ue / kinvisc # (incomplete)


    if viscousflow_type == "laminar":
        # 1. compute thwaite's parameter
        # 2. compute H and C_f using thwaite's parameter
        Re = 1.0 # (remove this)

    elif viscousflow_type == "turbulent":
        H = 1.40
        C_f = 0.026 * Re^(-1.0/7.0)
    
    else:
        raise ValueError("viscous flow must [ laminar ] or [ turbulent ].")
    
    # --- perform momentum integral thickness ---
    
    
    # --- (initialise) viscous terms ---
    viscous_terms = dict()
    viscous_terms["H"]   = H
    viscous_terms["C_f"] = C_f
    viscous_terms["Re"]  = Re

    return viscous_terms


def momentumthickness_integral(geom, flow, results, viscous_terms):

    
    
    
    
    return viscous_terms