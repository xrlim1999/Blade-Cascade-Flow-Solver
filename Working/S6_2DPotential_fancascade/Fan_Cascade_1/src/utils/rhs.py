import numpy as np


def camber_from_midpoints(x, y):
    """
    Construct camber-line coordinates from already-matched upper/lower midpoints.
    
    Inputs:
      x_upper, y_upper : arrays of midpoints on the upper surface
      x_lower, y_lower : arrays of midpoints on the lower surface
        (assumed same length and symmetric indexing)
    
    Returns:
      x_c, y_c : camber-line coordinates (averaged)
      x_u, y_u : upper_airfoil coordinates (LE --> TE)
      x_l, y_l : lower-airfoil coordinates (LE --> TE)

    """
    if y.shape[0] % 2 != 0:
        raise ValueError("Upper and Lower airfoils have uneven number of nodes.")

    n_points = int(y.shape[0] / 2)


    x_u = np.asarray(x[:n_points][::-1]) ; x_l = np.asarray(x[n_points:])
    y_u = np.asarray(y[:n_points][::-1]) ; y_l = np.asarray(y[n_points:])

    x_c = 0.5 * (x_u + x_l)
    y_c = 0.5 * (y_u + y_l)

    return x_c, y_c, x_u, y_u, x_l, y_l

def correction_all(x_c, y_c, x_u, y_u, x_l, y_l,
                   x_m, y_m, slope_m, t,
                   r_c, r1, r2, AVR, U1_ref=1.0):
    """
    Compute correction terms for irrotationality within inner-blade regions

    Inputs:
      x_u, y_u : camber-line coordinates (averaged)
      x_u, y_u : array of midpoints on the upper surface
      x_l, y_l : array of midpoints on the lower surface
      
    
    Returns:
      c_omega_m : array of correction terms for each node along the camber line

    """
    Nc = len(x_c)

    # ========================================================
    ## Area between each camber-line node (trapezium rule)
    # ========================================================
    """ camber-strip dimensions """
    # Arc length between consecutive camber nodes
    ds_c   = np.sqrt(np.diff(x_c)**2 + np.diff(y_c)**2) # camber-strip length
    dth    = np.sqrt((x_u-x_l)**2 + (y_u-y_l)**2)       # camber-strip thickness
    dA_seg = 0.5 * (dth[:-1] + dth[1:]) * ds_c         # trapezoid area per segment

    """ Distribute segment areas to nodes (half to each endpoint) """
    dA_c = np.zeros(Nc)
    dA_c[0]    += 0.5 * dA_seg[0]
    dA_c[-1]   += 0.5 * dA_seg[-1]
    dA_c[1:-1] += 0.5 * (dA_seg[:-1] + dA_seg[1:])

    # ========================================================
    ## sigma(ξ): linear AVR -> constant along camber
    # ========================================================
    eps = 1e-14 # guard against infinity values
    L_c = np.sum(ds_c)
    sigma_c = np.full(Nc, U1_ref * (AVR - 1.0) / max(L_c, eps)) # (Nc,)

    # ========================================================
    ## Shared terms for Influence Kernel
    # ========================================================
    k = 2.0*np.pi / float(t)
    dx = x_m[:, None] - x_c[None, :] # (Nm, Nc)
    dy = y_m[:, None] - y_c[None, :] # (Nm, Nc)

    sinb = np.sin(slope_m)[:, None] # (Nm, 1)
    cosb = np.cos(slope_m)[:, None] # (Nm, 1)
    denom = ( np.cosh(k*dx) - np.cos(k*dy) ) + eps

    # ========================================================
    ## Influence Sigma kernel K_{m,c}
    # ========================================================
    numer_sigma = ( np.sinh(k*dx) * cosb + np.sin(k*dy) * sinb )
    K_mc = (1.0/(2.0*t)) * (numer_sigma / denom)

    # --- assemble unscaled C_{sigma,m} ---
    weights_c_sigma = dA_c * sigma_c                            # (Nc,)
    C_sigma_m_raw = (K_mc * weights_c_sigma[None, :]).sum(axis=1) # (Nm,)


    # ========================================================
    ## Omega(ξ_c) = r_c^2 - 0.5*(r1^2+r2^2)
    # ========================================================
    r_c = np.asarray(r_c, dtype=float)
    rbar2 = 0.5*(r1**2 + r2**2)
    omega_c = r_c**2 - rbar2

    # ========================================================
    ## Influence Omega kernel K_{m,c}
    # ========================================================
    numer_omega = ( - np.sinh(k*dx) * sinb + np.sin(k*dy) * cosb )
    K_mc = (1.0/(2.0*t)) * (numer_omega / denom)

    # --- assemble unscaled C_{Omega,m} ---
    weights_c_omega = dA_c * omega_c                              # (Nc,)
    C_omega_m_raw = (K_mc * weights_c_omega[None, :]).sum(axis=1) # (Nm,)

    return C_sigma_m_raw, C_omega_m_raw

def right_hand_side_rotor_unit(m, xmid, ymid, ds, slope, t,
                          r_c, r1, r2, r_m, AVR=1.0):
    """
    Calculate Right-Hand Side values

    Inputs:
      m          : number of panels
      xmid, ymid : array of midpoint coordinates of each panel
      ds         : array of length of each panel
      slope      : array of angle of each panel
      t          : airfoil pitch
      r_c        : array of radial coordinate of each camber-line node
      r1         : radial coordinate of L.E.
      r2         : radial coordinate of T.E.
      r_m        : array of radial coordinate of each panel midpoint node
      AVR        : axial velocity ratio
      
    
    Returns:
      rhs_U     : array of right-hand side values for unit solution of U = 1.0
      rhs_V     : array of right-hand side values for unit solution of V = 1.0
      rhs_Omega : array of right-hand side values for unit solution of Omega = 1.0

    """
    rhs_U = np.zeros(m) ; rhs_V = np.zeros(m) ; rhs_Omega = np.zeros(m)
    x_le = np.min(xmid)
    x_te = np.max(xmid)

    x_c, y_c, x_u, y_u, x_l, y_l = camber_from_midpoints(xmid, ymid)

    # --- camber-line correction (C_sigma and unscaled C_omega) ---
    C_sigma_m_raw, C_omega_m_raw = correction_all(x_c, y_c, x_u, y_u, x_l, y_l,
                                                  xmid, ymid, slope, t,
                                                  r_c, r1, r2, AVR, U1_ref=1.0)
    """ Unit Solution 1 : radial U = 1.0"""
    rhs_U = - ( 1.0 + ((xmid - x_le) / (x_te - x_le)) * (AVR-1.0) ) * np.cos(slope) + C_sigma_m_raw

    """ Unit Solution 2 : radial V = 1.0"""
    rhs_V = - np.sin(slope)

    """ Unit Solution 3 : blade Ω = 1.0"""
    # --- baseline Ω unit RHS (displacement flow from rotation) ---
    rbar2 = 0.5*(r1**2 + r2**2)
    dw_m = (r_m**2 - rbar2)                   # (m,)
    rhs_Omega_uncorr = - dw_m * np.sin(slope) # (m,)

    # --- final Ω RHS: baseline + scaled correction ---
    numer = np.sum(C_omega_m_raw * ds)
    denom = np.sum(dw_m * np.sin(slope) * ds)
    # guard against infinite values
    if abs(denom) < 1e-14:
        chi = 1.0
    else:
        chi = numer / denom
    
    rhs_Omega = rhs_Omega_uncorr + (C_omega_m_raw / chi)
    

    return rhs_U, rhs_V, rhs_Omega

def right_hand_side_stator_unit(m, slope):
    """
    Build RHS vectors for U- and V-unit problems.
    Each RHS is evaluated at all m control points.
    """
    rhs_U, rhs_V = np.zeros(m), np.zeros(m)

    rhs_U = - ( (1.0 * np.cos(slope)) + (0.0 * np.sin(slope)) )
    rhs_V = - ( (0.0 * np.cos(slope)) + (1.0 * np.sin(slope)) )

    return rhs_U, rhs_V
