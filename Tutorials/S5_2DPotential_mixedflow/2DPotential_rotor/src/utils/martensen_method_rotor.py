import numpy as np
import math


def data_preparation(x, y, n_datapoints):

    ds     = np.zeros(n_datapoints-1)
    sine   = np.zeros(n_datapoints-1)
    cosine = np.zeros(n_datapoints-1)
    slope  = np.zeros(n_datapoints-1)
    xmid   = np.zeros(n_datapoints-1)
    ymid   = np.zeros(n_datapoints-1)

    # set initial x and y values
    x1 = x[0]
    y1 = y[0]

    # constant for tangent angle limits
    ex = 1e-6

    for n in range(0, n_datapoints-1):

        if n < n_datapoints-1:
            x2 = x[n+1]
            y2 = y[n+1]
        else:
            x2 = x[0]
            y2 = y[0]

        ds[n]     = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        sine[n]   = (y2 - y1) / ds[n]
        cosine[n] = (x2 - x1) / ds[n]

        abscos = abs(cosine[n])

        if abscos > ex:
            t = math.atan(sine[n]/cosine[n]) # compute angle of ds wrt horizontal axis
        else:
            t = None # as the division will blow up
        
        if abscos <= ex:
            slope[n] = (sine[n] / abs(sine[n])) * np.pi / 2.0 # sets slope to + or - (90 deg)
        elif cosine[n] > ex:
            slope[n] = t # angle within TANGENT quadrant
        elif cosine[n] < -ex:
            slope[n] = t - np.pi # angle within SINE quadrant

        # compute coordinates of pivotal points
        xmid[n] = (x1 + x2) * 0.5
        ymid[n] = (y1 + y2) * 0.5

        # move to next coordinates
        x1, y1 = x2, y2

    return ds, sine, cosine, slope, xmid, ymid

def apply_back_diagonal_correction(coup, ds, n_passes=2):
    M = coup.shape[0]

    floor = max(1e-12 * float(np.mean(ds)), 1e-18)
    ds_safe = np.where(np.isfinite(ds) & (ds > floor), ds, floor)

    if not np.all(np.isfinite(coup)):
        raise ValueError("coup has NaN/inf before BDC.")

    for _ in range(n_passes):
        for j in range(M):
            i_opp = M - 1 - j
            denom = ds_safe[i_opp]

            # compensated sum: sum_i coup[i,j]*ds_safe[i], excluding diagonal & i_opp
            col = coup[:, j] * ds_safe
            s = float(math.fsum(col) - coup[i_opp, j]*ds_safe[i_opp])

            coup[i_opp, j] = -s / denom

def coupling_coefficient_cascade(t, ds, sine, cosine, slope, xmid, ymid, n_midpoints):

    if not np.isfinite(t) or t <= 0:
        raise ValueError("Pitch t must be positive for cascade kernel.")

    m = n_midpoints
    coup = np.zeros((m, m))

    slope = np.unwrap(slope)

    # ---- self terms (same as single-foil) ----
    coup[0,0]     = -0.5 - (slope[1] - slope[m-1] - 2.0*np.pi)/(8.0*np.pi)
    coup[m-1,m-1] = -0.5 - (slope[0] - slope[m-2] - 2.0*np.pi)/(8.0*np.pi)
    def idx(k): return int(k % m)
    for i in range(m):
        dtheta    = slope[idx(i+1)] - slope[idx(i-1)] - 2.0*np.pi
        coup[i,i] = -0.5 - dtheta/(8.0*np.pi)

    # ---- periodic cascade kernel (Lewis) with core regularization ----
    # core ~ 20% of the smaller of the two panel lengths
    ds_min = float(np.min(ds))
    core   = 0.2 * ds_min
    core_over_t_sq = (2*np.pi*core / t)**2
    tiny = 1e-14

    for i in range(m):
        for j in range(m):
            if j == i:
                continue

            dx = xmid[j] - xmid[i]
            dy = ymid[j] - ymid[i]

            X = 2*np.pi*dx / t
            Y = 2*np.pi*dy / t

            # Lewis periodic kernel denominator
            denom = np.cosh(X) - np.cos(Y)

            # Regularize very close pairs: as X,Y -> 0, denom ~ 0.5*(X^2 + Y^2)
            # Ensure denom never falls below that for a "core" distance
            denom = max(denom, 0.5*core_over_t_sq, tiny)

            u =  (0.5 / t) * (np.sin(Y)  / denom)
            v = -(0.5 / t) * (np.sinh(X) / denom)

            # Project onto panel tangents (your convention)
            coup[j, i] = (u * cosine[j] + v * sine[j]) * ds[i]
            coup[i, j] = -(u * cosine[i] + v * sine[i]) * ds[j]

    # ---- Back Diagonal Correction (same as before) ----
    apply_back_diagonal_correction(coup, ds)

    # Fail fast if something is off
    if not np.all(np.isfinite(coup)):
        raise RuntimeError("Non-finite entries in cascade coupling matrix.")
    col_resid = np.abs(coup.T @ ds)
    if col_resid.max() >= 1e-12:
        raise RuntimeError(f"BDC residual too large after cascade kernel: max={col_resid.max():.3e}")
    
    return coup

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

def vorticity_solution_kutta(coup, rhs_U, rhs_V, rhs_Omega, ds, mode='circulation', tol=1e-10):
    """
    Solve A*gamma = rhs with a single Kutta edge-unloading closure.

    mode:
      'gamma'        -> gamma_upper + gamma_lower = 0
      'circulation'  -> gamma_u*ds_u + gamma_l*ds_l = 0

    i_u, i_l:
      indices of the two trailing-edge unknowns (upper and lower). By default,
      assumes i_u=0 and i_l=N-1 (N = coup.shape[0])

    """
    A       = coup.copy().astype(float)

    # Prepare RHS copies and enforce Kutta RHS = 0
    b_U     = rhs_U.copy()    ; b_U[-1]    = 0.0
    b_V     = rhs_V.copy()    ; b_V[-1]    = 0.0
    b_Omega = rhs_Omega.copy(); b_Omega[-1] = 0.0

    # T.E. indices
    i_u, i_l = 0, (A.shape[0]-1)

    row = np.zeros(A.shape[1])
    # remove all vorticities elements in last row of <coup> matrix except for the 2 TE nodes
    if mode == 'gamma':
        row[i_u] = 1.0
        row[i_l] = 1.0
    elif mode == 'circulation':
        row[i_u] = ds[i_u]
        row[i_l] = ds[i_l]
    else:
        raise ValueError("mode must be 'gamma' or 'circulation'")

    # Replace LAST equation with Kutta (this removes the nullspace introduced by BDC)
    A[-1, :]    = row

    # tiny ridge for tough geometries (optional)
    A[-1, -1] += 1e-14

    vorticity_U     = np.linalg.solve(A, b_U)
    vorticity_V     = np.linalg.solve(A, b_V)
    vorticity_Omega = np.linalg.solve(A, b_Omega)

    # report
    if mode == 'gamma':
        r_U     = vorticity_U[i_u] + vorticity_U[i_l]
        r_V     = vorticity_V[i_u] + vorticity_V[i_l]
        r_Omega = vorticity_Omega[i_u] + vorticity_Omega[i_l]
    else:
        r_U = vorticity_U[i_u]*ds[i_u] + vorticity_U[i_l]*ds[i_l]
        r_V = vorticity_V[i_u]*ds[i_u] + vorticity_V[i_l]*ds[i_l]
        r_Omega = vorticity_Omega[i_u]*ds[i_u] + vorticity_Omega[i_l]*ds[i_l]

    # Report if any violate tolerance
    def _fmt(arr): return f"upper={arr[i_u]:.3e}, lower={arr[i_l]:.3e}"
    if (abs(r_U) > tol) or (abs(r_V) > tol) or (abs(r_Omega) > tol):
        print("Kutta check - FAILED")
        print(f"  residuals ({mode}): U={r_U:.3e}, V={r_V:.3e}, Ω={r_Omega:.3e}")
        print(f"  TE indices: upper={i_u}, lower={i_l}")
        print(f"  TE vorticities: U: {_fmt(vorticity_U)} | V: {_fmt(vorticity_V)} | Ω: {_fmt(vorticity_Omega)}")

    return vorticity_U, vorticity_V, vorticity_Omega

