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

def right_hand_side_simple(m, slope, alpha, W):
    """
    Calculate Right-Hand Side values
    """
    rhs = np.zeros(m)

    for i in range(m):
        rhs[i] = -W * (math.cos(alpha)*math.cos(slope[i]) + math.sin(alpha)*math.sin(slope[i]))

    return rhs

def vorticity_solution_kutta(coup, rhs, ds, mode='circulation'):
    """
    Solve A*gamma = rhs with -edge unloading (Kutta) as the single closure.
    mode:
    - 'gamma'        -> enforce gamma_upper + gamma_lower = 0     (what you asked)
    - 'circulation'  -> enforce gamma_u*ds_u + gamma_l*ds_l = 0   (length-weighted)
    """
    A = coup.copy()
    b = rhs.copy()

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
    A[-1, :] = row
    b[-1]    = 0.0

    # tiny ridge for tough geometries (optional)
    A[-1, -1] += 1e-14

    vorticity = np.linalg.solve(A, b)

    # report
    if mode == 'gamma':
        kutta_res = vorticity[i_u] + vorticity[i_l]
    else:
        kutta_res = vorticity[i_u]*ds[i_u] + vorticity[i_l]*ds[i_l]
    
    if kutta_res >= 1e-10:
        print("Kutta check - FAILED")
        print(f"Kutta residual ({mode}) = {kutta_res:.3e}  |  TE indices: upper={i_u}, lower={i_l} | TE vorticities: upper={vorticity[i_u]}, lower={vorticity[i_l]}")

    return vorticity
