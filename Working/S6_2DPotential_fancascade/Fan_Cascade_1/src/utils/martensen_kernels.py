import numpy as np
import math


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
