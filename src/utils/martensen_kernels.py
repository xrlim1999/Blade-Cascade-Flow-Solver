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

def coupling_coefficient_bound(geom):

    # unpack -- geom --
    ds     = geom["ds"]
    sine   = geom["sine"]
    cosine = geom["cosine"]
    slope  = geom["slope"]
    xmid   = geom["xmid"]
    ymid   = geom["ymid"]
    t      = geom["pitch"]

    if not np.isfinite(t) or t <= 0:
        raise ValueError("Pitch t must be positive for cascade kernel.")

    m = geom["n_panels"]
    coup = np.zeros((m, m))

    slope = np.unwrap(slope)

    # ---- periodic cascade kernel (Lewis) with core regularization ----
    # core ~ 20% of the smaller of the two panel lengths
    ds_min = float(np.min(ds))
    core   = 0.2 * ds_min
    core_over_t_sq = (2*np.pi*core / t)**2
    tiny = 1e-14

    for i in range(m):
        for j in range(m):
            if j == i:
                coup[i, j] = -0.5
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

def coupling_coefficient_trailing(geom, beta_wake_init=None):
    """
    Build the trailing-vortex influence matrix L_mn.
    If wake_dir is None, estimate the wake direction from the TE camber-line
    and force it to point downstream (+x).
    wake_dir shape: (m, 2), rows are [cos(beta_wake_j), sin(beta_wake_j)].
    """

    # unpack -- geom --
    ds     = geom["ds"]
    sine   = geom["sine"]
    cosine = geom["cosine"]
    xmid   = geom["xmid"]
    ymid   = geom["ymid"]

    m = geom["n_panels"]
    core = 0.2 * float(np.min(ds))

    # coupling matrix L
    L = np.zeros((m, m))

    # --- determine initial wake direction from TE geometry ---
    if beta_wake_init is None:
        dx_te = xmid[-1] - xmid[0]
        dy_te = ymid[-1] - ymid[0]

        beta_wake_init = np.arctan2(dy_te, dx_te) + (np.pi/2) # angle of TE camber-line wrt +x

    wd = np.array([np.cos(beta_wake_init), np.sin(beta_wake_init)])
    wake_dir = np.tile(wd, (m, 1)) # same direction for all panels

    # Build L (induced tangential velocity at field j due to unit trailing filament from source i)
    for j in range(m):          # field (collocation) index — project onto tangent at j

        for i in range(m):      # source index

            if j == i:
                continue

            dx = xmid[i] - xmid[j]
            dy = ymid[i] - ymid[j]
            r2 = dx*dx + dy*dy + core*core

            # 2D Biot–Savart-like kernel; t_wake = (tx, ty) at source i
            tx, ty = wake_dir[i]
            u =  ty / r2              # induced x-velocity component
            v = -tx / r2              # induced y-velocity component

            # Resolve along field panel j tangent (cosine[j], sine[j])
            L[j, i] = (u * cosine[j] + v * sine[j]) * ds[i]

    return L, beta_wake_init

