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
    slope  = np.unwrap(geom["slope"])
    xmid   = geom["xmid"]
    ymid   = geom["ymid"]

    m = geom["n_panels"]
    coup = np.zeros((m, m))

    # ---- periodic cascade kernel (Lewis) with core regularization ----
    # core ~ 20% of the smaller of the two panel lengths
    inv2pi = 1.0 / (2.0*np.pi)

    def idx(k):
        return k % m

    # compute self-inducing coupling coefficients
    # coup[0,0] = - 0.5 - (slope[1] - slope[m-1] - 2.0*np.pi) / (8.0*np.pi)
    # coup[m-1,m-1] = - 0.5 - (slope[0] - slope[m-2] - 2.0*np.pi) / (8.0*np.pi)

    for i in range(1, m-1):
        dtheta = slope[idx(i+1)] - slope[idx(i-1)]
        coup[i,i] = -0.5 - dtheta / (8.0*np.pi)   # no -2π needed

    for i in range(m):
        for j in range(m):
            if j == i:
                continue

             # ---- off-diagonal: free-space Biot–Savart with soft core ----
            eps2 = (0.2 * float(np.min(ds)))**2  # e.g., 0.1–0.3 of min panel length

            dx = xmid[i] - xmid[j]
            dy = ymid[i] - ymid[j]

            r2 = dx*dx + dy*dy + eps2

            # velocity at i due to unit-strength vortex sheet on panel j
            u = (-dy) * inv2pi / r2
            v = (+dx) * inv2pi / r2

            # Project onto panel tangents (your convention)
            coup[i, j] = (u * cosine[i] + v * sine[i]) * ds[j]

    # ---- Back Diagonal Correction (same as before) ----
    apply_back_diagonal_correction(coup, ds)

    # Fail fast if something is off
    if not np.all(np.isfinite(coup)):
        raise RuntimeError("Non-finite entries in cascade coupling matrix.")
    
    col_resid = np.abs(coup.T @ ds)
    if col_resid.max() >= 1e-12:
        raise RuntimeError(f"BDC residual too large after cascade kernel: max={col_resid.max():.3e}")
    
    return coup

