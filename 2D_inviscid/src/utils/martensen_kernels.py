import numpy as np
import math


def apply_back_diagonal_correction(coup, ds, n_passes=2):
    """
    Apply the Back Diagonal Correction (BDC) to the coupling matrix.

    Enforces the no-through-flow condition by ensuring each column of the
    coupling matrix satisfies:

        sum_i( coup[i,j] * ds[i] ) = 0  for all j

    This is achieved by adjusting the opposite-diagonal entry coup[i_opp, j]
    (where i_opp = M-1-j) to absorb any residual column sum. The correction
    is applied in-place and repeated for n_passes to improve convergence.

    Parameters
    ----------
    coup : ndarray, shape (M, M)
        Coupling coefficient matrix. Modified in-place.
    ds : ndarray, shape (M,)
        Panel lengths.
    n_passes : int, optional
        Number of correction passes. Default is 2.

    Raises
    ------
    ValueError
        If coup contains any NaN or Inf entries before correction.
    """
    M = coup.shape[0]

    # guard against zero or non-finite panel lengths
    floor = max(1e-12 * float(np.mean(ds)), 1e-18)
    ds_safe = np.where(np.isfinite(ds) & (ds > floor), ds, floor)

    if not np.all(np.isfinite(coup)):
        raise ValueError("coup has NaN/inf before BDC.")

    for _ in range(n_passes):
        for j in range(M):
            i_opp = M - 1 - j  # opposite-diagonal index for column j
            denom = ds_safe[i_opp]

            # column sum weighted by ds, excluding the opposite-diagonal entry
            col = coup[:, j] * ds_safe
            s = float(math.fsum(col) - coup[i_opp, j] * ds_safe[i_opp])

            # adjust opposite-diagonal entry to zero the column sum
            coup[i_opp, j] = -s / denom

def coupling_coefficient_bound(geom):
    """
    Compute the coupling coefficient matrix for a vortex panel method.

    Each entry coup[i, j] represents the tangential velocity induced at the
    midpoint of panel i by a unit-strength vortex sheet distributed over
    panel j. The off-diagonal terms use the free-space 2D Biot-Savart kernel
    with a small vortex core regularisation to prevent singularities. The
    diagonal (self-inducing) terms use the Martensen curvature formula.

    After assembly, the Back Diagonal Correction (BDC) is applied to enforce
    the no-through-flow condition on each column.

    Parameters
    ----------
    geom : dict
        Panel geometry dictionary as returned by data_preparation, containing:
            "ds"       : panel lengths, shape (m,)
            "sine"     : sine of panel tangent angles, shape (m,)
            "cosine"   : cosine of panel tangent angles, shape (m,)
            "slope"    : panel tangent angles [rad], shape (m,)
            "xmid"     : panel midpoint x coordinates, shape (m,)
            "ymid"     : panel midpoint y coordinates, shape (m,)
            "n_panels" : number of panels, int

    Returns
    -------
    coup : ndarray, shape (m, m)
        Coupling coefficient matrix.

    Raises
    ------
    RuntimeError
        If coup contains non-finite entries after assembly.
        If the BDC column residual exceeds 1e-12.
    """
    # ----------------------
    # unpack arrays
    # ----------------------
    ds     = geom["ds"]
    sine   = geom["sine"]
    cosine = geom["cosine"]
    slope  = np.unwrap(geom["slope"]) # remove ±2π jumps for correct finite differences
    xmid   = geom["xmid"]
    ymid   = geom["ymid"]
    m      = geom["n_panels"]

    coup = np.zeros((m, m))

    # ---- periodic cascade kernel (Lewis) with core regularization ----
    # core ~ 20% of the smaller of the two panel lengths
    inv2pi = 1.0 / (2.0*np.pi) # helper variable (1/2pi)

    # -------------------------------------------
    # Compute self-inducing Coupling Coefficients
    # -------------------------------------------
    for i in range(1, m-1):
        dtheta = slope[i+1] - slope[i-1]
        coup[i,i] = -0.5 - dtheta / (8.0*np.pi) # no -2π needed

    for i in range(m):
        for j in range(m):
            if j == i:
                continue

             # vortex core regularisation: 
             #    - prevents Biot-Savart kernel from blowing up when r² → 0
            eps2 = (0.2 * float(np.min(ds)))**2

            dx = xmid[i] - xmid[j]
            dy = ymid[i] - ymid[j]

            r2 = dx*dx + dy*dy + eps2

            # velocity at i due to unit-strength vortex sheet on panel j
            u = (-dy) * inv2pi / r2
            v = (+dx) * inv2pi / r2

            # Project onto panel tangents (your convention)
            coup[i, j] = (u * cosine[i] + v * sine[i]) * ds[j]

    # ------------------------
    # Back Diagonal Correction
    # ------------------------
    apply_back_diagonal_correction(coup, ds)

    # ------------------------
    # Failure check
    # ------------------------
    if not np.all(np.isfinite(coup)):
        raise RuntimeError("Non-finite entries in cascade coupling matrix.")
    
    col_resid = np.abs(coup.T @ ds)
    if col_resid.max() >= 1e-12:
        raise RuntimeError(f"BDC residual too large after cascade kernel: max={col_resid.max():.3e}")
    
    return coup
