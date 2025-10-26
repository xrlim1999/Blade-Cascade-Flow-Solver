from __future__ import annotations
import numpy as np
from scipy.interpolate import interp1d, Akima1DInterpolator


def rearrange_airfoil(x, y, *, normalize: bool = True, tol: float = 1e-8):
    """
    Reorganise airfoil coordinates to:
        TE -> upper -> LE -> lower -> TE
    and close the TE by averaging the two TE y-values.

    Parameters
    ----------
    1. x, y : array-like
          Raw airfoil coordinates (any starting point/order).
    2. normalize : bool, default True
          If True, shift/scale so x in [0, 1] and y scaled by chord.
    3. tol : float
          Tolerance for detecting the two TE points (near max x).

    Returns
    -------
    xout, yout, x_le, y_le, x_te, y_te, chord
        xout, yout follow TE->upper->LE->lower->TE with closed TE.
        x_le,y_le are LE coordinates (after normalization if normalize=True).
        x_te,y_te are TE coordinates (after normalization if normalize=True).
        chord is x_te - x_le (before normalization if normalize=False,
        equals 1.0 if normalize=True).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = int(x.size)
    if n < 4:
        raise ValueError("Need at least 4 points for an airfoil outline.")
    
    # --- Find TE candidates (max x) ---
    x_max = np.max(x)
    te_idxs = np.where(np.isclose(x, x_max, atol=tol, rtol=0.0))[0]

    if te_idxs.size >= 2:

        te_first  = int(te_idxs[0])
        te_second = int(te_idxs[-1])

        # If TE isn't at ends, rotate so segment is contiguous [te_start:te_end]
        if te_second != n-1:

            # roll so that te_start goes to 0
            roll = (n-1) - te_second
            te_first = int(te_idxs[0]) + roll

            x = np.roll(x, roll); y = np.roll(y, roll)

    else:
        # Only one TE detected: rotate so TE is at index 0 and duplicate at end
        te0 = int(np.argmax(x))
        x = np.roll(x, -te0); y = np.roll(y, -te0)
        x = np.append(x, x[0]); y = np.append(y, y[0])
        n = x.size

    x_loop = np.concatenate([x[0:te_first+1][::-1], x[te_first+1:]])
    y_loop = np.concatenate([y[0:te_first+1][::-1], y[te_first+1:]])

    # --- Find LE (min x) along the TE->...->TE loop ---
    ile = int(np.argmin(x))
    x_le_raw, y_le_raw = x[ile], y[ile]

    # --- Average TE y to close TE cleanly ---
    x_te_raw = x[0]  # ~ x_max
    y_te_avg = 0.5 * (y[0] + y[-1])

    # The current path is TE -> (upper to LE) -> (lower to TE), as in Selig format.
    # Remove duplicated middle LE point when we rebuild to avoid double-counting.

    # --- Optional normalize to chord [0,1] ---
    if normalize:

        x_min, x_max2 = np.min(x_loop), np.max(x_loop)
        chord = x_max2 - x_min

        if chord <= tol:
            raise ValueError("Degenerate chord length detected.")
        xnew = (x_loop - x_min) / chord
        ynew = y_loop / chord
        x_le = (x_le_raw - x_min) / chord
        y_le = y_le_raw / chord
        x_te = (x_te_raw - x_min) / chord
        y_te = y_te_avg / chord
        chord_out = 1.0

    else:
        xnew, ynew = x_loop, y_loop
        x_le, y_le = x_le_raw, y_le_raw
        x_te, y_te = x_te_raw, y_te_avg
        chord_out = x_te - x_le

    max_thick = max(ynew) - min(ynew)

    return xnew, ynew, x_le, y_le, x_te, y_te, chord_out, max_thick

def _monotone_interp(x, y):

    x = np.asarray(x); y = np.asarray(y)
    xu, idx = np.unique(x, return_index=True)
    yu = y[idx]
    keep = np.ones_like(xu, dtype=bool)
    keep[1:] &= (np.diff(xu) > 1e-12)
    xu, yu = xu[keep], yu[keep]
    if xu.size >= 4:
        try:
            return interp1d(xu, yu, kind='cubic', bounds_error=False, fill_value='extrapolate')
        except Exception:
            return Akima1DInterpolator(xu, yu)
    elif xu.size >= 3:
        return Akima1DInterpolator(xu, yu)
    else:
        return interp1d(xu, yu, kind='linear', bounds_error=False, fill_value='extrapolate')
    
def resample_airfoil_cosine_finite_TE(x_loop, y_loop, n_points : int = 200):
    """
    Resample a closed loop (TE→upper→LE→lower→TE) with a *finite-thickness TE*.
    Returns n_panels+1 nodes; last node duplicates the first (upper-TE) to close.
    """
    x = np.asarray(x_loop).ravel()
    y = np.asarray(y_loop).ravel()

    # ensure closed exactly once
    if not (np.isclose(x[0], x[-1]) and np.isclose(y[0], y[-1])):
        x = np.r_[x, x[0]]; y = np.r_[y, y[0]]

    # find LE (minimum x); split: TE→LE (upper), LE→TE (lower)
    i_le = int(np.argmin(x))
    x_upper, y_upper = x[:i_le+1], y[:i_le+1] # TE(upper) → LE
    x_lower, y_lower = x[i_le:], y[i_le:]     # LE → TE(lower), exclude final dup TE

    # build interpolants in LE→TE direction
    f_upper = _monotone_interp(x_upper[::-1], y_upper[::-1])  # upper reversed to LE→TE
    f_lower = _monotone_interp(x_lower,       y_lower)        # lower already LE→TE

    # cosine abscissae along chord [0,1]
    if n_points % 2:
        n_points += 1

    n_side = n_points // 2
    beta = np.linspace(0, np.pi, n_side + 1)    # include both ends (LE & TE)
    x_cos = 0.5 * (1 - np.cos(beta))            # 0..1 (LE→TE)

    yu = f_upper(x_cos)   # LE→TE (upper)
    yl = f_lower(x_cos)   # LE→TE (lower)

    # enforce a single LE point
    y_le = 0.5 * (yu[0] + yl[0])  # or 0.0 if you want sharp LE at y=0 after normalize
    yu[0] = yl[0] = y_le

    # --- finite-thickness TE: keep distinct TE y's, do NOT average/collapse ---
    # assemble TE→upper→LE→lower→TE:
    # start at TE(upper)=yu[-1], traverse to LE, then to TE(lower)=yl[-1],
    # and finally close back to TE(upper)=yu[-1] with a short vertical panel.
    x_new = np.r_[x_cos[::-1], x_cos[1:]]  # [1..0] + [0..1] + close at 1
    y_new = np.r_[yu[::-1],     yl[1:]  ]

    return x_new, y_new

def rotate_about_point(x, y, xc, yc, beta, counterclockwise_positive=True):
    """Rigid rotation of all nodes about (xc,yc)."""
    if not counterclockwise_positive:
        beta = -beta
    
    beta = np.deg2rad(beta)

    c, s = np.cos(beta), np.sin(beta)

    x = np.asarray(x, float); y = np.asarray(y, float)
    xr = xc + (x - xc)*c - (y - yc)*s
    yr = yc + (x - xc)*s + (y - yc)*c

    return xr, yr

def reposition_airfoil(x, y, tol: float = 1e-12):
    """
    Reposition the airfoil so that:
      - Leading edge (min x) is at x = 0
      - Airfoil is vertically centered, i.e. max(y) and min(y) symmetric about 0

    Parameters
    ----------
    x, y : array-like
        Closed-loop airfoil coordinates (TE->upper->LE->lower->TE).
    tol : float, optional
        Small tolerance to guard against numerical issues.

    Returns
    -------
    xr, yr : ndarray
        Repositioned coordinates with LE at 0 and vertically symmetric about x-axis.
    """
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()

    # --- Shift LE to x = 0 ---
    x_le = np.min(x)
    xr = x - x_le

    # --- Vertically center ---
    y_max, y_min = np.max(y), np.min(y)
    y_shift = 0.5 * (y_max + y_min)
    yr = y - y_shift

    return xr, yr

def rotate_airfoil_about_te(x, y, beta_deg, pivot : str = 'mid'):
    """
    Rotate the closed airfoil loop about the trailing edge by beta_deg.
    pivot: 'mid'  -> midpoint between upper/lower TE nodes (recommended)
           'upper'-> upper TE node
    """
    # upper and lower TE node index
    i_u, i_l = (x.shape[0]-1), 0

    if pivot == 'mid':
        xc = 0.5*(x[i_u] + x[i_l])
        yc = 0.5*(y[i_u] + y[i_l])
    elif pivot == 'upper':
        xc, yc = x[i_u], y[i_u]
    else:
        raise ValueError("pivot must be 'mid' or 'upper'")

    xr, yr = rotate_about_point(x, y, xc, yc, beta_deg, counterclockwise_positive=True)

    # keep exact closure if your first and last nodes are duplicates
    if np.allclose([x[0], y[0]], [x[-1], y[-1]]):
        xr[-1], yr[-1] = xr[0], yr[0]

    xr_repos, yr_repos = reposition_airfoil(xr, yr)

    return xr_repos, yr_repos

