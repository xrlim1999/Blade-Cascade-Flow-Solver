from __future__ import annotations
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d, Akima1DInterpolator


def load_airfoil_xy(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load aerofoil coordinates from a two-column text file (x, y)
    """
    data = np.loadtxt(path)

    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected 2+ columns, got shape {data.shape}")
    
    x = data[:, 0].astype(float)
    y = data[:, 1].astype(float)
    
    return x, y

def rearrange_airfoil(x, y, *, tol: float = 1e-8):
    """
    Reorganise raw airfoil coordinates into a standardised closed loop:
        TE(upper) -> upper -> LE -> lower -> TE(lower)

    Handles both sharp (1 TE node) and finite-thickness (2 TE nodes) trailing
    edges. Coordinates are always normalised by chord so that x lies in [0, 1].
    The trailing edge y is averaged between the upper and lower TE nodes.
    Maximum thickness is computed as the largest perpendicular distance between
    the upper and lower surfaces at matched chordwise stations.

    Parameters
    ----------
    x, y : array-like
        Raw airfoil coordinates in any starting point or order.
    tol : float, optional
        Absolute tolerance for detecting TE and LE candidate points.
        Default is 1e-8.

    Returns
    -------
    xnew : ndarray
        Normalised x coordinates, ordered TE->upper->LE->lower->TE, x in [0, 1].
    ynew : ndarray
        Normalised y coordinates, ordered TE->upper->LE->lower->TE.
    chord : float
        Always 1.0 after normalisation.
    x_le : float
        Normalised x coordinate of the leading edge (~0.0).
    y_le : float
        Normalised y coordinate of the leading edge.
    x_te : float
        Normalised x coordinate of the trailing edge (~1.0).
    y_te : float
        Normalised y coordinate of the trailing edge (averaged if two TE nodes).
    max_thick : float
        Maximum thickness as a fraction of chord.

    Raises
    ------
    ValueError
        If fewer than 4 input points are provided.
        If the number of TE candidates is not 1 or 2.
        If the chord length is degenerate (effectively zero).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = int(x.size)
    if n < 4:
        raise ValueError("Need at least 4 points for an airfoil outline.")
    
    # ----------------------------------------------
    # Shift coordinates based on TE
    #   - TE indexes start and/or end at index 0/end
    # ----------------------------------------------
    te_inds = np.where(np.isclose(x, np.max(x), atol=tol, rtol=0.0))[0]

    # shift coordinates
    x_shifted, y_shifted = x.copy(), y.copy()
    num_te = te_inds.size
    if num_te == 2:
        te_first_ind  = int(te_inds[0])
        te_second_ind = int(te_inds[-1])

        # If TE isn't at ends, rotate so segment is contiguous [te_start:te_end]
        if te_second_ind != n-1:
            # roll so that te_start goes to 0, te_end goes to n-1
            roll = (n-1) - te_second_ind

            x_shifted = np.roll(x, roll)
            y_shifted = np.roll(y, roll)

    elif num_te == 1:
        te_first_ind = int(te_inds[0])

        # Only one TE detected: rotate so TE is at index 0 and duplicate at end
        x_shifted = np.roll(x, -te_first_ind)
        y_shifted = np.roll(y, -te_first_ind)

        x_shifted = np.append(x_shifted, x_shifted[0])
        y_shifted = np.append(y_shifted, y_shifted[0])

        n = x_shifted.size
    
    else:
        raise ValueError(f"Expected 1 or 2 trailing-edge candidates, got {te_inds.size}. Check input coordinates.")

    # rearrange into standardised coordinate order (TE --> upper --> LE --> lower --> TE)
    x_shifted = np.concatenate([x_shifted[0:te_first_ind+1][::-1], x_shifted[te_first_ind+1:]])
    y_shifted = np.concatenate([y_shifted[0:te_first_ind+1][::-1], y_shifted[te_first_ind+1:]])

    # -------------------
    # Locate Leading Edge
    # -------------------
    le_inds = np.where(np.isclose(x_shifted, np.min(x_shifted), atol=tol, rtol=0.0))[0]
    if le_inds.size == 1:
        x_le_raw = x_shifted[le_inds[0]]
        y_le_raw = y_shifted[le_inds[0]]

    elif le_inds.size == 2:
        x_le_raw = np.mean(x_shifted[le_inds])
        y_le_raw = np.mean(y_shifted[le_inds])

        # replace all LE candidates with a singular averaged LE point
        x_shifted[le_inds[0]] = x_le_raw
        y_shifted[le_inds[0]] = y_le_raw
        x_shifted = np.delete(x_shifted, le_inds[1:])
        y_shifted = np.delete(y_shifted, le_inds[1:])
    
    else:
        raise ValueError(f"Expected 1 or 2 leading-edge candidates, got {le_inds.size}. Check input coordinates.")

    # ------------------------------------
    # Locate new Trailing Edge coordinates
    # ------------------------------------
    x_te_raw = x_shifted[0]  # ~ x_max
    y_te_avg = 0.5 * (y_shifted[0] + y_shifted[-1])

    # --------------------------------
    # Normalise coordinates with chord
    # --------------------------------
    x_min, x_max = np.min(x_shifted), np.max(x_shifted)
    chord = x_max - x_min
    if chord <= 1e-4:
        raise ValueError("Degenerate chord length detected.")
    
    xnew = (x_shifted - x_min) / chord
    ynew = y_shifted / chord
    x_le = (x_le_raw - x_min) / chord
    y_le = y_le_raw / chord
    x_te = (x_te_raw - x_min) / chord
    y_te = y_te_avg / chord

    chord = 1.0

    # -------------
    # Max thickness
    # -------------
    le_ind = np.argmin(xnew)

    x_upper = xnew[:le_ind+1][::-1]
    y_upper = ynew[:le_ind+1][::-1]
    x_lower = xnew[le_ind:]
    y_lower = ynew[le_ind:]
    
    # interpolate lower surface onto upper surface [identical] x-stations
    f_lower = np.interp(x_upper, x_lower, y_lower)
    max_thick = np.max(y_upper - f_lower)

    return xnew, ynew, chord, x_le, y_le, x_te, y_te, max_thick

def _monotone_interp(x, y):
    """
    Build a 1D interpolant from (x, y) data, robust to duplicate or
    near-duplicate x values.

    Preprocessing removes exact duplicates via np.unique and filters out
    near-duplicate x values (gap < 1e-12) to ensure numerical stability.
    The interpolation method degrades gracefully based on the number of
    unique points remaining:
        >= 4 points : cubic spline (scipy interp1d, kind='cubic')
        >= 3 points : Akima spline (more robust near sharp changes)
        <  3 points : linear interpolation (last resort fallback)

    Parameters
    ----------
    x : array-like
        x coordinates of the data points. Must be broadly monotone
        (duplicates and near-duplicates are handled internally).
    y : array-like
        y coordinates of the data points, same length as x.

    Returns
    -------
    interpolant : callable
        A callable f(x_new) that evaluates the interpolant at new x
        positions. Extrapolation is enabled outside the data range.

    Notes
    -----
    This function is intended for airfoil surface interpolation where
    the LE point (x=0) may appear in both upper and lower surface arrays.
    The duplicate removal ensures the interpolant remains well-defined.

    """
    x = np.asarray(x)
    y = np.asarray(y)

    # ----------------
    # Data clean-up
    # ----------------
    # remove duplicate x-values
    xu, idx = np.unique(x, return_index=True)
    yu = y[idx]

    # remove near-duplicates 
    #   (since floats can be very close to each other and cause issues)
    keep = np.ones_like(xu, dtype=bool)
    keep[1:] &= (np.diff(xu) > 1e-12)

    xu = xu[keep]
    yu = yu[keep]

    # -------------------------------------------------
    # Interpolotation
    #   - method chosen based on how many points remain
    # -------------------------------------------------
    if xu.size >= 4:
        try:
            return interp1d(xu, yu, kind='cubic', bounds_error=False, fill_value='extrapolate')
        except Exception:
            return Akima1DInterpolator(xu, yu)
    elif xu.size >= 3:
        return Akima1DInterpolator(xu, yu)
    else:
        return interp1d(xu, yu, kind='linear', bounds_error=False, fill_value='extrapolate')
    
def resample_airfoil_cosine_finite_TE(x, y, n_points : int = 200):
    """
    Resample a closed airfoil loop using cosine spacing along the chord,
    preserving a finite-thickness trailing edge.

    Takes a TE->upper->LE->lower->TE ordered coordinate loop from
    rearrange_airfoil and resamples it onto a cosine-spaced grid.
    Cosine spacing clusters points near the LE and TE where curvature
    is highest, and spaces them more widely near midchord.

    The output is an open loop — the closing TE panel is formed
    implicitly by data_preparation connecting the last node back to
    the first node.

    Parameters
    ----------
    x, y : array-like
        Closed-loop airfoil coordinates ordered TE->upper->LE->lower->TE.
        If not already closed, the first point is appended to close the loop.
    n_points : int, optional
        Total number of output nodes. Must be even — odd values are
        silently incremented by 1. Half the points are placed on each
        surface. Default is 200.

    Returns
    -------
    x_new : ndarray
        Cosine-resampled x coordinates, ordered TE->upper->LE->lower->TE.
        Shape: (n_points,). x lies in [0, 1].
    y_new : ndarray
        Cosine-resampled y coordinates, same ordering as x_new.
        Shape: (n_points,).

    Notes
    -----
    The LE point is averaged between the upper and lower surface
    interpolants to enforce a single shared LE node.
    The TE y values are kept distinct (not averaged) to preserve
    the finite thickness of the trailing edge.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    # ---------------------------------------------
    # Ensure coordinate loop is closed exactly once
    # ---------------------------------------------
    if not (np.isclose(x[0], x[-1]) and np.isclose(y[0], y[-1])):
        x = np.r_[x, x[0]]
        y = np.r_[y, y[0]]

    # ---------------------------------------------
    # Split upper and lower coordinates
    #   - LE and TE coordinates included
    #   - sequence: LE -> TE
    # ---------------------------------------------
    le_ind = int(np.argmin(x))
    x_upper, y_upper = x[:le_ind+1][::-1], y[:le_ind+1][::-1]
    x_lower, y_lower = x[le_ind:-1], y[le_ind:-1]

    # ---------------------------
    # Build Interpolant
    # ---------------------------
    f_upper = _monotone_interp(x_upper, y_upper)
    f_lower = _monotone_interp(x_lower, y_lower)

    # -------------------------------------------
    # Cosine spacing along chord
    #   - Cosine spacing clusters points near the
    #     LE and TE where curvature is hightest
    #   - spaces them more widely near midchord
    # -------------------------------------------
    if n_points % 2:
        n_points += 1
    n_side = n_points // 2

    # evenly spaced angles from 0 → π (both endpoints included)
    beta = np.linspace(0, np.pi, n_side + 1)

    # cosine transformation: equal Δbeta → unequal Δx
    #   beta=0   → x=0.0 (LE)
    #   beta=π/2 → x=0.5 (midchord)
    #   beta=π   → x=1.0 (TE)
    x_cos = 0.5 * (1 - np.cos(beta))

    # evaluate upper and lower surface y values at cosine x stations
    yu = f_upper(x_cos)
    yl = f_lower(x_cos)

    # enforce a single LE point
    y_le = 0.5 * (yu[0] + yl[0])  # or 0.0 if you want sharp LE at y=0 after normalize
    yu[0] = yl[0] = y_le

    # -----------------------------------
    # Assemble interpolated coordinates
    #   - final array: open-loop
    # -----------------------------------
    x_new = np.r_[x_cos[::-1], x_cos[1:]]
    y_new = np.r_[yu[::-1]   , yl[1:]   ]

    return x_new, y_new

def _rotate_about_point(x, y, x_pivot_pt, y_pivot_pt, beta_rad):
    """
    Rigid rotation of all nodes about pivot points (x_pivot_pt, y_pivot_pt)
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    beta_rad *= -1

    cosine = np.cos(beta_rad)
    sine   = np.sin(beta_rad)

    dx = x_pivot_pt - x
    dy = y_pivot_pt - y

    x_rotated = x_pivot_pt - (dx * cosine - dy * sine)
    y_rotated = y_pivot_pt - (dx * sine + dy * cosine)

    return x_rotated, y_rotated

def _reposition_airfoil(x, y):
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

def rotate_airfoil_about_te(x, y, beta_rad):
    """
    Rotate a closed airfoil loop about its trailing edge midpoint.

    The pivot point is taken as the midpoint between the upper and lower
    trailing edge nodes (index 0 and index -1). A positive beta_rad rotates
    the leading edge upward (clockwise when viewed conventionally with x
    pointing right).

    After rotation, the airfoil is repositioned so that the leading edge
    sits at x = 0 and the geometry is vertically centered about y = 0.

    Parameters
    ----------
    x, y : array-like
        Closed-loop airfoil coordinates ordered TE->upper->LE->lower->TE.
    beta_rad : float
        Rotation angle in radians. Positive values tilt the leading edge up.

    Returns
    -------
    x_out, y_out : ndarray
        Rotated and repositioned airfoil coordinates, same ordering as input.
    """
    # upper and lower TE node index
    i_u = 0
    i_l = (len(x)-1)

    # ---------------------------------
    # Determine pivot point
    #   - taken as center of 2 TE nodes
    # ---------------------------------
    x_pivot_pt = 0.5*(x[i_u] + x[i_l])
    y_pivot_pt = 0.5*(y[i_u] + y[i_l])

    # ---------------------------------
    # Perform airfoil rotation
    # ---------------------------------
    x_rotated, y_rotated = _rotate_about_point(x, y, x_pivot_pt, y_pivot_pt, beta_rad)

    # keep exact closure if your first and last nodes are duplicates
    if np.allclose([x[0], y[0]], [x[-1], y[-1]]):
        x_rotated[-1], y_rotated[-1] = x_rotated[0], y_rotated[0]

    # ----------------------------------
    # Reposition airfoil at center (x,y)
    # ----------------------------------
    xr_repos, yr_repos = _reposition_airfoil(x_rotated, y_rotated)

    return xr_repos, yr_repos
