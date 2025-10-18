import numpy as np
from .martensen_kernels import coupling_coefficient_bound, coupling_coefficient_trailing


def build_Cds(ds: np.ndarray, i_le: int, include_self=False) -> np.ndarray:
    """
    Set the influence matrix of each panel by the upstream panels.
        - Top and bottom panels do not affect each other
        - Only panels (n) upstream have an effect on the current panel (m)

    Return Aerofoil Surface integral operator: Cds = influence of upstream panels in
                                                     top and bottom surfaces
    """
    Ns = len(ds)
    Cds = np.zeros((Ns, Ns), dtype=float)

    k_top  = 0 if include_self else 1   # top: use UPPER triangle
    k_bot  = 0 if include_self else -1  # bottom: use LOWER triangle

    # --- TOP surface block: indices [0 : i_le)
    ds_top = ds[:i_le]
    L_top = np.triu(np.ones((i_le, i_le)), k=k_top)
    Cds[:i_le, :i_le] = L_top @ np.diag(ds_top)

    # --- BOTTOM surface block: indices [i_le : m)
    ds_bot = ds[i_le:]
    L_bot = np.tril(np.ones((Ns - i_le, Ns - i_le)), k=k_bot)  # exclude self
    Cds[i_le:, i_le:] = L_bot @ np.diag(ds_bot)

    return Cds

def build_IJ(K: np.ndarray, L: np.ndarray, ds: np.ndarray, r: float, p: int, i_le: int):
    """
    Return (I, J, A) for Fourier mode p.
      I = K  (bound-vortex block)
      J = -(i*m/r) * (L @ Cds)  (trailing-sheet block)
      A = I + J
    """
    Cds = build_Cds(ds, i_le)
    I = K.astype(complex)
    J = -(1j * p / r) * (L @ Cds)

    return I, J, I + J

def assemble_modes(K: np.ndarray, L: np.ndarray, ds: np.ndarray, r: float, A_modes: np.ndarray, i_le: int):
    """
    Sum over modes p with user weights A_p:
      A_total = sum_p A_p * (I^{(p)} + J^{(p)})
    """
    A_tot = np.zeros_like(K, dtype=complex)

    for p, Ap in enumerate(np.asarray(A_modes)):
        _, _, A_p = build_IJ(K, L, ds, r, p, i_le)
        A_tot += Ap * A_p

    return A_tot

def solve_gamma(K_mn, ds, rhs):
    """
    Solve ∫(K_mn * gamma_n dS_n) = -v_tm  (discretized form)
    K_modes: summed matrix (I+J+...)
    ds: panel lengths
    rhs: freestream tangential velocity array (-v_t∞)
    """
    N = len(ds)
    A_system = 0.5*np.eye(N) + K_mn @ np.diag(ds)
    gamma = np.linalg.solve(A_system, rhs)
    gamma = np.real_if_close(gamma)

    return gamma

def compute_exit_angle(gamma_b, ds, pitch, r, Ux_in, Vy_in, Omega, beta_wake):
    """
    Compute exit flow angle using Γ/t approximation, for rotor or stator.

    Parameters
    ----------
    gamma_b : array-like [N]
        Solved bound circulation strength per panel (m/s).
    ds : array-like [N]
        Panel lengths (m).
    pitch : float or array-like
        Blade pitch (distance between blades).
    Ux_in : float
        Axial inlet velocity component (absolute frame).
    Vy_in : float
        Tangential inlet velocity component (absolute frame).
    r : array-like [N]
        Radial coordinate (m) of each section (for Ω*r correction).
    Omega : float
        Rotational speed (rad/s). Set 0 for stator.
    wake_dir : array-like [N,2]
        Unit vectors [cos(β_wake), sin(β_wake)] for wake direction (in rotating frame).

    Returns
    -------
    beta_exit, beta_wake, delta_beta : arrays [N] (radians)
    """

    # --- Step 1: Section circulation Γ(z) ---
    Gamma = np.sum(gamma_b * ds)  # or per-section if grouping panels by z

    # --- Step 2: Mean tangential velocity change ---
    delta_Vy = Gamma / pitch

    # --- Step 3: Absolute frame exit velocity ---
    Vx_abs = np.full_like(r, Ux_in, dtype=float)
    Vy_abs = np.full_like(r, Vy_in + delta_Vy, dtype=float)

    # --- Step 4: Convert to rotating frame (for comparison to wake_dir) ---
    Vx_rel = Vx_abs
    Vy_rot = Omega * r
    Vy_rel = Vy_abs - Vy_rot

    # --- Step 5: Compute angles ---
    beta_exit = np.arctan2(Vy_rel, Vx_rel)

    delta_beta = np.arctan2(np.sin(beta_exit - beta_wake),
                            np.cos(beta_exit - beta_wake))

    return beta_exit, beta_wake, delta_beta

def solve_cascade_iterative(geom, flow, A_modes, rhs: np.ndarray,
                            max_iter=100, relax=0.5, tol_deg=1e-2):
    """
    Iteratively solve for gamma and wake_dir consistency.
    geom: dict with xmid, ymid, ds, sine, cosine, slope, pitch t, etc
    flow: dict with U_inf, V_inf, Omega.
    """
    # --- unpack geometry ---
    xmid, ymid   = geom["xmid"], geom["ymid"]
    ds           = geom["ds"]
    sine, cosine = geom["sine"], geom["cosine"]
    slope        = geom["slope"]
    t            = geom["pitch"]
    r            = geom["r"]

    # --- unpack flow ---
    U_inf = flow["U"]
    V_inf = flow["V"]
    Omega = flow.get("Omega", 0.0)

    # --- LE index ---
    i_le = int(np.argmin(xmid))

    # --- build 2D kernel (K) ---
    K = coupling_coefficient_bound(geom)

    ## Iteratively align wake_dir with exit flow direction using Γ/t approximation
    beta_wake_history = []
    beta_wake_init = None

    for k in range(max_iter):

        # Build L with current wake_dir
        L, beta_wake_init_updated = coupling_coefficient_trailing(geom, beta_wake_init=beta_wake_init)

        # --- construct K_mn using fourier modes ---
        K_mn = assemble_modes(K, L, ds, r, A_modes, i_le)

        # --- solve for bound vorticity ---
        gamma_b = solve_gamma(K_mn, ds, rhs)

        # --- compute exit flow angle ---
        beta_exit, beta_wake, delta_beta = compute_exit_angle(gamma_b, ds, t, r, U_inf, V_inf, Omega, beta_wake_init_updated)

        # --- check convergence ---
        # Log progress
        max_err = float(np.max(np.degrees(np.abs(delta_beta))))
        # print(f"[Iter {k+1:02d}] β_wake={np.degrees(beta_wake):+6.3f}°, "
        #     f"β_exit={np.degrees(beta_exit):+6.3f}°, Δβ={np.degrees(delta_beta):+6.3f}°")

        # Append current iteration values to history before updating
        beta_wake_history.append({
            "iter": k+1,
            "beta_exit": beta_exit,
            "beta_wake": beta_wake,
            "delta_beta": delta_beta
        })

        if max_err < tol_deg:
            # print(f"Beta_wake (error) = {np.degrees(delta_beta):.3f} deg | "
            #       f"Beta_wake (guess) = {np.degrees(beta_wake):.3f} deg | "
            #       f"Beta_wake (calculated) = {np.degrees(beta_exit):.3f} deg.")
            # print(f"✅ Wake direction converged after {k+1} iterations.")
            break
        elif k+1 == max_iter:
            print(f"Beta_wake (error) = {np.degrees(delta_beta):.3f} deg | "
                  f"Beta_wake (guess) = {np.degrees(beta_wake):.3f} deg | "
                  f"Beta_wake (calculated) = {np.degrees(beta_exit):.3f} deg.")
            print(f"Wake direction - Convergence Failed after {k+1} iterations.")

        # Relaxed update of wake direction
        beta_wake_init = beta_wake + (relax * delta_beta)
    
    # << end of iteration >>>
   
    return gamma_b, beta_wake, beta_wake_history

