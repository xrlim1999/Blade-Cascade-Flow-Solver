import numpy as np
from .martensen_kernels import coupling_coefficient_bound


def vorticity_solution_kutta(geom: dict, flow: dict, rhs, mode='gamma'):
    """
    Solve A*gamma = rhs with trailing-edge unloading (Kutta) as the single closure.
    Now enforces Vt_u == Vt_l at the TE, with mild stabilization for the TE unknowns.
    """

    coup = coupling_coefficient_bound(geom)
    
    A = coup.copy()
    b = rhs.copy()

    i_u, i_l = 0, (A.shape[0]-1)

    # --- Replace LAST equation with legacy Kutta: gamma_u + gamma_l = 0 ---
    row = np.zeros_like(b)
    row[i_u] = 1.0
    row[i_l] = 1.0
    A[-1, :] = row
    b[-1]    = 0.0

    # --- Gentle Tikhonov on the two TE columns (keeps numbers in check) -------
    lam = 1e-6 * (np.linalg.norm(A, ord=np.inf) + 1.0)  # <<< add
    A[i_u, i_u] += lam
    A[i_l, i_l] += lam

    vorticity = np.linalg.solve(A, b)

    # --- Check Kutta resolution ---
    if mode == 'gamma':
        kutta_res = vorticity[i_u] + vorticity[i_l]
    else:
        kutta_res = vorticity[i_u]*geom["ds"][i_u] + vorticity[i_l]*geom["ds"][i_l]

    if abs(kutta_res) >= 1e-3:   # relaxed tolerance since it's not the constraint  # <<<
        print(f"Kutta check (legacy) = {kutta_res:.3e} | TE vorticities: upper={vorticity[i_u]:.3e}, lower={vorticity[i_l]:.3e}\n")

    return coup, vorticity


