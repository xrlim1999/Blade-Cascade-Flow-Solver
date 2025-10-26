import numpy as np


def vorticity_solution_kutta_rotor(coup, rhs_U, rhs_V, rhs_Omega, ds, mode='circulation', tol=1e-10):
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
    A[-1, :] = row

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

def vorticity_solution_kutta_stator(coup, rhs_U, rhs_V, ds, mode='circulation', tol=1e-10):
    """
    Solve A*gamma = rhs with -edge unloading (Kutta) as the single closure.
    mode:
    - 'gamma'        -> enforce gamma_upper + gamma_lower = 0     (what you asked)
    - 'circulation'  -> enforce gamma_u*ds_u + gamma_l*ds_l = 0   (length-weighted)
    """
    A = coup.copy()

    b_U = rhs_U.copy() ; b_U[-1] = 0.0
    b_V = rhs_V.copy() ; b_V[-1] = 0.0

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

    # tiny ridge for tough geometries (optional)
    A[-1, -1] += 1e-14

    vorticity_U = np.linalg.solve(A, b_U)
    vorticity_V = np.linalg.solve(A, b_V)

    # report
    if mode == 'gamma':
        r_U     = vorticity_U[i_u] + vorticity_U[i_l]
        r_V     = vorticity_V[i_u] + vorticity_V[i_l]
    else:
        r_U = vorticity_U[i_u]*ds[i_u] + vorticity_U[i_l]*ds[i_l]
        r_V = vorticity_V[i_u]*ds[i_u] + vorticity_V[i_l]*ds[i_l]
    
    # Report if any violate tolerance
    def _fmt(arr): return f"upper={arr[i_u]:.3e}, lower={arr[i_l]:.3e}"
    if (abs(r_U) > tol) or (abs(r_V) > tol):
        print("Kutta check - FAILED")
        print(f"  residuals ({mode}): U={r_U:.3e}, V={r_V:.3e}")
        print(f"  TE indices: upper={i_u}, lower={i_l}")
        print(f"  TE vorticities: U: {_fmt(vorticity_U)} | V: {_fmt(vorticity_V)}")

    return vorticity_U, vorticity_V
