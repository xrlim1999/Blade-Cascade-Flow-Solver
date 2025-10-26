import math
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from scipy.interpolate import splprep, splev
from scipy.interpolate import interp1d, Akima1DInterpolator

import matplotlib
from matplotlib.path import Path
matplotlib.use('Qt5Agg')  # or 'Qt6Agg' if you have PyQt6
import matplotlib.pyplot as plt

"""
Load aerofoil data
"""
def data_load(filename):

    data = np.loadtxt(filename)

    # Split into x and y coords
    xdata = data[:,0]
    ydata = data[:,1]

    return xdata, ydata

"""
Data Preparation
"""
def rearrange_airfoil(x, y):
    """
    Reorganise airfoil coordinates such this is:
        - TE --> upper --> LE --> lower --> TE
        - TE is closed/joined
    """
    # Ensure trailing edge closure
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    # Leading edge index (minimum x)
    idx_te = np.argmax(x)

    # Split surfaces: TE->LE (upper), LE->TE (lower)
    x_upper = x[:idx_te+1]; y_upper = y[:idx_te+1]  # TE->LE
    x_lower = x[idx_te+1:]; y_lower = y[idx_te+1:]  # LE->TE

    # Make the trailing edge y-coords the same
    x_te_avg = x_upper[-1]
    y_te_avg = (y_upper[-1] + y_lower[-1]) / 2
    if y_upper[-1] != y_te_avg:
        x_loop = np.r_[x_te_avg, x_lower[::-1], x_upper[1:], x_te_avg]
        y_loop = np.r_[y_te_avg, y_lower[::-1], y_upper[1:], y_te_avg]
    else:
        x_loop = np.r_[x_upper[-1], x_lower[::-1], x_upper[1:]]
        y_loop = np.r_[y_upper[-1], y_lower[::-1], y_upper[1:]]

    # Optionally normalize to chord [0,1] (comment out if your data already is)
    x_min, x_max = x_loop.min(), x_loop.max()
    chord = x_max - x_min
    xnew = (x_loop - x_min) / chord
    ynew = y_loop / chord  # keeps aspect ratios consistent if you normalized x

    x_le = x_upper[0]; y_le = y_upper[0]
    x_te = x_loop[0] ; y_te = y_loop[0]
    chord = x_te - x_le

    return xnew, ynew, x_le, y_le, x_te, y_te, chord

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

def resample_airfoil_cosine_finite_TE(x_loop, y_loop, n_points=200):
    """
    Resample a closed loop (TE→upper→LE→lower→TE) with a *finite-thickness TE*.
    Returns n_panels+1 nodes; last node duplicates the first (upper-TE) to close.
    """
    x = np.asarray(x_loop).ravel()
    y = np.asarray(y_loop).ravel()

    # ensure closed exactly once
    if not (np.isclose(x[0], x[-1]) and np.isclose(y[0], y[-1])):
        x = np.r_[x, x[0]]; y = np.r_[y, y[0]]
    # else:
        # x[-1], y[-1] = x[0], y[0]

    # find LE (minimum x); split: TE→LE (upper), LE→TE (lower)
    i_le = int(np.argmin(x))
    x_upper, y_upper = x[:i_le+1], y[:i_le+1]   # TE(upper) → LE
    x_lower, y_lower = x[i_le:], y[i_le:]   # LE → TE(lower), exclude final dup TE

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

    # --- finite-thickness TE: keep distinct TE y's, do NOT average/collapse ---
    # assemble TE→upper→LE→lower→TE:
    # start at TE(upper)=yu[-1], traverse to LE, then to TE(lower)=yl[-1],
    # and finally close back to TE(upper)=yu[-1] with a short vertical panel.
    x_new = np.r_[x_cos[::-1], x_cos[1:]]  # [1..0] + [0..1] + close at 1
    y_new = np.r_[yu[::-1],     yl[1:]  ]  # close to *upper* TE value

    return x_new, y_new

def rotate_about_point(x, y, xc, yc, beta, counterclockwise_positive=True):
    """Rigid rotation of all nodes about (xc,yc)."""
    if not counterclockwise_positive:
        beta = -beta

    c, s = np.cos(beta), np.sin(beta)

    x = np.asarray(x); y = np.asarray(y)
    xr = xc + (x - xc)*c - (y - yc)*s
    yr = yc + (x - xc)*s + (y - yc)*c

    return xr, yr

def rotate_airfoil_about_te(x, y, beta_deg, pivot='mid'):
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

    return xr, yr

def data_preparation(x, y, n_datapoints):
    global ds, sine, cosine, slope, xmid, ymid

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


"""
Coupling Coefficients
"""
def apply_back_diagonal_correction(n_passes=2):
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


def coupling_coefficient(n_midpoints):
    global coup, slope

    m = n_midpoints
    
    coup = np.zeros((m,m))

    slope = np.unwrap(slope)  # <-- make it continuous around the loop

    # compute self-inducing coupling coefficients
    coup[0,0] = - 0.5 - (slope[1] - slope[m-1] - 2.0*np.pi) / (8.0*np.pi)
    coup[m-1,m-1] = - 0.5 - (slope[0] - slope[m-2] - 2.0*np.pi) / (8.0*np.pi)

    def idx(k):
        return k % m

    for i in range(m):
        dtheta = slope[idx(i+1)] - slope[idx(i-1)]
        coup[i,i] = -0.5 - dtheta / (8.0*np.pi)   # no -2π needed

    # compute coupling coefficients for j <> i
    for i in range(0, m):
        for j in range(0, m):
            if j != i:
                dx = xmid[j] - xmid[i]
                dy = ymid[j] - ymid[i]
                r2 = dx**2 + dy**2

                core = 0.2 * ds[j] # 10–30% of panel j length works well
                if r2 < core*core:
                    print("small r2")
                    r2 = core*core

                u =  dy / (2.0*np.pi * r2)
                v = -dx / (2.0*np.pi * r2)

                coup[j,i] =  (u * cosine[j] + v * sine[j]) * ds[i]
                coup[i,j] = -(u * cosine[i] + v * sine[i]) * ds[j]


    # (optional) quick sanity check
    # assert len(ds) == len(xmid) == len(ymid) == m
    # assert np.all(np.isfinite(ds)) and np.all(ds > 0.0)
    # assert np.all(np.isfinite(coup))

    # compute Back Diagonal Correction (BCD)
    apply_back_diagonal_correction()

    # ensure BCD was computed correctly
    col_resid = np.abs(coup.T @ ds)
    if col_resid.max() < 1e-14:
        print("BDC check - PASSED")
    else:
        print("BDC check - FAILED")
        print("BDC max col residual (should be ~0): ", col_resid.max())
        print("BDC sum col residual (should be ~0): ", col_resid.sum())
    print("")

"""
Calculate Right-Hand Side values
"""
def right_hand_side():
    global rhs
    rhs = np.zeros(m)

    for i in range(m):
        rhs[i] = -W * (math.cos(alpha)*math.cos(slope[i]) + math.sin(alpha)*math.sin(slope[i]))


"""
Solve for surface vorticity element strengths
"""
def vorticity_solution_kutta(mode='gamma'):
    """
    Solve A*gamma = rhs with trailing-edge unloading (Kutta) as the single closure.
    mode:
      - 'gamma'        -> enforce gamma_upper + gamma_lower = 0     (what you asked)
      - 'circulation'  -> enforce gamma_u*ds_u + gamma_l*ds_l = 0   (length-weighted)
    """
    global vorticity
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
    
    if kutta_res < 1e-14:
        print("Kutta check - PASSED")
    else:
        print("Kutta check - FAILED")
        print(f"Kutta residual ({mode}) = {kutta_res:.3e}  |  TE indices: upper={i_u}, lower={i_l} | TE vorticities: upper={vorticity[i_u]}, lower={vorticity[i_l]}")
    print("")


"""
Aerofoil performance metrics
"""
def airfoil_performance(plot_cp=False):
    global chord

    circulation = np.sum(vorticity * ds)
    CL = circulation / (W * chord)

    Cp = np.zeros_like(vorticity)
    Cp = 1 - (vorticity/W)**2
    if plot_cp:
        plt.figure(figsize=(8,6))
        plt.plot(xmid, Cp, 'k-', linewidth=1.2)
        plt.gca().invert_yaxis()
        plt.xlabel('x')
        plt.ylabel('Pressure Coefficient (Cp)')
        plt.title("Pressure Coefficient around aerofoil")

        plt.grid(True, which='both', linestyle='--', alpha=0.6)

        plt.show()

    print(f"========== Results ==========\n")
    print(f"Freestream velocity        = {W:.0f} m/s")
    print(f"Freestream angle-of-attack = {np.rad2deg(alpha):.1f} deg")
    print(f"Blade tilt angle           = {np.rad2deg(beta):.1f} deg (+ve means downwards tile)")
    print(f"Effective angle-of-attack  = {np.rad2deg(alpha-beta):.1f} deg (+ve means downwards tile)")
    print(f"Circulation                = {circulation:.2f} m^2/s")
    print(f"Lift coefficient           = {CL:.2f}")
    print(f"\n========== End ==========\n")


"""
Plot flow around the aerofoil
"""
def flow_visualisation():
    global xnew, ynew, chord
    global X, Y, U, V
    global n_particles, trajectories
    global alpha, beta

    print("Starting flow computations...\n")

    def ensure_ccw_closed(x, y):
        """Ensure a closed loop (first==last) and Couter-Clockwise (CCW) orientation."""
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()

        # 1. Ensure closure (first point == last point)
        if not (np.isclose(x[0], x[-1]) and np.isclose(y[0], y[-1])):
            x = np.r_[x, x[0]]
            y = np.r_[y, y[0]]

        # 2. Compute signed area using shoelace formula
        area2 = np.sum(x[:-1]*y[1:] - x[1:]*y[:-1])
        if area2 < 0:  # CW → flip to CCW
            x = x[::-1]; y = y[::-1]
        
        return x, y

    def build_panel_geometry(x, y):
        """
        From CLOSED CCW nodes x,y (len M+1), build panel endpoints and outward normals.
        Returns dict with x0,y0,x1,y1, dx,dy, S, tx,ty, nx,ny
        """
        x0, y0 = x[:-1], y[:-1]     # panel starts
        x1, y1 = x[ 1:], y[ 1:]     # panel ends

        dx = x1 - x0; dy = y1 - y0  
        S  = np.hypot(dx, dy)       # panel lengths

        # check if any panels are duplicated or too small value
        if np.any(S <= 1e-14):
            raise ValueError("Zero-length panel detected (duplicate nodes).")
        
        tx = dx / S; ty = dy / S # unit tangent vectors

        # CCW outward normal (rotate right -90 degrees)
        nx =  ty                 
        ny = -tx

        return dict(x0=x0, y0=y0, x1=x1, y1=y1, dx=dx, dy=dy, S=S, tx=tx, ty=ty, nx=nx, ny=ny)

    def make_airfoil_path(x, y):
        """Matplotlib Path for point-in-polygon tests."""
        # np.c_ --> horizontally concantate x and y
        return Path(np.c_[x, y])

    def mask_inside_field(X, Y, path):
        """Return boolean mask of points inside the airfoil (same shape as X,Y)."""
        pts = np.c_[X.ravel(), Y.ravel()]
        inside = path.contains_points(pts).reshape(X.shape)
        return inside

    def project_point_to_surface(pt, panels, eps=1e-6):
        """
        Project a point onto the nearest panel segment, then nudge outward along that panel's normal.
        pt: (2,) array-like
        panels: dict from build_panel_geometry
        """
        px, py = pt
        x0, y0 = panels['x0'], panels['y0']
        dx, dy = panels['dx'], panels['dy']
        S      = panels['S']
        nx, ny = panels['nx'], panels['ny']

        # ===================================
        # Find closest panel to current point
        # ===================================
        # vector from each start-of-panel to current point
        rx = px - x0
        ry = py - y0

        # param along segment, clamped [0,1]
        t = (rx*dx + ry*dy) / (S*S) # dot-product of r & d
        t = np.clip(t, 0.0, 1.0)    # fix <t> to be 0 or 1 if it goes out of range

        # closest points on segments
        cx = x0 + t*dx
        cy = y0 + t*dy

        # pick nearest segmet
        d2 = (px - cx)**2 + (py - cy)**2
        k  = np.argmin(d2)

        # outward nudge + additional "safety" gap of eps in normal direction
        return np.array([cx[k] + eps*nx[k], cy[k] + eps*ny[k]])

    def point_in_airfoil(pt, path):
        """ Check if point is within airfoil """
        # point in airfoil: True
        # point not in airfoil: False
        return path.contains_point((pt[0], pt[1]))

    # ============================================
    # Build path & panel geometry (once)
    # ============================================
    xnew, ynew = ensure_ccw_closed(xnew, ynew)
    panels = build_panel_geometry(xnew, ynew)
    airfoil_path = make_airfoil_path(xnew, ynew)

    # --------------------------
    # Define 2D grid for velocity field
    # --------------------------
    x_min, x_max = x_le-(chord*1.0), x_te+(chord*1.0)
    y_min, y_max = y_te-(chord*1.0), y_te+(chord*1.0)
    # Account for change in aerofoil-tilt angle
    if beta > 0.0:
        y_min = y_min - (np.tan(beta)*chord)*1.0

    elif beta < 0.0:
        y_max = y_max + (np.tan(beta)*chord)*1.0

    nx, ny = 200, 200

    X, Y = np.meshgrid(np.linspace(x_min, x_max, nx),
                    np.linspace(y_min, y_max, ny))

    # ============================================
    # Compute velocity field on grid, then mask inside airfoil
    # (Keeps your original panel summation; just removes the cylinder mask)
    # ============================================
    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    for i in range(X.shape[1]): # rows
        for j in range(X.shape[0]): # columns
            # Skip inside-airfoil points (optional early continue for speed)
            if airfoil_path.contains_point((X[j,i], Y[j,i])):
                U[j,i] = np.nan; V[j,i] = np.nan
                continue

            u = W * np.cos(alpha)
            v = W * np.sin(alpha)

            # influence of vortex panels at midpoints (xmid, ymid) with strengths vorticity[k]
            for k in range(m):
                dx = X[j,i] - xmid[k]
                dy = Y[j,i] - ymid[k]
                r2 = dx*dx + dy*dy
                if r2 < 1e-12:
                    r2 = 1e-12
                u +=  vorticity[k] * dy * ds[k] / (2*np.pi*r2)
                v += -vorticity[k] * dx * ds[k] / (2*np.pi*r2)

            U[j,i] = u
            V[j,i] = v

    # (If you didn’t early-continue above, you can mask now:)
    # inside = mask_inside_field(X, Y, airfoil_path)
    # U[inside] = np.nan; V[inside] = np.nan; Q[inside] = np.nan

    # --------------------------
    # Particle starting positions
    # --------------------------
    n_particles = 121 # number of starting particles to track

    # determine y_range of starting particles
    x_length = x_max - x_min
    y_length = y_max - y_min
    y_delta  = abs(np.tan(alpha) * x_length)
    # airflow angle
    if alpha == 0.0:
        y_max = y_max + (y_length*0.5)
        y_min = y_min - (y_length*0.5)

    elif alpha > 0.0:
        y_max = y_max + y_delta*1.5
        y_min = y_min - y_delta*2.5

    else:
        y_max = y_max + y_delta*2.5
        y_min = y_min - y_delta*1.5

    
    # start x and y positions
    y_start  = np.linspace(y_min, y_max, n_particles)
    x_start = np.full(n_particles, x_min) # create an array of size(n_particles) and each filled with (x_min) value

    particles = np.vstack([x_start, y_start]).T  # particle positions at each timestep [ shape (n_particles, 2) ]

    # ============================================
    # Interpolators (note axes order: (y,x))
    # ============================================
    y_axis = Y[:, 0]
    x_axis = X[0, :]
    u_interp = RegularGridInterpolator((y_axis, x_axis), U, bounds_error=False, fill_value=None)
    v_interp = RegularGridInterpolator((y_axis, x_axis), V, bounds_error=False, fill_value=None)

    # ============================================
    # Particle advection with no-penetration (RK2 midpoint)
    # ============================================
    dt = 0.01
    n_steps = 500
    trajectories = np.zeros((n_particles, n_steps, 2))
    trajectories[:, 0, :] = particles  # (x,y) starts

    for t in range(1, n_steps):
        for p in range(n_particles):
            pos = trajectories[p, t-1, :]

            # If currently inside (can happen after previous step), push out
            if point_in_airfoil(pos, airfoil_path):
                pos = project_point_to_surface(pos, panels)
                trajectories[p, t-1, :] = pos

            # velocity at current position
            u1 = u_interp([pos[1], pos[0]])[0]
            v1 = v_interp([pos[1], pos[0]])[0]

            # midpoint predictor
            mid = pos + 0.5*dt*np.array([u1, v1])
            # project midpoint back to surface if within aerofoil
            if point_in_airfoil(mid, airfoil_path):
                mid = project_point_to_surface(mid, panels)

            u2 = u_interp([mid[1], mid[0]])[0]
            v2 = v_interp([mid[1], mid[0]])[0]

            new_pos = pos + dt*np.array([u2, v2])

            # enforce no-penetration after step
            if point_in_airfoil(new_pos, airfoil_path):
                new_pos = project_point_to_surface(new_pos, panels)

            trajectories[p, t, :] = new_pos


"""
Plot Results
"""
def plot_flow(object=True, track_velocity=True, track_particle=True):
    global n_particles, trajectories

    print("Starting flow visualisation plot...\n")


    plt.figure(figsize=(8,6))

    # Plot the object
    if object is True:
        # Draw smooth closed airfoil outline
        plt.plot(xnew, ynew, 'k-', linewidth=1.2)

        # Fill the inside with solid black
        plt.fill(xnew, ynew, color='black', zorder=3)

    # Plot velocity field
    if track_velocity is True:
        U_plot = np.nan_to_num(U, nan=0.0)
        V_plot = np.nan_to_num(V, nan=0.0)
        plt.streamplot(X, Y, U_plot, V_plot, density=1.5, color='lightblue')

    # Plot particle trajectory
    if track_particle is True:
        for p in range(n_particles):
            traj = trajectories[p,:,:]
            # only plot points inside grid
            mask = (traj[:,0] >= X.min()) & (traj[:,0] <= X.max()) & \
                (traj[:,1] >= Y.min()) & (traj[:,1] <= Y.max())
            plt.plot(traj[mask,0], traj[mask,1], 'blue')

    # plt.xlim(X.min(), X.max())
    # plt.ylim(Y.min(), Y.max())
    plt.gca().set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    title = f'2D potential flow with freestream angle = {np.degrees(alpha):.1f} deg | Blade title angle = {np.degrees(beta):.1f} deg'
    plt.title(title)
    
    plt.show()


"""
MAIN PROGRAM
"""
## ============
#  Parameters
## =============
W     = 100.0 # freestream velocity [m/s]
rho   = 1.225 # freestream air density [kg/m^3]
alpha_deg = 4.0 # freestream angle-of-attack [deg]
alpha = np.radians(alpha_deg) # freestream angle-of-attack [rad]
beta_deg  = 0.0
beta  = np.radians(beta_deg) # freestream angle-of-attack [rad]
m     = 100  # number of aerofoil profile data input points (m+1 for completing the profile)

x, y = data_load('aerofoil_coords.txt')

xdata, ydata, x_le, y_le, x_te, y_te, chord = rearrange_airfoil(x, y)

xnew, ynew = resample_airfoil_cosine_finite_TE(xdata, ydata, n_points=m)

xnew, ynew = rotate_airfoil_about_te(xnew, ynew, beta, pivot='mid')

data_preparation(xnew, ynew, n_datapoints=xnew.shape[0])

coupling_coefficient(n_midpoints=xmid.shape[0])

right_hand_side()

vorticity_solution_kutta()

airfoil_performance(plot_cp=True)

# flow_visualisation()

# plot_flow(object=True, track_velocity=False, track_particle=False)

print("\nEnd Of Simulation.\n")