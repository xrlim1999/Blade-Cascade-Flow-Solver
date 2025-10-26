import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator, CubicSpline

import matplotlib
from matplotlib.path import Path
matplotlib.use('Qt5Agg')  # or 'Qt6Agg' if you have PyQt6
import matplotlib.pyplot as plt


def airfoil_visualisation(geom, plotsize: tuple[float, float] = (8, 6), plot_save=True):

    print("Starting airfoil plot...")

    # --- Unpack values ---
    x        = geom["x"]
    y        = geom["y"]
    beta_deg = np.rad2deg(geom["beta"])
    chord    = geom["chord"]

    plt.figure(figsize=plotsize)

    # --- plot ---
    # Draw smooth closed airfoil outline
    plt.plot(x, y, 'k-', linewidth=1.2)

    # Fill the inside with solid black
    plt.fill(x, y, color='black', zorder=3)

    # -- plot boundaries ---
    vert_size = np.max(y)-np.min(y)

    x_buffer = chord*0.1
    y_buffer = vert_size*0.1

    min_x = np.min(x)-x_buffer
    max_x = np.max(x)+x_buffer

    min_y = np.min(y) - y_buffer
    max_y = np.max(y) + y_buffer

    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)

    # --- labels and titles ---
    plt.gca().set_aspect('equal')
    plt.xlabel('X (normalised by chord)')
    plt.ylabel('Y (normalised by chord)')
    title = 'Aerofoil Plot'
    plt.title(title)

    print("    --> airfoil plot render completed.")

    if plot_save:
        # --- create a folder if it doesn't exist ---
        out_dir = "figures/airfoil"
        os.makedirs(out_dir, exist_ok=True)

        # --- save to file ---
        filename = f"airfoil_beta{beta_deg:.0f}.png"
        filepath = os.path.join(out_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight') # high-res, cropped
        plt.close()

        print(f"    --> plot saved in folder: {out_dir}")

    print("Airfoil plot completed.\n")
    
    plt.show()


def smooth_curve(x, y, n_samp=1000):
    """
    Return a visually smooth curve through (x,y) using cubic splines in arclength.
    """
    x = np.asarray(x).ravel(); y = np.asarray(y).ravel()
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]

    # Guard against repeated points
    keep = np.r_[True, np.diff(s) > 1e-12]
    s, x, y = s[keep], x[keep], y[keep]
    csx = CubicSpline(s, x, bc_type='natural')
    csy = CubicSpline(s, y, bc_type='natural')
    S   = np.linspace(s[0], s[-1], n_samp)

    return csx(S), csy(S)


def plot_flow(xnew, ynew, U, V, X, Y,
                n_particles, trajectories, alpha, beta,
                track_particle: bool = True, track_velocity: bool = False,
                plot_save: bool = True):
    
    # --- unpack values ---
    alpha = np.rad2deg(alpha)
    beta  = np.rad2deg(beta)

    print("Starting flow plot...")

    # closed copy for visuals
    x_closed = np.r_[xnew, xnew[0]]
    y_closed = np.r_[ynew, ynew[0]]
    xs, ys   = smooth_curve(x_closed, y_closed, n_samp=1200)

    # --- plot velocity field ---
    if track_velocity is True:

        plt.figure(figsize=(8,6))

        # --- plot Object ---
        # Draw smooth closed airfoil outline
        plt.plot(xs, ys, 'k-', linewidth=1.2)

        # Fill the inside with solid black
        plt.fill(xs, ys, color='black', zorder=3)

        U_plot = np.nan_to_num(U, nan=0.0)
        V_plot = np.nan_to_num(V, nan=0.0)
        plt.streamplot(X, Y, U_plot, V_plot, density=2.0, color='lightblue')
        
        filename = f"velocity_alpha_{alpha:.1f}_beta{beta:.1f}.png"

        # --- plot parameters ---
        plt.xlim(X.min(), X.max())
        plt.ylim(Y.min(), Y.max())
        plt.gca().set_aspect('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        title = f'2D potential flow with freestream angle = {alpha:.1f}\u00b0 | Blade title angle = {beta:.1f}\u00b0'
        plt.title(title)

        if plot_save:
            # --- create a folder if it doesn't exist ---
            out_dir = "figures/flow"
            os.makedirs(out_dir, exist_ok=True)

            # --- save to file ---
            filepath = os.path.join(out_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight') # high-res, cropped
            plt.close()

            print(f"    --> particle trajectory plot saved in folder: {out_dir}")

    # --- plot particle trajectory ---
    if track_particle is True:

        plt.figure(figsize=(8,6))

        # --- plot Object ---
        # Draw smooth closed airfoil outline
        plt.plot(xs, ys, 'k-', linewidth=0.8)

        # Fill the inside with solid black
        plt.fill(xs, ys, color='black', zorder=3)

        for p in range(n_particles):
            traj = trajectories[p,:,:]
            # only plot points inside grid
            mask = (traj[:,0] >= X.min()) & (traj[:,0] <= X.max()) & \
                (traj[:,1] >= Y.min()) & (traj[:,1] <= Y.max())
            plt.plot(traj[mask,0], traj[mask,1], 'blue')
        
        filename = f"particle_alpha_{alpha:.1f}_beta{beta:.1f}.png"

        # --- plot parameters ---
        plt.xlim(X.min(), X.max())
        plt.ylim(Y.min(), Y.max())
        plt.gca().set_aspect('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        title = f'2D potential flow with freestream angle = {alpha:.1f}\u00b0 | Blade title angle = {beta:.1f}\u00b0'
        plt.title(title)

        if plot_save:
            # --- create a folder if it doesn't exist ---
            out_dir = "figures/flow"
            os.makedirs(out_dir, exist_ok=True)

            # --- save to file ---
            filepath = os.path.join(out_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight') # high-res, cropped
            plt.close()

            print(f"    --> velocity field plot saved in folder: {out_dir}")

    print("Flow plot completed.\n")

    # <<< end of flow plot function >>>


def flow_visualisation(geom: dict, flow: dict, vorticity, 
                       flowplot_params,
                       track_particle: bool = True, track_velocity: bool = False,
                       plot_save=True):
    """
    Plot flow around the aerofoil and return (X, Y, U, V, trajectories).
    """

    print("Starting flow computations...")
    
    # --- Unpack values---
    x     = geom["x"]
    y     = geom["y"]
    xmid  = geom["xmid"]
    ymid  = geom["ymid"]
    chord = geom["chord"]
    beta  = geom["beta"]
    ds    = geom["ds"]
    m     = geom["n_panels"]

    alpha = flow["alpha_1"]
    W     = flow["W"]

    np_x, np_y  = flowplot_params["np_x"], flowplot_params["np_y"]
    n_particles = flowplot_params["n_particles"]
    dt          = flowplot_params["dt"]
    n_steps     = flowplot_params["n_steps"]

    """ Helper Functions """
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
        # print(f"area2 = {area2:.2f}")
        if area2 < 0:  # CW → flip to CCW
            x = x[::-1]; y = y[::-1]
        
        return x, y

    def panels_from_geometry(geom):

        x = geom["x"]; y = geom["y"]
        x0, y0 = x[:-1], y[:-1]
        x1, y1 = x[1:],  y[1:]
        dx, dy = x1-x0, y1-y0

        S      = geom["ds"]

        tx     = geom["cosine"]
        ty     = geom["sine"]

        nx, ny =  ty, -tx

        return dict(x0=x0,y0=y0,x1=x1,y1=y1,dx=dx,dy=dy,S=S,tx=tx,ty=ty,nx=nx,ny=ny,
                    xmid=geom["xmid"], ymid=geom["ymid"])

    def make_airfoil_path(x, y):
        """Matplotlib Path for point-in-polygon tests."""
        # np.c_ --> horizontally concantate x and y
        return Path(np.c_[x, y])

    def mask_inside_field(X, Y, path):
        """Return boolean mask of points inside the airfoil (same shape as X,Y)."""
        pts = np.c_[X.ravel(), Y.ravel()]
        return path.contains_points(pts, radius=-1e-4).reshape(X.shape)

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

    """ Build path & panel geometry (once) """
    xnew, ynew = ensure_ccw_closed(x, y)
    panels = panels_from_geometry(geom)
    airfoil_path = make_airfoil_path(xnew, ynew)

    """ Define 2D grid for velocity field """
    # --- enlarge computational field (initial) ---
    x_min, x_max = np.min(xnew)-(chord*1.0), np.max(xnew)+(chord*1.5)
    y_min, y_max = np.min(ynew)-(chord*0.5), np.max(ynew)+(chord*0.5)

    # --- (account for)) change in aerofoil-tilt angle ---
    if beta > 0.0:
        y_add_top_1 = + (np.tan(beta) * chord) * 1.0
        y_add_bot_1 = - 0.0

    elif beta < 0.0:
        y_add_top_1 = 0.0
        y_add_bot_1 = - np.tan(beta) * chord * 1.0

    else:
        y_add_top_1 = 0.0
        y_add_bot_1 = 0.0

    # --- (account for) airflow angle ---
    x_length = x_max - x_min
    y_delta  = abs(np.tan(alpha) * x_length)

    if alpha > 0.0:
        y_add_top_2 = + y_delta*2.5
        y_add_bot_2 = - y_delta*1.5

    elif alpha < 0.0:
        y_add_top_2 = + y_delta*1.5
        y_add_bot_2 = - y_delta*2.5
    else:
        y_add_top_2 = 0.0
        y_add_bot_2 = 0.0
    
    y_min += y_add_bot_1 + y_add_bot_2
    y_max += y_add_top_1 + y_add_top_2

    X, Y = np.meshgrid(np.linspace(x_min, x_max, np_x),
                       np.linspace(y_min, y_max, np_y))

    """
    (compute) Velocity Field
        Keeps your original panel summation; just removes the cylinder mask
    """
    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    for i in range(X.shape[1]): # rows
        for j in range(X.shape[0]): # columns
            # Skip inside-airfoil points (optional early continue for speed)
            if airfoil_path.contains_point((X[j,i], Y[j,i])):
                U[j,i] = np.nan; V[j,i] = np.nan
                continue
                
            # --- freestream velocity ---
            u = W * np.cos(alpha)
            v = W * np.sin(alpha)

            # --- influence of vortex panels at midpoints (xmid, ymid) with strengths vorticity[k] ---
            for k in range(m):
                dx = X[j,i] - xmid[k]
                dy = Y[j,i] - ymid[k]
                r2 = dx*dx + dy*dy

                core = 5e-4 * float(np.mean(ds))       # ds from geom
                r2 = max(r2, core*core)

                u += -vorticity[k] * dy * ds[k] / (2*np.pi*r2)
                v +=  vorticity[k] * dx * ds[k] / (2*np.pi*r2)

            U[j,i] = u
            V[j,i] = v

    # --- Apply mask once, after the loop ---
    U_masked = U.copy(); V_masked = V.copy()
    inside = mask_inside_field(X, Y, airfoil_path)

    U_masked[inside] = np.nan
    V_masked[inside] = np.nan

    """ 
    (initialise) Particle starting positions 
    """
    # --- set start x and y positions ---
    y_start  = np.linspace(y_min, y_max, n_particles)
    x_start = np.full(n_particles, x_min) # create an array of size(n_particles) and each filled with (x_min) value

    particles = np.vstack([x_start, y_start]).T  # particle positions at each timestep [ shape (n_particles, 2) ]

    # --- Interpolators (note axes order: (y,x)) ---
    x_axis, y_axis = X[0, :], Y[:, 0]
    U_i = U_masked
    V_i = V_masked

    u_interp = RegularGridInterpolator((y_axis, x_axis), U_i, bounds_error=False, fill_value=np.nan)
    v_interp = RegularGridInterpolator((y_axis, x_axis), V_i, bounds_error=False, fill_value=np.nan)

    """ 
    Particle Trajectory (compute)
      RK2 midpoint 
    """
    trajectories = np.zeros((n_particles, n_steps, 2))
    trajectories[:, 0, :] = particles  # (x,y) starts

    point_in_airfoil_ = point_in_airfoil
    project_point_to_surface_ = project_point_to_surface
    u_interp_ = u_interp
    v_interp_ = v_interp
    airfoil_path_ = airfoil_path
    panels_ = panels

    half_dt = 0.5 * dt
    sixth = 1.0 / 6.0
    vel = np.empty(2)

    for t in range(1, n_steps):

        for p in range(n_particles):

            pos = trajectories[p, t-1, :]

            # If currently inside (can happen after previous step), push out
            if point_in_airfoil_(pos, airfoil_path_):
                pos = project_point_to_surface_(pos, panels_)
                trajectories[p, t-1, :] = pos

            # velocity at current position
            u1 = u_interp_([pos[1], pos[0]])[0]
            v1 = v_interp_([pos[1], pos[0]])[0]
            vel[:] = (u1, v1)

            # midpoint predictor
            mid1 = pos + half_dt*vel
            # project midpoint back to surface if within aerofoil
            if point_in_airfoil_(mid1, airfoil_path_):
                mid1 = project_point_to_surface_(mid1, panels_)

            u2 = u_interp_([pos[1], pos[0]])[0]
            v2 = v_interp_([pos[1], pos[0]])[0]
            vel[:] = (u2, v2)

            # midpoint predictor
            mid2 = pos + half_dt*vel
            # project midpoint back to surface if within aerofoil
            if point_in_airfoil_(mid2, airfoil_path_):
                mid2 = project_point_to_surface_(mid2, panels_)

            u3 = u_interp_([pos[1], pos[0]])[0]
            v3 = v_interp_([pos[1], pos[0]])[0]
            vel[:] = (u3, v3)

            # midpoint predictor
            mid3 = pos + dt*vel
            # project midpoint back to surface if within aerofoil
            if point_in_airfoil_(mid3, airfoil_path_):
                mid3 = project_point_to_surface_(mid3, panels_)

            u4 = u_interp_([pos[1], pos[0]])[0]
            v4 = v_interp_([pos[1], pos[0]])[0]

            uf = sixth*(u1 + 2.0*u2 + 2.0*u3 + u4)
            vf = sixth*(v1 + 2.0*v2 + 2.0*v3 + v4)

            new_pos = pos + dt*np.array([uf, vf])

            # enforce no-penetration after step
            if point_in_airfoil_(new_pos, airfoil_path_):
                new_pos = project_point_to_surface_(new_pos, panels_)

            trajectories[p, t, :] = new_pos

        if t % (n_steps // 10) == 0:  # every 1%
            t_perc = int(t / n_steps * 100.0)
            print(f"    time-stepping {t_perc}% completed")
    
    flowfield = dict()
    flowfield["Trajectories"] = trajectories
    flowfield["U"]            = U
    flowfield["V"]            = V
    flowfield["X"]            = X
    flowfield["Y"]            = Y

    print("Flow computations completed.\n")
        
    # --- plot flow visualisation ---
    plot_flow(xnew, ynew, U, V, X, Y,
                n_particles, trajectories, alpha, beta,
                track_particle=track_particle, track_velocity=track_velocity,
                plot_save=plot_save)
    
    return flowfield

    # <<< end of flow plot computations >>>



