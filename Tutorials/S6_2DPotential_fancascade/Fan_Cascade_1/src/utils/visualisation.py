import numpy as np
from scipy.interpolate import RegularGridInterpolator

import matplotlib
from matplotlib.path import Path
matplotlib.use('Qt5Agg')  # or 'Qt6Agg' if you have PyQt6
import matplotlib.pyplot as plt


def airfoil_visualisation(x, y, chord, plotsize: tuple[float, float] = (8, 6)):

    plt.figure(figsize=plotsize)

    plt.plot(x, y, 'k.', markersize=3, zorder=4)

    # fill the inside of aerofoil with solid colour (black)
    # plt.plot(x, y, 'k', linewidth=1.2)
    # plt.fill(x, y, color='black', zorder=3)

    vert_size = np.max(y)-np.min(y)

    # boundaries
    plt.xlim(0, chord)
    plt.ylim(vert_size, -vert_size)

    plt.gca().set_aspect('equal')
    plt.xlabel('X (normalised by chord)')
    plt.ylabel('Y (normalised by chord)')
    title = 'Aerofoil visualisation'
    plt.title(title)
    
    plt.show()


def flow_visualisation(xnew, ynew, xmid, ymid, chord, alpha, beta, W, ds, vorticity, 
                       n_particles=101, dt=0.005, n_steps=500):
    """
    Plot flow around the aerofoil and return (X, Y, U, V, trajectories).
    """

    print("Starting flow computations...\n")

    # -----------------------
    # Helpers (scoped inside)
    # -----------------------
    def ensure_ccw_closed(x, y):
        """Ensure closed loop and CCW orientation."""
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        if not (np.isclose(x[0], x[-1]) and np.isclose(y[0], y[-1])):
            x = np.r_[x, x[0]]
            y = np.r_[y, y[0]]
        area2 = np.sum(x[:-1]*y[1:] - x[1:]*y[:-1])
        if area2 < 0:  # CW → flip to CCW
            x = x[::-1]; y = y[::-1]
        return x, y

    def build_panel_geometry(x, y):
        """From CLOSED CCW nodes, build panel endpoints and outward normals."""
        x0, y0 = x[:-1], y[:-1]
        x1, y1 = x[1:],  y[1:]
        dx, dy = x1 - x0, y1 - y0
        S = np.hypot(dx, dy)
        if np.any(S <= 1e-14):
            raise ValueError("Zero-length panel detected (duplicate nodes).")
        tx, ty = dx / S, dy / S
        nx, ny =  ty, -tx   # CCW outward normal
        return dict(x0=x0, y0=y0, x1=x1, y1=y1, dx=dx, dy=dy, S=S, tx=tx, ty=ty, nx=nx, ny=ny)

    def make_airfoil_path(x, y):
        """Matplotlib Path for point-in-polygon tests."""
        return Path(np.c_[x, y])

    def mask_inside_field(X, Y, path):
        """
        Boolean mask of points inside the airfoil (same shape as X,Y).
        Treat boundary as inside to close any TE gap.
        """
        pts = np.c_[X.ravel(), Y.ravel()]
        return path.contains_points(pts, radius=-1e-9).reshape(X.shape)

    def point_in_airfoil(pt, path):
        """Point-in-polygon check; boundary considered inside."""
        return path.contains_point((pt[0], pt[1]), radius=-1e-9)

    def project_point_to_surface(pt, panels, eps=5e-4):
        """
        Project a point onto the nearest panel, then nudge outward along that panel's normal.
        """
        px, py = pt
        x0, y0 = panels['x0'], panels['y0']
        dx, dy = panels['dx'], panels['dy']
        S      = panels['S']
        nx, ny = panels['nx'], panels['ny']

        rx = px - x0
        ry = py - y0
        t = (rx*dx + ry*dy) / (S*S)
        t = np.clip(t, 0.0, 1.0)

        cx = x0 + t*dx
        cy = y0 + t*dy

        d2 = (px - cx)**2 + (py - cy)**2
        k  = np.argmin(d2)

        return np.array([cx[k] + eps*nx[k], cy[k] + eps*ny[k]])

    # ============================================
    # Build path & panel geometry (once)
    # ============================================
    xnew, ynew   = ensure_ccw_closed(xnew, ynew)
    panels       = build_panel_geometry(xnew, ynew)
    airfoil_path = make_airfoil_path(xnew, ynew)

    # --------------------------
    # Define 2D grid for velocity field
    # --------------------------
    x_le = np.min(xnew); x_te = np.max(xnew)
    y_te = ynew[np.argmax(xnew)]

    x_min, x_max = x_le - (chord*1.0), x_te + (chord*1.0)
    y_min, y_max = y_te - (chord*1.0), y_te + (chord*1.0)

    # Account for airfoil tilt beta
    if beta > 0.0:
        y_min -= np.tan(beta)*chord
    elif beta < 0.0:
        y_max += np.tan(beta)*chord

    nx, ny = 200, 200
    X, Y = np.meshgrid(np.linspace(x_min, x_max, nx),
                       np.linspace(y_min, y_max, ny))

    # ============================================
    # Compute velocity field on grid
    # ============================================
    def panel_velocity_at_point(px, py, panels, gamma, core_eps=1e-12):
        """
        Induced velocity at a field point (px,py) from constant-strength
        vortex panels with strengths gamma[k] (per unit length).
        Uses the analytic formula: u = (gamma/2π)*Δθ * n_hat  for each panel.
        """
        x0 = panels['x0'];  y0 = panels['y0']
        x1 = panels['x1'];  y1 = panels['y1']
        tx = panels['tx'];  ty = panels['ty']   # unit tangent (CCW)
        nx = panels['nx'];  ny = panels['ny']   # unit outward normal (CCW)

        # Vectors from field point to the two endpoints
        rx0 = px - x0;  ry0 = py - y0
        rx1 = px - x1;  ry1 = py - y1

        # Project to each panel's local frame (t,n)
        r0_t = rx0*tx + ry0*ty
        r0_n = rx0*nx + ry0*ny
        r1_t = rx1*tx + ry1*ty
        r1_n = rx1*nx + ry1*ny

        # Avoid atan2(0,0)
        r0_t = np.where((np.abs(r0_t) < core_eps) & (np.abs(r0_n) < core_eps), core_eps, r0_t)
        r0_n = np.where((np.abs(r0_t) < core_eps) & (np.abs(r0_n) < core_eps), core_eps, r0_n)
        r1_t = np.where((np.abs(r1_t) < core_eps) & (np.abs(r1_n) < core_eps), core_eps, r1_t)
        r1_n = np.where((np.abs(r1_t) < core_eps) & (np.abs(r1_n) < core_eps), core_eps, r1_n)

        # Signed angles to each endpoint in the panel's local frame
        th0 = np.arctan2(r0_n, r0_t)
        th1 = np.arctan2(r1_n, r1_t)

        # Δθ in (-π, π]
        dth = th1 - th0
        dth = np.where(dth >  np.pi, dth - 2*np.pi, dth)
        dth = np.where(dth <= -np.pi, dth + 2*np.pi, dth)

        # Sum of panel contributions, along each panel's outward normal
        # NOTE: gamma is per-unit-length strength; do NOT multiply by panel length.
        factor = (gamma / (2.0*np.pi)) * dth
        u = np.sum(factor * nx)
        v = np.sum(factor * ny)
        return u, v

    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    for i in range(X.shape[1]):      # x-index
        for j in range(X.shape[0]):  # y-index
            u = W * np.cos(alpha)
            v = W * np.sin(alpha)

            # # OLD: Influence of vortex panels at midpoints (xmid, ymid)
            # for k in range(vorticity.shape[0]):
            #     dx = X[j, i] - xmid[k]
            #     dy = Y[j, i] - ymid[k]
            #     r2 = dx*dx + dy*dy
            #     if r2 < 1e-10:
            #         r2 = 1e-10
            #     u +=  vorticity[k] * dy * ds[k] / (2*np.pi*r2)
            #     v += -vorticity[k] * dx * ds[k] / (2*np.pi*r2)

            # NEW: analytic straight-panel influence
            du, dv = panel_velocity_at_point(X[j, i], Y[j, i], panels, vorticity)
            u += du;  v += dv

            U[j, i] = u
            V[j, i] = v

    # =========================
    # Mask inside the airfoil
    # =========================
    inside = mask_inside_field(X, Y, airfoil_path)
    U[inside] = np.nan
    V[inside] = np.nan

    # --------------------------
    # Particle starting positions
    # --------------------------
    x_length = x_max - x_min
    y_length = y_max - y_min
    y_delta  = abs(np.tan(alpha) * x_length)

    if alpha == 0.0:
        y_top = y_max + 0.5*y_length
        y_bot = y_min - 0.5*y_length
    elif alpha > 0.0:
        y_top = y_max + 1.5*y_delta
        y_bot = y_min - 2.5*y_delta
    else:
        y_top = y_max + 2.5*y_delta
        y_bot = y_min - 1.5*y_delta

    y_start = np.linspace(y_bot, y_top, n_particles)
    x_start = np.full(n_particles, x_min)
    particles = np.vstack([x_start, y_start]).T  # (n_particles, 2)

    # ============================================
    # Interpolators (note axes order: (y, x))
    # ============================================
    y_axis = Y[:, 0]
    x_axis = X[0, :]
    u_interp = RegularGridInterpolator((y_axis, x_axis), U, bounds_error=False, fill_value=None)
    v_interp = RegularGridInterpolator((y_axis, x_axis), V, bounds_error=False, fill_value=None)

    # ============================================
    # Particle advection with no-penetration (RK2)
    # ============================================
    trajectories = np.zeros((n_particles, n_steps, 2))
    trajectories[:, 0, :] = particles

    for t in range(1, n_steps):
        for p in range(n_particles):
            pos = trajectories[p, t-1, :]

            if point_in_airfoil(pos, airfoil_path):
                pos = project_point_to_surface(pos, panels)
                trajectories[p, t-1, :] = pos

            u1 = u_interp([pos[1], pos[0]])[0]
            v1 = v_interp([pos[1], pos[0]])[0]

            mid = pos + 0.5*dt*np.array([u1, v1])
            if point_in_airfoil(mid, airfoil_path):
                mid = project_point_to_surface(mid, panels)

            u2 = u_interp([mid[1], mid[0]])[0]
            v2 = v_interp([mid[1], mid[0]])[0]

            new_pos = pos + dt*np.array([u2, v2])
            if point_in_airfoil(new_pos, airfoil_path):
                new_pos = project_point_to_surface(new_pos, panels)

            trajectories[p, t, :] = new_pos

    return X, Y, U, V, trajectories


def plot_actual(xnew, ynew, U, V, X, Y,
                n_particles, trajectories, alpha, beta,
                object=True, track_particle=True, track_velocity=True):

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

    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.gca().set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    title = f'2D potential flow with freestream angle = {np.degrees(alpha):.1f} deg | Blade title angle = {np.degrees(beta):.1f} deg'
    plt.title(title)
    
    plt.show()

