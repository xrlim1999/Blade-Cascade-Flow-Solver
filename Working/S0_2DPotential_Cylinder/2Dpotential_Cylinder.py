"""
Solver: Calculation of a flow past a circular cylinder

Flow: 2D incompressible, potential flow

"""

import math
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import matplotlib
matplotlib.use('Qt5Agg')  # or 'Qt6Agg' if you have PyQt6
import matplotlib.pyplot as plt

"""
Define
"""
## Define parameters
m      = 100  # number of aerofoil profile data input points (m+1 for completing the profile)
radius = 0.5 # radius of cylinder [m]
xc, yc = 0.0, 0.0 # center of circle
W      = 1.0 # freestream velocity [m/s]
alpha  = 20.0/180.0*np.pi # freestream angle-of-attack

## set required matrices and vectors
xdata  = np.empty(m+1) # x-coord of input data points
ydata  = np.empty(m+1) # y-coord of input data points
xmid   = np.empty(m) # x-coord of pivot points
ymid   = np.empty(m) # y-coord of pivot points
ds     = np.empty(m) # element length
sine   = np.empty(m) # sine operation --> dy/ds
cosine = np.empty(m) # cosine operation --> dx/ds
slope  = np.empty(m) # angle of ds with respect to horizontal axis [rad]

"""
Compute aerofoil profile coordinates
"""
def input_data():
    
    dfi = (2.0*np.pi) / m # angle change between each data point

    for i in range(0, m+1):
        fi = (i) * dfi
        xdata[i] = radius * (1.0-np.cos(fi)) - radius + xc
        ydata[i] = radius * np.sin(fi) + yc

"""
Profile data preparation
"""
def data_preparation():
    # global ds, sine, cosine, slope, xmid, ymid

    # set initial x and y values
    x1 = xdata[0]
    y1 = ydata[0]

    # constant for tangent angle limits
    ex = 1e-6

    for n in range(0, m):
        x2 = xdata[n+1]
        y2 = ydata[n+1]

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

        # Project onto the cylinder
        dx = xmid[n] - xc
        dy = ymid[n] - yc
        r = np.hypot(dx, dy)
        xmid[n] = xc + radius * dx / r
        ymid[n] = yc + radius * dy / r

        # move to next coordinates
        x1, y1 = x2, y2

"""
Coupling Coefficients
"""
def coupling_coefficient():
    global coup, slope
    coup = np.empty((m,m))

    # after data_preparation() filled ds, sine, cosine, and slope
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
    for i in range(0, m-1):
        for j in range(0, m-1):
            if j != i:
                dx = xmid[j] - xmid[i]
                dy = ymid[j] - ymid[i]
                r2 = dx**2 + dy**2

                u =  dy / (2.0*np.pi * r2)
                v = -dx / (2.0*np.pi * r2)

                coup[j,i] =  (u * cosine[j] + v * sine[j]) * ds[i]
                coup[i,j] = -(u * cosine[i] + v * sine[i]) * ds[j]

"""
Matrix Inversion of coupling coefficients
"""
def invert_matrix():
    global coup_inv

    # Step 1: L-U Decomposition
    L = np.eye(m)
    U = np.zeros((m,m))

    for i in range(m):
        # Compute U row i
        for j in range(i, m):
            sum_u = sum(L[i,k]*U[k,j] for k in range(i))
            U[i,j] = coup[i,j] - sum_u
        
        # Compute L column i
        for j in range(i+1, m):
            sum_l = sum(L[j,k]*U[k,i] for k in range(i))
            L[j,i] = (coup[j,i] - sum_l) / U[i,i]
    
    # Step 2: Inveres L-U
    coup_inv = np.empty((m,m))
        
    for j in range(m):

        # solve L y = e_j (forward substitution)
        y = np.zeros(m)
        e = np.zeros(m)
        e[j] = 1.0 # only the jth term of jth column in identitiy matrix is 1.0

        for i in range(m):
            y[i] = e[i] - sum(L[i,k]*y[k] for k in range(i))

        # solve U y = e_j (backward substitution)
        x = np.zeros(m)

        for i in reversed(range(m)):
            x[i] = (y[i] - sum(U[i,k]*x[k] for k in range(i+1, m))) / U[i,i]

        # Set j-th column of inverse
        coup_inv[:,j] = x

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
def vorticity_solution():
    global vorticity
    vorticity = np.zeros(m)

    # build A and b as usual
    A = coup.copy()
    b = rhs.copy()

    # replace the last equation with: sum_j gamma_j ds_j = 0
    A[-1, :] = ds
    b[-1] = 0.0

    vorticity = np.linalg.solve(A, b)

    gamma = np.sum(vorticity * ds)   # should be ~0
    print(f"Total Circulation = {gamma:.2f}")


input_data()
data_preparation()
coupling_coefficient()
# invert_matrix()
right_hand_side()
vorticity_solution()


"""
Plot flow around the cylinder
"""
# --------------------------
# Define 2D grid for velocity field
# --------------------------
x_min, x_max = -1.5, 1.5
y_min, y_max = -1.5, 1.5

nx, ny = 500, 500

X, Y = np.meshgrid(np.linspace(x_min, x_max, nx),
                   np.linspace(y_min, y_max, ny))

# --------------------------
# Compute velocity field (U, V) on grid
# --------------------------
U = np.zeros_like(X)
V = np.zeros_like(Y)
Q = np.zeros_like(U)

for i in range(nx):
    for j in range(ny):
        
        # Mask the velocity field inside the cylinder
        dx = X[j,i] - xc
        dy = Y[j,i] - yc
        if dx**2 + dy**2 <= radius**2:
            U[j,i] = np.nan
            V[j,i] = np.nan
            continue

        u = W * np.cos(alpha)
        v = W * np.sin(alpha)

        for k in range(m):
            dx = X[j,i] - xmid[k]
            dy = Y[j,i] - ymid[k]
            r2 = dx**2 + dy**2
            
            r2 = max(r2, 1e-12)  # tiny, preserves near-wall velocities

            u +=  vorticity[k] * dy * ds[k] / (2*np.pi*r2)
            v += -vorticity[k] * dx * ds[k] / (2*np.pi*r2)
        
        q = np.sqrt(u**2 + v**2)

        U[j,i] = u
        V[j,i] = v
        Q[j,i] = q

# --------------------------
# Particle starting positions
# --------------------------
n_particles = 121
y_start = np.linspace(y_min*2.0, y_max*2.0, n_particles)
x_start = np.full(n_particles, x_min) # create an array of size(n_particles) and each filled with (x_min) value
particles = np.vstack([x_start, y_start]).T  # particle positions at each timestep [ shape (n_particles, 2) ]


# --------------------------
# Create velocity interpolators with extrapolation
# --------------------------
u_interp = RegularGridInterpolator(
    (np.linspace(Y.min(), Y.max(), Y.shape[0]),
     np.linspace(X.min(), X.max(), X.shape[1])),
    U, bounds_error=False, fill_value=None
)
v_interp = RegularGridInterpolator(
    (np.linspace(Y.min(), Y.max(), Y.shape[0]),
     np.linspace(X.min(), X.max(), X.shape[1])),
    V, bounds_error=False, fill_value=None
)
q_interp = RegularGridInterpolator(
    (np.linspace(Y.min(), Y.max(), Y.shape[0]),
     np.linspace(X.min(), X.max(), X.shape[1])),
    Q, bounds_error=False, fill_value=None
)

# --------------------------
# Integrate particle trajectories using midpoint (RK2) with grid clipping
# --------------------------
dt = 0.01
n_steps = 500
trajectories = np.zeros((n_particles, n_steps, 2))
trajectories[:,0,:] = particles # input initial particle positions

for t in range(1, n_steps):
    for p in range(n_particles):
        pos = trajectories[p,t-1,:]

        # velocity at current position
        u1 = u_interp([pos[1], pos[0]])[0]
        v1 = v_interp([pos[1], pos[0]])[0]

        # midpoint position
        mid_pos = pos + 0.5 * dt * np.array([u1, v1])

        # velocity at midpoint
        u2 = u_interp([mid_pos[1], mid_pos[0]])[0]
        v2 = v_interp([mid_pos[1], mid_pos[0]])[0]

        # advance with midpoint slope
        new_pos = pos + dt * np.array([u2, v2])

        dx, dy = new_pos - np.array([xc, yc])
        r = np.hypot(dx, dy)
        if r < radius:
            eps = 1e-6
            new_pos = np.array([xc, yc]) + (radius + eps) * np.array([dx, dy]) / (r + 1e-12)

        # clip new position to grid boundaries
        # new_pos[0] = np.clip(new_pos[0], X.min(), X.max())
        # new_pos[1] = np.clip(new_pos[1], Y.min(), Y.max())

        trajectories[p,t,:] = new_pos

# --------------------------
# Plot streamlines and particle trajectories (limited to grid)
# --------------------------
def plot_flow(object=True, track_velocity=True, track_particle=True):
    global n_particles, trajectories, X, Y, U, V

    plt.figure(figsize=(8,6))

    # Plot the object
    if object is True:
        plt.scatter(xmid, ymid, color='black')  # cylinder panels

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
    plt.title('Streamlines and particle trajectories (clipped to grid)')
    
    plt.show()

plot_flow(object=True, track_velocity=False, track_particle=True)
