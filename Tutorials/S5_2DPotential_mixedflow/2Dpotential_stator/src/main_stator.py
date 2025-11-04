import numpy as np

from .utils import (
    load_airfoil_xy,
    rearrange_airfoil, resample_airfoil_cosine_finite_TE, rotate_airfoil_about_te,
    data_preparation, coupling_coefficient_cascade, right_hand_side_stator_unit, vorticity_solution_kutta,
    airfoil_performance_in_cascade, display_cascade_performance,
    airfoil_visualisation, flow_visualisation, plot_actual
)

""" Starting flow conditions """
rho = 1.225 # air density [kg/m^3]
W   = 50.0 # inlet flow velocity [m/s]

alpha_deg = 2.0 # inlet flow angle [deg]
alpha     = np.deg2rad(alpha_deg)

U = W * np.cos(alpha)
V = W * np.sin(alpha)

""" Blade parameters """
m        = 100   # number of aerofoil profile data input points (m+1 for completing the profile)
beta_deg = -3.0 # tilt agnle of airfoil, +ve means LE pointing downwards (rotated anti-clockwise about TE)
beta     = np.deg2rad(beta_deg)

t_ratio  = 1.0   # blade pitch (normalised by chord length)

r_c = np.ones(int(m/2)) # camber node radial coord
r_m = np.ones(int(m))   # panel element radial coord
r1  = 1.0 # starting blade element radial position
r2  = 1.0 # final blade element radial position

## ===============================
#  Load and resample aerofoil data
## ===============================
x, y = load_airfoil_xy("src/data/aerofoil_coords.txt")
xdata, ydata, x_le, y_le, x_te, y_te, chord, max_thick = rearrange_airfoil(x, y)
xnew, ynew = resample_airfoil_cosine_finite_TE(xdata, ydata, n_points=m)
xnew, ynew = rotate_airfoil_about_te(xnew, ynew, beta_deg, pivot='mid')

## ================================
#  Pre-calculation data preparation
## ================================
t = chord * t_ratio
n_datapoints = xnew.shape[0] # number of new data points

ds, sine, cosine, slope, xmid, ymid = data_preparation(xnew, ynew, n_datapoints)

n_midpoints = xmid.shape[0]
if n_midpoints != m:
    raise ValueError(f"Dimension error : number of midpoint nodes = {n_midpoints} | number of radial points = {r_m.shape[0]}")

coup = coupling_coefficient_cascade(t, ds, sine, cosine, slope, xmid, ymid, n_midpoints)

## ==================
#  Cascade Solution
## ==================
""" 
Unit solutions for : 
  1) U = 1.0 
  2) V = 1.0
"""
rhs_U, rhs_V = right_hand_side_stator_unit(m, slope)
vorticity_U, vorticity_V  = vorticity_solution_kutta(coup, rhs_U, rhs_V, ds, mode="circulation")

""" Full Cascade solution """
results, blade = airfoil_performance_in_cascade(rho, alpha, beta, chord, t,
                                   ds, xmid, ymid, sine, cosine, slope, U, V,
                                   vorticity_U, vorticity_V, cascade_type="stator",
                                   plot_cp=False, plot_save=True)
blade["airfoil_coords"] = np.column_stack((xnew, ynew))

display_cascade_performance(results, blade)

## =======================
#  Plots and Visualisation
## =======================
# plot_airfoil = False
# plot_flow    = False
# track_particle = True
# track_velocity = False

# n_particles = 121
# dt = 0.01
# n_steps = 500

# if plot_airfoil or plot_flow:
#     if plot_flow:
#         # compute flowfield vectors """"
#         X, Y, U, V, trajectories = flow_visualisation(xnew, ynew, xmid, ymid, chord, 
#                                                       alpha, beta, W, ds, results["vorticity"], 
#                                                       n_particles=n_particles, dt=dt, n_steps=n_steps)
#         # plot flowfield
#         plot_actual(xnew, ynew, U, V, X, Y,
#                     n_particles, trajectories, results["alpha_start"], blade["tilt_angle"], 
#                     object=True, track_particle=track_particle, track_velocity=track_velocity)
#     else:
#         # only show airfoil
#         airfoil_visualisation(blade["airfoil_coords"][:,0], blade["airfoil_coords"][:,1], blade["chord"])


# print("Script passed.\n")   