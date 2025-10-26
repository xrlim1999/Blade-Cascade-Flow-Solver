import numpy as np

from .utils import (
    load_airfoil_xy,
    rearrange_airfoil, resample_airfoil_cosine_finite_TE, rotate_airfoil_about_te,
    data_preparation, coupling_coefficient_cascade, right_hand_side_simple, vorticity_solution_kutta,
    airfoil_performance_cascade, calculate_exit_conditions, cascade_performance, display_cascade_performance,
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
coup = coupling_coefficient_cascade(t, ds, sine, cosine, slope, xmid, ymid, n_midpoints)

## ==================
#  Cascade Solution
## ==================
""" Unit-Solution 1: U = 1.0 """
rhs_u       = right_hand_side_simple(n_midpoints, slope, alpha=0.0, W=1.0)
vorticity_u = vorticity_solution_kutta(coup, rhs_u, ds, mode="circulation")

""" Unit-Solution 2: V = 1.0 """
rhs_v       = right_hand_side_simple(n_midpoints, slope, alpha=np.pi/2, W=1.0)
vorticity_v = vorticity_solution_kutta(coup, rhs_v, ds, mode="circulation")

""" Actual-Solution: W = real inflow velocity """
results_u = airfoil_performance_cascade(rho, alpha, beta, chord, vorticity_u, W, t,
                                        ds, xmid, ymid, sine, cosine, slope,
                                        plot_cp=False, print_results=False)
results_v = airfoil_performance_cascade(rho, alpha, beta, chord, vorticity_v, W, t,
                                        ds, xmid, ymid, sine, cosine, slope,
                                        plot_cp=False, print_results=False)
results = dict()
results["circulation"] = (U * results_u["circulation"]) + (V * results_v["circulation"])
results["vorticity"]   = (U * vorticity_u) + (V * vorticity_v)


""" Save starting flow conditions and blade designs """
results["alpha_start"] = alpha_deg
results["W_start"] = W ; results["U_start"] = U ; results["V_start"] = V

blade = dict()
blade["tilt_angle"] = beta_deg
blade["pitch"] = t ; blade["chord"] = chord ; blade["airfoil_coords"] = np.column_stack((xnew, ynew))
blade["num_datapoints"] = n_datapoints

""" Exit flow conditions """
# angle, absolute velocit (W), and tangential velocity (V)
results["alpha_exit"], results["W_exit"], results["V_exit"] = calculate_exit_conditions(results_u, results_v, t, alpha, U, V)
# axial velocity (uncahged due to mass continuity)
results["U_exit"] = U

""" Cascade Performance """
results = cascade_performance(results, rho, W, results["W_exit"], chord)
results_temp = airfoil_performance_cascade(rho, alpha, beta, chord, results["vorticity"], W, t,
                                            ds, xmid, ymid, sine, cosine, slope,
                                            plot_cp=True, print_results=False)
display_cascade_performance(results, blade)

## =======================
#  Plots and Visualisation
## =======================
plot_airfoil = False
plot_flow    = False
track_particle = True
track_velocity = False

n_particles = 121
dt = 0.01
n_steps = 500

if plot_airfoil or plot_flow:
    if plot_flow:
        # compute flowfield vectors """"
        X, Y, U, V, trajectories = flow_visualisation(xnew, ynew, xmid, ymid, chord, 
                                                      alpha, beta, W, ds, results["vorticity"], 
                                                      n_particles=n_particles, dt=dt, n_steps=n_steps)
        # plot flowfield
        plot_actual(xnew, ynew, U, V, X, Y,
                    n_particles, trajectories, results["alpha_start"], blade["tilt_angle"], 
                    object=True, track_particle=track_particle, track_velocity=track_velocity)
    else:
        # only show airfoil
        airfoil_visualisation(blade["airfoil_coords"][:,0], blade["airfoil_coords"][:,1], blade["chord"])


print("Script passed.\n")   