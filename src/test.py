import numpy as np

def exit_angle_from_gamma_pitch(
    gamma_b, ds, pitch, Ux_in, Vy_in, r, Omega, wake_dir
):
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
    Vy_rot = Omega * r
    Vx_rel = Vx_abs
    Vy_rel = Vy_abs - Vy_rot

    # --- Step 5: Compute angles ---
    beta_exit = np.arctan2(Vy_rel, Vx_rel)
    beta_wake = np.arctan2(wake_dir[:,1], wake_dir[:,0])

    delta_beta = np.arctan2(np.sin(beta_exit - beta_wake),
                            np.cos(beta_exit - beta_wake))

    return beta_exit, beta_wake, delta_beta

# Example parameters for one section
Ux_in = 50.0           # m/s axial inflow
Vy_in = 0.0            # m/s tangential inflow (axial inflow)
Omega = 200.0          # rad/s (rotor); 0 for stator
pitch = 0.1            # m
r = np.array([1.0])    # 1 m radius
ds = np.linspace(0.002, 0.002, 20)
gamma_b = np.linspace(2.0, 3.0, 20)
wake_dir = np.tile([np.cos(5*np.pi/180), np.sin(5*np.pi/180)], (1,1))  # wake at +5°

beta_exit, beta_wake, delta_beta = exit_angle_from_gamma_pitch(
    gamma_b, ds, pitch, Ux_in, Vy_in, r, Omega, wake_dir
)

print(f"β_exit = {np.degrees(beta_exit)[0]:.2f}°")
print(f"β_wake = {np.degrees(beta_wake)[0]:.2f}°")
print(f"Δβ = {np.degrees(delta_beta)[0]:+.2f}°")