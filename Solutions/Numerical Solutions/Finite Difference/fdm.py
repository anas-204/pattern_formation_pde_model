import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Spatial derivative functions
def dss004(xl, xu, n, u):
    """Fourth-order finite difference for first derivatives"""
    dx = (xu - xl) / (n - 1)
    ux = np.zeros(n)
    # Left boundary
    ux[0] = (-25 * u[0] + 48 * u[1] - 36 * u[2] + 16 * u[3] - 3 * u[4]) / (12 * dx)
    # Left interior
    ux[1] = (-3 * u[0] - 10 * u[1] + 18 * u[2] - 6 * u[3] + u[4]) / (12 * dx)
    # Interior points
    for i in range(2, n - 2):
        ux[i] = (u[i - 2] - 8 * u[i - 1] + 8 * u[i + 1] - u[i + 2]) / (12 * dx)
    # Right interior
    ux[n - 2] = (-u[n - 5] + 6 * u[n - 4] - 18 * u[n - 3] + 10 * u[n - 2] + 3 * u[n - 1]) / (12 * dx)
    # Right boundary
    ux[n - 1] = (3 * u[n - 5] - 16 * u[n - 4] + 36 * u[n - 3] - 48 * u[n - 2] + 25 * u[n - 1]) / (12 * dx)
    return ux

def dss044(xl, xu, n, u, ux, n1, nu):
    """Fourth-order finite difference for second derivatives with Neumann BCs"""
    dx = (xu - xl) / (n - 1)
    uxx = np.zeros(n)
    # Left boundary (Neumann)
    uxx[0] = 2 * (u[1] - u[0] - ux[0] * dx) / (dx ** 2)
    # Left interior (second-order)
    uxx[1] = (u[0] - 2 * u[1] + u[2]) / (dx ** 2)
    # Interior points (fourth-order)
    for i in range(2, n - 2):
        uxx[i] = (-u[i - 2] + 16 * u[i - 1] - 30 * u[i] + 16 * u[i + 1] - u[i + 2]) / (12 * dx ** 2)
    # Right interior (second-order)
    uxx[n - 2] = (u[n - 3] - 2 * u[n - 2] + u[n - 1]) / (dx ** 2)
    # Right boundary (Neumann)
    uxx[n - 1] = 2 * (u[n - 2] - u[n - 1] + ux[n - 1] * dx) / (dx ** 2)
    return uxx

def pde_system(t, y, nx, xl, xu, D1, D2, D3, k, n1, nu):
    """PDE system for the three-variable model"""
    # Unpack state vector
    u1 = y[0:nx]
    u2 = y[nx:2 * nx]
    u3 = y[2 * nx:3 * nx]

    # Compute first derivatives
    u1x = dss004(xl, xu, nx, u1)
    u2x = dss004(xl, xu, nx, u2)
    u3x = dss004(xl, xu, nx, u3)

    # Apply Neumann BCs (zero flux)
    u1x[0] = 0; u1x[-1] = 0
    u2x[0] = 0; u2x[-1] = 0
    u3x[0] = 0; u3x[-1] = 0

    # Compute second derivatives
    u1xx = dss044(xl, xu, nx, u1, u1x, n1, nu)
    u2xx = dss044(xl, xu, nx, u2, u2x, n1, nu)
    u3xx = dss044(xl, xu, nx, u3, u3x, n1, nu)

    # Calculate nonlinear terms
    term1 = np.zeros(nx)
    term2 = np.zeros(nx)
    term3 = np.zeros(nx)
    for i in range(nx):
        den = 1 / (k[2] + u2[i]) ** 2
        term1[i] = k[1] * u1[i] * den * u2xx[i]
        term2[i] = k[1] * den * u1x[i] * u2x[i]
        term3[i] = -2 * k[1] * u1[i] * den / (k[2] + u2[i]) * u2x[i] ** 2

    # Compute time derivatives
    u1t = np.zeros(nx)
    u2t = np.zeros(nx)
    u3t = np.zeros(nx)
    for i in range(nx):
        u1t[i] = D1 * u1xx[i] - (term1[i] + term2[i] + term3[i]) + \
                 k[3] * u1[i] * (k[4] * u3[i] ** 2 / (k[9] + u3[i] ** 2) - u1[i])
        u2t[i] = D2 * u2xx[i] + k[5] * u3[i] * (u1[i] ** 2 / (k[6] + u1[i] ** 2)) - \
                 k[7] * u1[i] * u2[i]
        u3t[i] = D3 * u3xx[i] - k[8] * u1[i] * (u3[i] ** 2 / (k[9] + u3[i] ** 2))

    # Pack derivatives
    dydt = np.concatenate([u1t, u2t, u3t])
    return dydt

# Main simulation
if __name__ == "__main__":
    # Set case parameters
    ncase = 6
    nx = 51
    xl, xu = 0, 1
    n1, nu = 2, 2  # Neumann BCs

    # Set parameters based on case
    D1, D2, D3 = 2.0e-06, 8.9e-06, 9.0e-06
    k = np.zeros(10)  # 1-indexed for k[1] to k[9]
    
    if ncase == 1:
        pass  # Default zeros
    elif ncase == 2:
        k[1], k[2] = 3.9e-09, 5.0e-06
    elif ncase == 3:
        k[1], k[2], k[3] = 3.9e-09, 5.0e-06, 1.62e-09
        k[4], k[9] = 3.5e+08, 4.0e-06
    elif ncase == 4:
        k[1], k[2], k[3] = 3.9e-09, 5.0e-06, 1.62e-09
        k[4], k[9] = 3.5e+08, 4.0e-06
        k[5], k[6], k[7] = 5.0e-07, 1.0e+18, 1.0e-13
    elif ncase == 5 or ncase == 6:
        k[1], k[2], k[3] = 3.9e-09, 5.0e-06, 1.62e-09
        k[4], k[9] = 3.5e+08, 4.0e-06
        k[5], k[6], k[7] = 5.0e-07, 1.0e+18, 1.0e-13
        k[8] = 1.0e-14

    # Spatial grid
    x_vals = np.linspace(xl, xu, nx)

    # Time points (5 hours with 30 min intervals)
    t_max = 5 * 3600  # seconds
    t_points = np.linspace(0, t_max, 11)

    # Initial conditions (Gaussian)
    u10, u20, u30 = 1.0e+08, 5.0e-06, 1.0e-03
    u1_0 = u10 * np.exp(-5 * x_vals ** 2)
    u2_0 = u20 * np.exp(-5 * x_vals ** 2)
    u3_0 = u30 * np.exp(-5 * x_vals ** 2)
    y0 = np.concatenate([u1_0, u2_0, u3_0])

    # Solve ODE system
    sol = solve_ivp(
        fun=lambda t, y: pde_system(t, y, nx, xl, xu, D1, D2, D3, k, n1, nu),
        t_span=[0, t_max],
        y0=y0,
        t_eval=t_points,
        method='BDF',  # Stiff solver
        atol=1e-6,
        rtol=1e-6
    )

    # Print computational statistics
    print(f"Solution computed with {sol.nfev} function evaluations")
    print(f"Number of time steps: {len(sol.t)}")

    # Extract solutions
    u1_sol = sol.y[:nx]
    u2_sol = sol.y[nx:2 * nx]
    u3_sol = sol.y[2 * nx:3 * nx]

    # Print last state results
    u1_last_state = u1_sol[:, -1]
    u2_last_state = u2_sol[:, -1]
    u3_last_state = u3_sol[:, -1]
    print("\nVerifying with storage data for the final state:")
    print("x (cm)\t\tu1 (cells/ml)\t\tu2 (M)\t\t\tu3 (M)")
    print("-" * 60)
    for j in range(len(x_vals)):
        u1_val = np.array(u1_last_state).flatten()[j]
        u2_val = np.array(u2_last_state).flatten()[j]
        u3_val = np.array(u3_last_state).flatten()[j]
        print(f"{x_vals[j]:.4f}\t\t{u1_val:.4e}\t\t{u2_val:.4e}\t\t{u3_val:.4e}")

    # Plot results
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    for i in range(0, len(sol.t)):  # Plot every other time point
        plt.plot(x_vals, u1_sol[:, i], label=f't={sol.t[i] / 3600:.1f}h')
    plt.title('Evolution of u1')
    plt.xlabel("Position x (cm)")
    plt.ylabel("u1(x,t), t=0,0.5,...,5")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    for i in range(0, len(sol.t)):
        plt.plot(x_vals, u2_sol[:, i], label=f't={sol.t[i] / 3600:.1f}h')
    plt.title('Evolution of u2')
    plt.xlabel("Position x (cm)")
    plt.ylabel("u2(x,t), t=0,0.5,...,5")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    for i in range(0, len(sol.t)):
        plt.plot(x_vals, u3_sol[:, i], label=f't={sol.t[i] / 3600:.1f}h')
    plt.title('Evolution of u3')
    plt.xlabel("Position x (cm)")
    plt.ylabel("u3(x,t), t=0,0.5,...,5")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Book results at t = 5 hours (from machine learning code)
    book_results = {
        "0.0": {"u1": 1.034e7, "u2": 2.480e-6, "u3": 3.449e-4},
        "0.2": {"u1": 9.783e6, "u2": 2.384e-6, "u3": 3.358e-4},
        "0.4": {"u1": 8.333e6, "u2": 2.133e-6, "u3": 3.116e-4},
        "0.6": {"u1": 6.603e6, "u2": 1.824e-6, "u3": 2.803e-4},
        "0.8": {"u1": 5.294e6, "u2": 1.576e-6, "u3": 2.541e-4},
        "1.0": {"u1": 4.825e6, "u2": 1.481e-6, "u3": 2.438e-4}
    }

    # Analysis points (x locations)
    analysis_points = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # Get FDM results at t=5 hours (last time step)
    last_time_index = -1
    sim_results = {}
    for point in analysis_points:
        # Find closest grid point in FDM cell centers
        idx = np.abs(x_vals - point).argmin()
        sim_results[str(point)] = {
            'u1': u1_sol[idx, last_time_index],
            'u2': u2_sol[idx, last_time_index],
            'u3': u3_sol[idx, last_time_index]
        }

    # Calculate relative errors
    relative_errors = {}
    u_errors = {}
    u_errors['u1'] = np.zeros(len(analysis_points))
    u_errors['u2'] = np.zeros(len(analysis_points))
    u_errors['u3'] = np.zeros(len(analysis_points))
    for i in range(len(analysis_points)):
        point_str = str(analysis_points[i])
        relative_errors[point_str] = {}
        for var in ['u1', 'u2', 'u3']:
            true_val = book_results[point_str][var]
            sim_val = sim_results[point_str][var]
            rel_error = abs(true_val - sim_val) / true_val * 100
            relative_errors[point_str][var] = rel_error
            u_errors[var][i] = rel_error

    # Print comparison table
    print("\n" + "="*80)
    print("ACCURACY ASSESSMENT AT t=5 HOURS (FDM SOLUTION)")
    print("="*80)
    print("x (cm)\t\tVariable\tBook Value\t\tFDM Value\t\tRelative Error (%)")
    print("-"*100)
    for point in analysis_points:
        point_str = str(point)
        print(f"{point_str}")
        for var in ['u1', 'u2', 'u3']:
            print(f"\t\t\t{var}\t\t\t{book_results[point_str][var]:.4e}\t\t{sim_results[point_str][var]:.4e}\t\t\t{relative_errors[point_str][var]:.2f}%")

    # Find min and max errors
    min_error_u1 = min(u_errors['u1'])
    max_error_u1 = max(u_errors['u1'])

    min_error_u2 = min(u_errors['u2'])
    max_error_u2 = max(u_errors['u2'])

    min_error_u3 = min(u_errors['u3'])
    max_error_u3 = max(u_errors['u3'])

    print("\nError Range:")
    print(f"Min Relative Error u1: {min_error_u1:.2f}%")
    print(f"Max Relative Error u1: {max_error_u1:.2f}%")

    print(f"Min Relative Error u2: {min_error_u2:.2f}%")
    print(f"Max Relative Error u2: {max_error_u2:.2f}%")

    print(f"Min Relative Error u3: {min_error_u3:.2f}%")
    print(f"Max Relative Error u3: {max_error_u3:.2f}%")

    print("="*80 + "\n")
