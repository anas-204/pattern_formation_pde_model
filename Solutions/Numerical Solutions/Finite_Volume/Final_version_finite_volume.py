import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Set up the spatial and temporal domains
L = 1.0          # Length of the domain [cm]
nx = 51          # Number of spatial points
dx = L/(nx-1)    # Spatial step size
x = np.linspace(0, L, nx)  # Spatial grid

# Time parameters
t_max = 5 * 3600  # 5 hours in seconds
t_points = 11      # Number of output times
t_eval = np.linspace(0, t_max, t_points)  # Output times

# Parameters for case 6 (all terms included)
params = {
    'D1': 2.0e-06,   # Diffusivity for u1 [cm²/s]
    'D2': 8.9e-06,   # Diffusivity for u2 [cm²/s]
    'D3': 9.0e-06,   # Diffusivity for u3 [cm²/s]
    'dx': dx,
    'k': np.array([  # Rate constants
        3.9e-09,    # k1 [M cm²/s]
        5.0e-06,    # k2 [M]
        1.62e-09,   # k3 [hr ml⁻¹ cell⁻¹]
        3.5e+08,    # k4 [cells/ml]
        5.0e-07,    # k5 [s⁻¹]
        1.0e+18,    # k6 [cells²/ml²]
        1.0e-13,    # k7 [ml/(cell s)]
        1.0e-14,    # k8 [s⁻¹]
        4.0e-06     # k9 [M²]
    ])
}

# Initial conditions (Gaussian functions)
u10 = 1.0e+08  # Initial cell density [cells/ml]
u20 = 5.0e-06  # Initial chemoattractant concentration [M]
u30 = 1.0e-03  # Initial stimulant concentration [M]
lambda_val = 5  # Gaussian width parameter

# Set up initial condition vector
y0 = np.zeros(3*nx)
y0[0:nx] = u10 * np.exp(-lambda_val * x**2)        # u1 initial condition
y0[nx:2*nx] = u20 * np.exp(-lambda_val * x**2)     # u2 initial condition
y0[2*nx:3*nx] = u30 * np.exp(-lambda_val * x**2)   # u3 initial condition

# Helper functions for derivatives with Neumann BCs
def first_derivative(u, dx):
    """Compute first derivative with Neumann BCs (central differences in interior)"""
    du = np.zeros_like(u)
    # Central differences in interior
    du[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
    # Neumann BCs (zero flux at boundaries)
    du[0] = 0
    du[-1] = 0
    return du

def second_derivative(u, du, dx):
    """Compute second derivative with Neumann BCs using ghost points"""
    d2u = np.zeros_like(u)
    # Central difference in interior
    d2u[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
    
    # Neumann BCs using ghost points: du/dx=0 at boundaries
    # Left boundary (i=0): u[-1] = u[1]
    d2u[0] = (2*u[1] - 2*u[0]) / dx**2
    # Right boundary (i=nx-1): u[nx] = u[nx-2]
    d2u[-1] = (2*u[-2] - 2*u[-1]) / dx**2
    return d2u

# Right-hand side function for ODE system
def rhs(t, y, dx, params):
    D1, D2, D3 = params['D1'], params['D2'], params['D3']
    k = params['k']
    
    # Split solution vector into components
    u1 = y[0:nx]
    u2 = y[nx:2*nx]
    u3 = y[2*nx:3*nx]
    
    # Compute first derivatives
    u1x = first_derivative(u1, dx)
    u2x = first_derivative(u2, dx)
    u3x = first_derivative(u3, dx)
    
    # Compute second derivatives
    u1xx = second_derivative(u1, u1x, dx)
    u2xx = second_derivative(u2, u2x, dx)
    u3xx = second_derivative(u3, u3x, dx)
    
    # Chemotaxis term expansion (equation 2.5 adapted for parameters)
    den = 1/(k[1] + u2)**2
    term1 = k[0] * u1 * den * u2xx
    term2 = k[0] * den * u1x * u2x
    term3 = -2 * k[0] * u1 * den / (k[1] + u2) * u2x**2
    
    # Time derivatives for each component
    du1dt = np.zeros(nx)
    du2dt = np.zeros(nx)
    du3dt = np.zeros(nx)
    
    # PDE equations (2.6)
    for i in range(nx):
        # u1 equation (cells)
        du1dt[i] = (D1 * u1xx[i] 
                   - (term1[i] + term2[i] + term3[i])
                   + k[2] * u1[i] * (k[3] * u3[i]**2 / (k[8] + u3[i]**2) - u1[i]))
        
        # u2 equation (chemoattractant)
        du2dt[i] = (D2 * u2xx[i]
                   + k[4] * u3[i] * (u1[i]**2 / (k[5] + u1[i]**2))
                   - k[6] * u1[i] * u2[i])
        
        # u3 equation (stimulant)
        du3dt[i] = (D3 * u3xx[i]
                   - k[7] * u1[i] * (u3[i]**2 / (k[8] + u3[i]**2)))
    
    # Combine derivatives into single vector
    dydt = np.concatenate((du1dt, du2dt, du3dt))
    return dydt

# Solve the ODE system
sol = solve_ivp(
    fun=lambda t, y: rhs(t, y, dx, params),
    t_span=[0, t_max],
    y0=y0,
    t_eval=t_eval,
    method='BDF',  # Stiff solver suitable for diffusion problems
    rtol=1e-6,
    atol=1e-8
)

# Extract solutions
u1_sol = sol.y[:nx, :]
u2_sol = sol.y[nx:2*nx, :]
u3_sol = sol.y[2*nx:3*nx, :]

# Plotting function
def plot_solution(x, t, u, title, ylabel, filename=None):
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(t)))
    
    for i, ti in enumerate(t):
        plt.plot(x, u[:, i], color=colors[i], 
                label=f't={ti:.1f} h' )
    
    plt.xlabel('x [cm]')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# Convert time to hours for plotting
t_hours = t_eval / 3600

# Plot the solutions
plot_solution(x, t_hours, u1_sol, 
             'Cell Density $u_1(x,t)$', 
             '$u_1$ [cells/ml]', 'u1_solution.png')

plot_solution(x, t_hours, u2_sol, 
             'Chemoattractant Concentration $u_2(x,t)$', 
             '$u_2$ [M]', 'u2_solution.png')

plot_solution(x, t_hours, u3_sol, 
             'Stimulant Concentration $u_3(x,t)$', 
             '$u_3$ [M]', 'u3_solution.png')

# Print computational statistics
print(f"Solution computed with {sol.nfev} function evaluations")
print(f"Number of time steps: {len(sol.t)}")


#------------------------------------------------------------------------------------------------------
# Accuracy calculation (added section)
# ======================================================================
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

# Get FD results at t=5 hours (last time step)
last_time_index = -1
sim_results = {}
for point in analysis_points:
    # Find closest grid point
    idx = np.abs(x - point).argmin()
    sim_results[str(point)] = {
        'u1': u1_sol[idx, last_time_index],
        'u2': u2_sol[idx, last_time_index],
        'u3': u3_sol[idx, last_time_index]
    }

# Calculate relative errors
relative_errors = {}
for point in analysis_points:
    point_str = str(point)
    relative_errors[point_str] = {}
    for var in ['u1', 'u2', 'u3']:
        true_val = book_results[point_str][var]
        sim_val = sim_results[point_str][var]
        rel_error = abs(true_val - sim_val) / true_val * 100
        relative_errors[point_str][var] = rel_error

# Print comparison table
print("\n" + "="*80)
print("ACCURACY ASSESSMENT AT t=5 HOURS (FD SOLUTION)")
print("="*80)
print("x (cm)\t\tVariable\tBook Value\t\tSimulated Value\t\tRelative Error (%)")
print("-"*100)
for point in analysis_points:
    point_str = str(point)
    for var in ['u1', 'u2', 'u3']:
        print(f"{point_str}\t\t{var}\t\t{book_results[point_str][var]:.4e}\t\t{sim_results[point_str][var]:.4e}\t\t{relative_errors[point_str][var]:.2f}%")

# Find min and max errors
all_errors = [relative_errors[p][var] for p in relative_errors for var in relative_errors[p]]
min_error = min(all_errors)
max_error = max(all_errors)
print("\nError Range:")
print(f"Min Relative Error: {min_error:.2f}%")
print(f"Max Relative Error: {max_error:.2f}%")
print("="*80 + "\n")


#------------------------------------------------------------------------------------------------------



dx = L / nx  # Control volume size
x_centers = np.linspace(dx/2, L - dx/2, nx)  # Cell centers
x_faces = np.linspace(0, L, nx + 1)           # Cell faces




# Initialize state vector
y0 = np.zeros(3 * nx)
y0[:nx] = u10 * np.exp(-lambda_val * x_centers**2)        # u1
y0[nx:2*nx] = u20 * np.exp(-lambda_val * x_centers**2)  # u2
y0[2*nx:] = u30 * np.exp(-lambda_val * x_centers**2)      # u3

# Helper functions for flux calculations
def harmonic_mean(a, b):
    """Harmonic mean for discontinuous coefficients"""
    return (2 * a * b) / (a + b + 1e-30)

def arithmetic_mean(a, b):
    """Arithmetic mean for smooth coefficients"""
    return (a + b) / 2

# Right-hand side function for FVM
def rhs_fvm(t, y, params):
    # Unpack parameters
    D1, D2, D3 = params['D1'], params['D2'], params['D3']
    k = params['k']
    dx = params['dx']
    n = nx
    
    # Split solution vector
    u1 = y[:n]
    u2 = y[n:2*n]
    u3 = y[2*n:]
    
    # Initialize derivatives
    du1dt = np.zeros(n)
    du2dt = np.zeros(n)
    du3dt = np.zeros(n)
    
    # Precompute chemotaxis coefficient at cell centers
    chi = (k[0] * u1) / (k[1] + u2)**2
    
    # ==================================================================
    # Flux calculations at control volume faces (i±1/2 interfaces)
    # ==================================================================
    
    # Initialize flux arrays
    J_diff_u1 = np.zeros(n+1)  # Diffusion flux for u1
    J_diff_u2 = np.zeros(n+1)  # Diffusion flux for u2
    J_diff_u3 = np.zeros(n+1)  # Diffusion flux for u3
    J_chemo = np.zeros(n+1)    # Chemotaxis flux
    
    # Interior faces (i = 1 to n-1)
    for i in range(1, n):
        # Diffusion fluxes (central difference)
        J_diff_u1[i] = -D1 * (u1[i] - u1[i-1]) / dx
        J_diff_u2[i] = -D2 * (u2[i] - u2[i-1]) / dx
        J_diff_u3[i] = -D3 * (u3[i] - u3[i-1]) / dx
        
        # Chemotaxis flux at interface (harmonic mean for coefficient)
        chi_face = harmonic_mean(chi[i-1], chi[i])
        du2dx = (u2[i] - u2[i-1]) / dx
        J_chemo[i] = chi_face * du2dx
    
    # Boundary faces (zero flux conditions)
    J_diff_u1[0] = 0; J_diff_u1[-1] = 0
    J_diff_u2[0] = 0; J_diff_u2[-1] = 0
    J_diff_u3[0] = 0; J_diff_u3[-1] = 0
    J_chemo[0] = 0; J_chemo[-1] = 0
    
    # ==================================================================
    # Source term calculations
    # ==================================================================
    S1 = k[2] * u1 * (k[3] * u3**2 / (k[8] + u3**2) - u1)
    S2 = k[4] * u3 * (u1**2 / (k[5] + u1**2)) - k[6] * u1 * u2
    S3 = -k[7] * u1 * (u3**2 / (k[8] + u3**2))
    
    # ==================================================================
    # Assemble time derivatives (flux balance + sources)
    # ==================================================================
    for i in range(n):
        # Flux divergence terms
        flux_div_u1 = (J_diff_u1[i] - J_diff_u1[i+1]) / dx - (J_chemo[i+1] - J_chemo[i]) / dx
        flux_div_u2 = (J_diff_u2[i] - J_diff_u2[i+1]) / dx
        flux_div_u3 = (J_diff_u3[i] - J_diff_u3[i+1]) / dx
        
        # Final time derivatives
        du1dt[i] = flux_div_u1 + S1[i]
        du2dt[i] = flux_div_u2 + S2[i]
        du3dt[i] = flux_div_u3 + S3[i]
    
    return np.concatenate((du1dt, du2dt, du3dt))

# Solve with FVM
sol_fvm = solve_ivp(
    fun=lambda t, y: rhs_fvm(t, y, params),
    t_span=[0, t_max],
    y0=y0,
    t_eval=t_eval,
    method='BDF',
    rtol=1e-6,
    atol=1e-8
)

# Extract FVM solutions
u1_fvm = sol_fvm.y[:nx, :]
u2_fvm = sol_fvm.y[nx:2*nx, :]
u3_fvm = sol_fvm.y[2*nx:, :]

# Plotting function comparison
def plot_comparison(x_fd, u_fd, x_fvm, u_fvm, t, title, ylabel):
    plt.figure(figsize=(12, 6))
    
    # FD results
    plt.subplot(1, 2, 1)
    colors = plt.cm.viridis(np.linspace(0, 1, len(t)))
    for i, ti in enumerate(t):
        plt.plot(x_fd, u_fd[:, i], color=colors[i], 
                label=f't={ti:.1f} h' )
    plt.xlabel('x [cm]')
    plt.ylabel(ylabel)
    plt.title(f'FD: {title}')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # FVM results
    plt.subplot(1, 2, 2)
    for i, ti in enumerate(t):
        plt.plot(x_fvm, u_fvm[:, i], color=colors[i], 
                label=f't={ti:.1f} h' )
    plt.xlabel('x [cm]')
    plt.ylabel(ylabel)
    plt.title(f'FVM: {title}')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}_comparison.png', dpi=300)
    plt.show()



# Plot comparison
plot_comparison(x, u1_sol, x_centers, u1_fvm, t_hours, 
               'Cell Density $u_1(x,t)$', '$u_1$ [cells/ml]')

plot_comparison(x, u2_sol, x_centers, u2_fvm, t_hours,
               'Chemoattractant $u_2(x,t)$', '$u_2$ [M]')

plot_comparison(x, u3_sol, x_centers, u3_fvm, t_hours,
               'Stimulant $u_3(x,t)$', '$u_3$ [M]')

# Conservation check
def check_conservation(u_fd, u_fvm, dx, name):
    """Check mass conservation properties"""
    mass_fd = np.sum(u_fd * dx, axis=0)
    mass_fvm = np.sum(u_fvm * dx, axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_hours, mass_fd, 'b-', label='FD')
    plt.plot(t_hours, mass_fvm, 'r--', label='FVM')
    plt.xlabel('Time [hours]')
    plt.ylabel('Total Mass')
    plt.title(f'Mass Conservation: {name}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f'mass_conservation_{name}.png', dpi=300)
    plt.show()
    
    print(f"Mass change FD ({name}): {(mass_fd[-1] - mass_fd[0])/mass_fd[0]:.2e}")
    print(f"Mass change FVM ({name}): {(mass_fvm[-1] - mass_fvm[0])/mass_fvm[0]:.2e}")