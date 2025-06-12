# Operator Splitting (Strang Splitting)

# Step 1 : Problem Setup
# import numpy as np
# from scipy.sparse import diags
# from scipy.sparse.linalg import spsolve
# import matplotlib.pyplot as plt
#
# # Parameters
# D1, D2, D3 = 1.0, 1.0, 1.0  # Diffusion coefficients
# k1, k2, k3, k4, k5, k6, k7, k8, k9 = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0  # Reaction coefficients
# L = 10.0    # Domain length
# N = 100     # Spatial points
# dx = L / (N - 1)
# x = np.linspace(0, L, N)
#
# # Initial conditions (Gaussian peaks)
# u1 = np.exp(-(x - L/2)**2 / 1.0)
# u2 = np.zeros(N)
# u3 = np.ones(N)
#
# # Boundary conditions (Neumann: du/dx = 0 at ends)
# def apply_bcs(u):
#     u[0] = u[1]    # du/dx = 0 at x=0
#     u[-1] = u[-2]  # du/dx = 0 at x=L
#     return u
#
# # step 2 : Diffusion Step (Implicit Crank-Nicolson)
# def diffuse(u, D, dt):
#     # Construct tridiagonal matrix: (I - dt*D/2 * Laplacian)
#     alpha = D * dt / (2 * dx ** 2)
#     A = diags([-alpha, 1 + 2 * alpha, -alpha], [-1, 0, 1], shape=(N, N), format='csr')
#
#     # Apply Neumann BCs (modify first/last rows)
#     A[0, 0] = 1 + alpha;
#     A[0, 1] = -alpha  # du/dx=0 at x=0
#     A[-1, -1] = 1 + alpha;
#     A[-1, -2] = -alpha  # du/dx=0 at x=L
#
#     # Solve (I - dt*D/2 ∇²)u_new = (I + dt*D/2 ∇²)u_old
#     b = u + (D * dt / 2) * (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx ** 2
#     b = apply_bcs(b)
#     return spsolve(A, b)
#
# # step 3 : Reaction Step (Explicit Euler)
#
# def react(u1, u2, u3, dt):
#     du1 = dt * (k3 * u1 * (k4 * u3**2 / (k9 + u3**2) - u1))
#     du2 = dt * (k5 * u3 * (u1**2 / (k6 + u1**2) - k7 * u1 * u2))
#     du3 = dt * (-k8 * u1 * (u3**2 / (k9 + u3**2)))
#     return u1 + du1, u2 + du2, u3 + du3
#
# # Step 4 : Strang Splitting Loop
#
# dt = 0.01  # Timestep
# T = 10.0  # Total time
# steps = int(T / dt)
#
# # Storage for plotting
# u1_history = [u1.copy()]
# u2_history = [u2.copy()]
# u3_history = [u3.copy()]
#
# for n in range(steps):
#     # Half-step diffusion
#     u1 = diffuse(u1, D1, dt / 2)
#     u2 = diffuse(u2, D2, dt / 2)
#     u3 = diffuse(u3, D3, dt / 2)
#
#     # Full-step reaction
#     u1, u2, u3 = react(u1, u2, u3, dt)
#
#     # Half-step diffusion
#     u1 = diffuse(u1, D1, dt / 2)
#     u2 = diffuse(u2, D2, dt / 2)
#     u3 = diffuse(u3, D3, dt / 2)
#
#     # Apply BCs and store
#     u1 = apply_bcs(u1)
#     u2 = apply_bcs(u2)
#     u3 = apply_bcs(u3)
#
#     if n % 100 == 0:  # Save every 100 steps
#         u1_history.append(u1.copy())
#         u2_history.append(u2.copy())
#         u3_history.append(u3.copy())
#
# # step 5 : Visualization
#
# # Plot results
# plt.figure(figsize=(12, 8))
# plt.subplot(3, 1, 1)
# for i, sol in enumerate(u1_history):
#     plt.plot(x, sol, alpha=0.5, label=f"t={i*dt*100:.1f}" if i % 2 == 0 else "")
# plt.title("$u_1(x,t)$")
# plt.legend()
#
# plt.subplot(3, 1, 2)
# for sol in u2_history:
#     plt.plot(x, sol, alpha=0.5)
# plt.title("$u_2(x,t)$")
#
# plt.subplot(3, 1, 3)
# for sol in u3_history:
#     plt.plot(x, sol, alpha=0.5)
# plt.title("$u_3(x,t)$")
#
# plt.tight_layout()
# plt.show()







# MOL Method

# import numpy as np
# from scipy.integrate import solve_ivp
# import matplotlib.pyplot as plt
#
# # Parameters
# D1, D2, D3 = 1.0, 1.0, 1.0  # Diffusion coefficients
# k1, k2, k3, k4, k5, k6, k7, k8, k9 = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0  # Reaction coefficients
# L = 10.0  # Domain length
# N = 100  # Number of spatial points
# dx = L / (N - 1)
# x = np.linspace(0, L, N)
#
# # Initial conditions (Gaussian for u1, zero for u2, constant for u3)
# u1 = np.exp(-(x - L / 2) ** 2 / 1.0)
# u2 = np.zeros(N)
# u3 = np.ones(N)
#
#
# # Laplacian (central difference)
# def laplacian(u):
#     return (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx ** 2
#
#
# # Cross-diffusion term: ∇·[k1 u1 / (k2 + u2)^2 ∇u2]
# def cross_diffusion(u1, u2):
#     grad_u2 = (np.roll(u2, -1) - np.roll(u2, 1)) / (2 * dx)  # Central difference for ∇u2
#     term = (k1 * u1) / (k2 + u2) ** 2 * grad_u2
#     return (np.roll(term, -1) - np.roll(term, 1)) / (2 * dx)  # ∇·(term)
#
#
# # Right-hand side (RHS) function for MoL
# def rhs(t, U):
#     u1, u2, u3 = U[:N], U[N:2 * N], U[2 * N:3 * N]
#
#     # Eq. 2.1a
#     du1dt = D1 * laplacian(u1) - cross_diffusion(u1, u2) + k3 * u1 * (k4 * u3 ** 2 / (k9 + u3 ** 2) - u1)
#
#     # Eq. 2.1b
#     du2dt = D2 * laplacian(u2) + k5 * u3 * (u1 ** 2 / (k6 + u1 ** 2)) - k7 * u1 * u2
#
#     # Eq. 2.1c
#     du3dt = D3 * laplacian(u3) - k8 * u1 * (u3 ** 2 / (k9 + u3 ** 2))
#
#     return np.concatenate([du1dt, du2dt, du3dt])
#
#
# # Solve using BDF (stiff-system solver)
# sol = solve_ivp(rhs, [0, 10], np.concatenate([u1, u2, u3]), method='BDF', rtol=1e-6)
#
# # Extract solutions
# u1_sol = sol.y[:N, -1]  # Final state of u1
# u2_sol = sol.y[N:2 * N, -1]
# u3_sol = sol.y[2 * N:3 * N, -1]
#
# plt.figure(figsize=(10, 6))
# plt.plot(x, u1_sol, label='$u_1$ (final)')
# plt.plot(x, u2_sol, label='$u_2$ (final)')
# plt.plot(x, u3_sol, label='$u_3$ (final)')
# plt.xlabel('Position (x)')
# plt.ylabel('Concentration')
# plt.title('Solution at $t=10$ (Method of Lines)')
# plt.legend()
# plt.grid()
# plt.show()


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters from Table 2.6 in the paper
D1 = 2.0e-06
D2 = 8.9e-06
D3 = 9.0e-06

# Rate constants
k1, k2, k3, k4, k5, k6, k7, k8, k9 = (
    3.9e-09, 5.0e-06, 1.62e-09, 3.5e+08, 5.0e-07, 1.0e+18, 1.0e-13, 1.0e-14, 4.0e-06
)

# Spatial grid - using finer resolution than original
nx = 101  # Increased from 51 to get better spatial resolution
xl = 0.0
xu = 1.0
xg = np.linspace(xl, xu, nx)
dx = (xu - xl) / (nx - 1)


# Initial Conditions with small perturbations
def initial_conditions(x):
    # Using Gaussian function as in paper
    u1 = 1e8 * np.exp(-5 * x ** 2)
    u2 = 5e-6 * np.exp(-5 * x ** 2)
    u3 = 1e-3 * np.exp(-5 * x ** 2)

    # Add small random noise to trigger pattern formation
    u1 += 1e6 * np.random.randn(nx)
    u2 += 5e-8 * np.random.randn(nx)
    u3 += 1e-5 * np.random.randn(nx)

    return np.concatenate((u1, u2, u3))


# First-order derivative (central difference)
def dss004(xl, xu, nx, u):
    dx = (xu - xl) / (nx - 1)
    du = np.zeros_like(u)
    for i in range(1, nx - 1):
        du[i] = (u[i + 1] - u[i - 1]) / (2 * dx)
    # Neumann BC
    du[0] = 0
    du[-1] = 0
    return du


# Second-order derivative (central difference)
def dss044(xl, xu, nx, u, ux, nl, nu):
    dx = (xu - xl) / (nx - 1)
    ddu = np.zeros_like(u)
    for i in range(1, nx - 1):
        ddu[i] = (u[i + 1] - 2 * u[i] + u[i - 1]) / dx ** 2
    # Neumann BC
    ddu[0] = 0
    ddu[-1] = 0
    return ddu


# RHS Function for ODE Integration
def p_form_2(t, u):
    u1 = u[:nx]
    u2 = u[nx:2 * nx]
    u3 = u[2 * nx:3 * nx]

    # First derivatives
    u1x = dss004(xl, xu, nx, u1)
    u2x = dss004(xl, xu, nx, u2)
    u3x = dss004(xl, xu, nx, u3)

    # Second derivatives
    u1xx = dss044(xl, xu, nx, u1, u1x, 2, 2)
    u2xx = dss044(xl, xu, nx, u2, u2x, 2, 2)
    u3xx = dss044(xl, xu, nx, u3, u3x, 2, 2)

    # Nonlinear terms
    term1 = np.zeros(nx)
    term2 = np.zeros(nx)
    term3 = np.zeros(nx)

    for i in range(nx):
        den = 1 / (k2 + u2[i]) ** 2
        term1[i] = k1 * u1[i] * den * u2xx[i]
        term2[i] = k1 * den * u1x[i] * u2x[i]
        term3[i] = -2 * k1 * u1[i] * den / (k2 + u2[i]) * u2x[i] ** 2

    # PDEs with all terms
    u1t = D1 * u1xx - (term1 + term2 + term3) + k3 * u1 * (k4 * u3 ** 2 / (k9 + u3 ** 2) - u1)
    u2t = D2 * u2xx + k5 * u3 * (u1 ** 2 / (k6 + u1 ** 2)) - k7 * u1 * u2
    u3t = D3 * u3xx - k8 * u1 * (u3 ** 2 / (k9 + u3 ** 2))

    return np.concatenate((u1t, u2t, u3t))


# Time settings (using time scale from paper)
t_span = [0, 5 * 3600]  # Convert hours to seconds
t_eval = np.linspace(0, 5 * 3600, 25)  # More evaluation points

# Solve with tighter tolerances
sol = solve_ivp(
    p_form_2,
    t_span,
    initial_conditions(xg),
    t_eval=t_eval,
    method='BDF',
    rtol=1e-8,  # Tighter relative tolerance
    atol=1e-10  # Tighter absolute tolerance
)

# Extract results
u1_sol = sol.y[:nx, :]
u2_sol = sol.y[nx:2 * nx, :]
u3_sol = sol.y[2 * nx:3 * nx, :]


# Plotting results - Following paper style
def plot_solution(x, solutions, titles):
    plt.figure(figsize=(12, 6))

    times_to_plot = [0, 5, 10, 15, 20]  # Select specific time points

    for idx in times_to_plot:
        t_hour = sol.t[idx] / 3600  # Convert back to hours
        for i in range(len(solutions)):
            plt.subplot(len(solutions), 1, i + 1)
            plt.plot(x, solutions[i][:, idx], label=f't={t_hour:.1f}h')
            plt.title(f'Solution {titles[i]}')
            plt.xlabel('x')
            plt.ylabel(titles[i])
            plt.legend()
            plt.grid(True)

    plt.tight_layout()
    plt.show()


# Plot individual components
plot_solution(xg, [u1_sol, u2_sol, u3_sol], ['u1(x,t)', 'u2(x,t)', 'u3(x,t)'])

# Term decomposition for analysis
chemo_terms = {
    'chemo1': np.zeros((nx, len(t_eval))),
    'chemo2': np.zeros((nx, len(t_eval))),
    'chemo3': np.zeros((nx, len(t_eval))),
    'chemo4': np.zeros((nx, len(t_eval))),
    'chemo5': np.zeros((nx, len(t_eval))),
    'chemo6': np.zeros((nx, len(t_eval))),
    'chemo7': np.zeros((nx, len(t_eval))),
    'chemo8': np.zeros((nx, len(t_eval))),
    'chemo9': np.zeros((nx, len(t_eval))),
    'chemo10': np.zeros((nx, len(t_eval)))
}


def compute_terms(u1, u2, u3, u1x, u2x, u3x, u1xx, u2xx, u3xx):
    term1 = np.zeros(nx)
    term2 = np.zeros(nx)
    term3 = np.zeros(nx)

    for i in range(nx):
        den = 1 / (k2 + u2[i]) ** 2
        term1[i] = k1 * u1[i] * den * u2xx[i]
        term2[i] = k1 * den * u1x[i] * u2x[i]
        term3[i] = -2 * k1 * u1[i] * den / (k2 + u2[i]) * u2x[i] ** 2

    chemo1 = D1 * u1xx
    chemo2 = -(term1 + term2 + term3)
    chemo3 = k3 * u1 * (k4 * u3 ** 2 / (k9 + u3 ** 2) - u1)
    chemo4 = D2 * u2xx
    chemo5 = k5 * u3 * (u1 ** 2 / (k6 + u1 ** 2)) - k7 * u1 * u2
    chemo6 = D3 * u3xx
    chemo7 = -k8 * u1 * (u3 ** 2 / (k9 + u3 ** 2))
    chemo8 = chemo1 + chemo2 + chemo3
    chemo9 = chemo4 + chemo5
    chemo10 = chemo6 + chemo7

    return chemo1, chemo2, chemo3, chemo4, chemo5, chemo6, chemo7, chemo8, chemo9, chemo10


# Compute and store each term at each time step
for it, t in enumerate(t_eval):
    u = sol.sol(t)
    u1 = u[:nx]
    u2 = u[nx:2 * nx]
    u3 = u[2 * nx:3 * nx]

    u1x = dss004(xl, xu, nx, u1)
    u2x = dss004(xl, xu, nx, u2)
    u3x = dss004(xl, xu, nx, u3)

    u1xx = dss044(xl, xu, nx, u1, u1x, 2, 2)
    u2xx = dss044(xl, xu, nx, u2, u2x, 2, 2)
    u3xx = dss044(xl, xu, nx, u3, u3x, 2, 2)

    chemo1, chemo2, chemo3, chemo4, chemo5, chemo6, chemo7, chemo8, chemo9, chemo10 = compute_terms(u1, u2, u3, u1x,
                                                                                                    u2x, u3x, u1xx,
                                                                                                    u2xx, u3xx)

    chemo_terms['chemo1'][:, it] = chemo1
    chemo_terms['chemo2'][:, it] = chemo2
    chemo_terms['chemo3'][:, it] = chemo3
    chemo_terms['chemo4'][:, it] = chemo4
    chemo_terms['chemo5'][:, it] = chemo5
    chemo_terms['chemo6'][:, it] = chemo6
    chemo_terms['chemo7'][:, it] = chemo7
    chemo_terms['chemo8'][:, it] = chemo8
    chemo_terms['chemo9'][:, it] = chemo9
    chemo_terms['chemo10'][:, it] = chemo10


# Plot selected terms
def plot_terms(x, terms, names):
    plt.figure(figsize=(12, 6))
    times_to_plot = [5, 10, 15, 20]

    for idx, name in enumerate(names):
        plt.subplot(len(names), 1, idx + 1)
        for it in times_to_plot:
            plt.plot(x, terms[name][:, it], label=f't={it * 0.5:.1f}h')
        plt.title(f'{name}: {terms[name][:, it].min():.2e} to {terms[name][:, it].max():.2e}')
        plt.xlabel('x')
        plt.ylabel(name)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()


# Plot key terms
plot_terms(xg, chemo_terms, ['chemo1', 'chemo8', 'chemo9', 'chemo10'])