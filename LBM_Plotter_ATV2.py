import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def process_csv(file_path, matrix_shape=None, headers=False):
    if headers:
        df = pd.read_csv(file_path, header=0)   # first row is headers
        header_labels = list(df.columns)        # keep column names
    else:
        df = pd.read_csv(file_path, header=None)  # no headers
        header_labels = None

    data = df.values
    time_steps = data.shape[0]

    if matrix_shape is None:
        infos_t = [data[t, 1:] for t in range(time_steps)]
    else:
        infos_t = [data[t, 1:].reshape(matrix_shape) for t in range(time_steps)]

    time = np.array([data[t, 0] for t in range(time_steps)])

    return time, infos_t, header_labels


def plot_vertical_velocity_profile(vel_hist, dim_y, delta_h, N=5, cmap_name="viridis",
                                   title="Vertical Velocity Profile"):
    plt.figure(figsize=(8, 6))
    y_positions = np.linspace(0, dim_y * delta_h, dim_y)

    # pick N equally spaced time indices
    time_indices = np.linspace(0, len(vel_hist) - 1, N, dtype=int)

    # get continuous colormap
    cmap = plt.get_cmap(cmap_name, N)
    total_saves = len(vel_hist)
    for i, mat in enumerate(vel_hist):
        velocity_profile = mat[:, 0]  # take velocity at first column
        percentage = 100 * i / (total_saves - 1)
        plt.plot(velocity_profile, y_positions,
                 color=cmap(i), linewidth=2, alpha=0.9,
                 label=f"t={percentage:.0f}%", marker='o', markersize=6)

    plt.xlabel('Velocity (u) [m/s]')
    plt.ylabel('Vertical Position (y) [m]')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(title=f"Simulation\nevolution", loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_cell_velocity_time_series(vel_hist, row, col, time, title="Velocity Time Series"):
    """Plot velocity evolution at a specific cell"""
    cell_velocity = np.array([mat[row][col] for mat in vel_hist])

    plt.figure(figsize=(10, 6))
    plt.plot(time, cell_velocity, 'bo-', linewidth=2, label='Simulated', markersize=16)
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.title(f'{title} at cell ({row}, {col})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def scale_error_convergence(scale_dim, scale_error, plot=False):
    log_dim_x = np.log10(scale_dim)
    log_error = np.log10(scale_error)

    # Linear regression in log-log space: log(error) = slope * log(N) + intercept
    slope, intercept = np.polyfit(log_dim_x, log_error, 1)

    # Generate fitted line
    fit_error = 10 ** (slope * log_dim_x + intercept)

    # Plot data and regression
    if plot:
        plt.figure(figsize=(6, 4))
        plt.loglog(scale_dim, fit_error, 'gs--', label=f'Fit: slope = {slope:.2f}', markersize=8)
        plt.loglog(scale_dim, scale_error, 'ro-', label='Measured error', markersize=16, markerfacecolor='none')
        plt.xlabel("Grid size (N)")
        plt.ylabel("Relative $L_2$ error")
        plt.title("Convergence plot")
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.tight_layout()
        plt.show()
    return slope

def compare_to_analytical(vel_hist_final, dim_y, delta_h, delta_t, fforce_x, visc, plot=False):
    """
    Compare ONLY fluid cells (exclude solid boundary cells)
    """
    x_centers = np.arange(dim_x)* delta_h + delta_h/2
    y_centres = np.arange(dim_y)* delta_h + delta_h/2 # Place analyzed positions in the center of each cell

    #y_fluid_idx = np.arange(0, dim_y)
    y_fluid_idx = np.arange(1, dim_y - 1)  # to discard fluid nodes
    x_fluid_idx = np.arange(0, dim_x)
    y_centers_analyzed = y_centres[y_fluid_idx]
    x_centers_analyzed = x_centers[x_fluid_idx]

    # bottom wall position (midpoint between node 0 and 1)
    y_wall_bottom = delta_h # Bottom wall is located in the end of the solid node

    # distance from bottom wall to fluid centers (this is the y used in analytic formula)
    y_from_wall = y_centers_analyzed - y_wall_bottom  # Distance from each node until the bottom wall
    # Channel height between walls
    H           = (dim_y - 2) * delta_h

    # simulated u (x-velocity) only for fluid cells
    # be explicit about indexing for numpy arrays
    simulated_velocity_fluid = np.asarray(vel_hist_final)[y_fluid_idx, 0]
    #analytical_velocity       = (fforce_x / (2.0 * visc)) * y_from_wall * (H - y_from_wall)

    # analytical parabolic profile
    posX_grid, posY_grid    = np.meshgrid(x_centers_analyzed, y_centers_analyzed)
    analytical_profile      = lambda x, y: (fforce_x / (2.0 * visc)) * y_from_wall * (H - y_from_wall)
    analytical_velocity     = analytical_profile(posX_grid,posY_grid)
    error                   = simulated_velocity_fluid - analytical_velocity


    # Plot comparison
    if plot:
        u_max = H ** 2 * fforce_x / (8 * visc)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        # Plot the velocity profile on the first subplot
        plt.style.use('default')

        axes[0].plot(simulated_velocity_fluid / u_max, y_centers_analyzed, 'ro-',
                     label='LBM Simulation', markersize=16, alpha=0.8, markerfacecolor='none')
        axes[0].plot(analytical_velocity / u_max, y_centers_analyzed, 'bs-',
                     label='Analytical Solution', linewidth=2, markersize=8, markerfacecolor='none')
        axes[0].set_xlabel('Velocity/Umax [m/s]')
        axes[0].set_ylabel('Vertical Position [m]')
        axes[0].set_title('Poiseuille Flow: Fluid Cells Only')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Plot the error on the second subplot
        axes[1].semilogx(np.abs(error), y_centers_analyzed, 'ko-', linewidth=2, markersize=16)
        axes[1].set_xlabel('Absolute Error [m/s]')
        axes[1].set_title('Error Profile (log scale)')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()  # Adjusts subplot params for a tight layout
        plt.show()

    # Calculate errors
    #l2_error = np.sqrt(np.mean((analytical_velocity - simulated_velocity_fluid) ** 2))
    l2_error    = (1/np.sqrt(1 * dim_y - 2)) * np.sum(np.sqrt(error**2))
    max_error   = 100*np.max(np.abs(error)/analytical_velocity)
    print(f"=== ANALYSIS (Fluid Cells Only) ===")
    print(f"Number of fluid cells evaluated: {len(y_fluid_idx)}")
    print(f"L2 Error: {l2_error:.6e} m/s")
    print(f"Max Error: {max_error:.6e} m/s")

    return l2_error

import pandas as pd
import matplotlib.pyplot as plt

def plot_infos(data, time, headers, title="Data Overview"):
    # First row = labels
    headers     = headers[1:]
    infos       = np.array(data).T
    # Create subplots
    fig, axes = plt.subplots(len(headers), 1, figsize=(8, 2.5*len(headers)), dpi=120, sharex=True)
    if len(headers) == 1:
        axes = [axes]

    for ax, values, label in zip(axes, infos, headers ):
        ax.plot(time, values, 'o-', linewidth=1.8, label=label, markersize=16)
        ax.set_ylabel(label)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="best")
        ax.set_ylim(min(min(values)*0.8,min(values))*1.1, max(max(values)*0.8,max(values)*1.1))
    
    axes[-1].set_xlabel('Time (s)')  # time label
    fig.suptitle(title, fontsize=14, weight="bold")
    plt.tight_layout()
    plt.show()

# Your file paths (correct dimensions)
files_solid = [((16,1), r"C:\Users\gabri\OneDrive\Documentos\LBM\output\Scale_1_16_solid.csv"),
         ((32,1), r"C:\Users\gabri\OneDrive\Documentos\LBM\output\Scale_1_32_solid.csv"),
         ((64,1), r"C:\Users\gabri\OneDrive\Documentos\LBM\output\Scale_1_64_solid.csv")]

files_x = [((16,1), r"C:\Users\gabri\OneDrive\Documentos\LBM\output\Scale_1_16_X.csv"),
         ((32,1), r"C:\Users\gabri\OneDrive\Documentos\LBM\output\Scale_1_32_X.csv"),
         ((64,1), r"C:\Users\gabri\OneDrive\Documentos\LBM\output\Scale_1_64_X.csv")]

files_y = [((16,1), r"C:\Users\gabri\OneDrive\Documentos\LBM\output\Scale_1_16_Y.csv"),
         ((32, 1), r"C:\Users\gabri\OneDrive\Documentos\LBM\output\Scale_1_32_Y.csv"),
         ((64, 1), r"C:\Users\gabri\OneDrive\Documentos\LBM\output\Scale_1_64_Y.csv")]

files_momentum = [((16,1), r"C:\Users\gabri\OneDrive\Documentos\LBM\output\Scale_1_16_momentum.csv"),
                ((32, 1), r"C:\Users\gabri\OneDrive\Documentos\LBM\output\Scale_1_32_momentum.csv"),
                ((64, 1), r"C:\Users\gabri\OneDrive\Documentos\LBM\output\Scale_1_64_momentum.csv")]

files_mass = [r"C:\Users\gabri\OneDrive\Documentos\LBM\output\Scale_1_16_infos.csv",
                r"C:\Users\gabri\OneDrive\Documentos\LBM\output\Scale_1_32_infos.csv",
                r"C:\Users\gabri\OneDrive\Documentos\LBM\output\Scale_1_64_infos.csv"]

# Physical parameters
visc        = 1e-3 # Physical viscosity
L           = 1.0  # Physical domain size
u0          = 0.05 # Physical characteristic velocity
T           = 5000.0
fforce_x    = u0*visc*8/(L**2)

scale_error     = []
scale_dim       = []

for (shape, file_x), (_, file_y), (_, file_solid), file_mass in zip(files_x, files_y, files_solid, files_mass):
    dim_y, dim_x    = shape  # Grid length: dim_y = rows, dim_x = columns

    # Load data
    time, vel_x_hist, _   = process_csv(file_x, (dim_y, dim_x))
    _,    vel_y_hist, _   = process_csv(file_y, (dim_y, dim_x))
    _,    solid, _        = process_csv(file_solid, (dim_y, dim_x))
    _,    mass, headers   = process_csv(file_mass, headers=True)

    # Calculate viscosity (from your C++ code)
    delta_t     = time[1] - time[0]
    print("delta_t: ",delta_t)
    delta_h     = L / dim_y
    print(f"\n=== Analyzing: {dim_y}x{dim_x} grid ===")
    print(f"delta_h: {delta_h:.6f}, Grid: {dim_y} rows x {dim_x} columns, u0 = {u0}")

    # Plot mass
    plot_infos(mass, time, headers, "Informations")

    # Plot results
    plot_vertical_velocity_profile(vel_x_hist, dim_y, delta_h, 10,'viridis',f"X-Velocity Profile - {dim_y}x{dim_x}")

    # Plot center cell evolution
    center_row, center_col = dim_y // 2, dim_x // 2
    plot_cell_velocity_time_series(vel_x_hist, center_row, center_col, time, "X-Velocity Evolution")

    # Compare with analytical solution
    l2 = compare_to_analytical(vel_x_hist[-1], dim_y, delta_h, delta_t, fforce_x, visc, plot=True)
    scale_error.append(l2)
    scale_dim.append(dim_y)

print("scale error:", scale_error)
print("scale dim: ",scale_dim)
scale_error_convergence(scale_dim, scale_error, plot=True)
