import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def process_single_matrix_file(file_path, matrix_shape):
    data = pd.read_csv(file_path, header=None).values
    time_steps = data.shape[0]
    matrixes_t = [data[t, 1:].reshape(matrix_shape) for t in range(time_steps)]
    time = np.array([data[t, 0] for t in range(time_steps)])
    return time, matrixes_t


def plot_vertical_velocity_profile(vel_hist, dim_y, delta_h, title="Vertical Velocity Profile"):
    """Plot velocity profiles at different times"""
    plt.figure(figsize=(10, 6))
    y_positions = np.linspace(0, dim_y * delta_h, dim_y)

    # Plot selected time steps
    time_indices = [0, len(vel_hist) // 4, len(vel_hist) // 2, 3 * len(vel_hist) // 4, len(vel_hist) - 1]
    colors = ['green', 'blue', 'orange', 'purple', 'red']
    labels = ['t=0', 't=25%', 't=50%', 't=75%', 't=final']

    for i, idx in enumerate(time_indices):
        # Take velocity at first column (x=0)
        velocity_profile = vel_hist[idx][:, 0]
        plt.plot(velocity_profile, y_positions, color=colors[i], alpha=0.8, label=labels[i], linewidth=2)

    plt.xlabel('Velocity (u) [m/s]')
    plt.ylabel('Vertical Position (y) [m]')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_cell_velocity_time_series(vel_hist, row, col, time, title="Velocity Time Series"):
    """Plot velocity evolution at a specific cell"""
    cell_velocity = np.array([mat[row][col] for mat in vel_hist])

    plt.figure(figsize=(10, 6))
    plt.plot(time, cell_velocity, 'b-', linewidth=2, label='Simulated')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.title(f'{title} at cell ({row}, {col})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_analytical_vs_simulated(vel_hist_final, dim_y, delta_h, delta_t, fforce_x, visc):
    """
    Compare ONLY fluid cells (exclude solid boundary cells)
    """
    # indices of fluid nodes (exclude top/bottom solid cells)
    fluid_idx = np.arange(1, dim_y - 1)

    # centres for all nodes and centres for fluid nodes
    y_centres = (np.arange(dim_y) + 0.5) * delta_h
    y_centres_fluid = y_centres[fluid_idx]

    # bottom wall position (midpoint between node 0 and 1)
    y_wall_bottom = delta_h  # (0.5 + 1.5)/2 * delta_h = delta_h

    # distance from bottom wall to fluid centers (this is the y used in analytic formula)
    y_from_wall = y_centres_fluid - y_wall_bottom
    # Channel height between walls
    H = (dim_y - 2) * delta_h

    # simulated u (x-velocity) only for fluid cells
    # be explicit about indexing for numpy arrays
    simulated_velocity_fluid = np.asarray(vel_hist_final)[fluid_idx, 0]

    # analytical parabolic profile
    analytical_velocity = (fforce_x / (2.0 * visc)) * y_from_wall * (H - y_from_wall)

    u_max = H ** 2 * fforce_x / (8 * visc)
    plt.figure(figsize=(12, 6))

    # Plot comparison
    plt.subplot(1, 2, 1)
    plt.plot(simulated_velocity_fluid/u_max, y_centres_fluid, 'ro-',
             label='LBM Simulation', markersize=4, alpha=0.8)
    plt.plot(analytical_velocity/u_max, y_centres_fluid, 'b-',
             label='Analytical Solution', linewidth=2)
    plt.xlabel('Velocity/Umax [m/s]')
    plt.ylabel('Vertical Position [m]')
    plt.title('Poiseuille Flow: Fluid Cells Only')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot error
    plt.subplot(1, 2, 2)
    error = np.abs(simulated_velocity_fluid - analytical_velocity)
    plt.semilogy(y_centres_fluid, error, 'k-', linewidth=2)
    plt.xlabel('Vertical Position [m]')
    plt.ylabel('Absolute Error [m/s]')
    plt.title('Error Profile (log scale)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Calculate errors
    l2_error = np.sqrt(np.mean(error**2))
    max_error = np.max(error)
    max_vel_analytical = np.max(analytical_velocity)

    print(f"=== ANALYSIS (Fluid Cells Only) ===")
    print(f"Number of fluid cells evaluated: {len(fluid_idx)}")
    print(f"L2 Error: {l2_error:.6e} m/s")
    print(f"Max Error: {max_error:.6e} m/s")
    print(f"Relative L2 Error: {l2_error/max_vel_analytical:.3%}")
    print(f"Relative Max Error: {max_error/max_vel_analytical:.3%}")


# Your file paths (correct dimensions)
files_solid = [((32,1), r"C:\Users\gabri\OneDrive\Documentos\LBM\output\Scale_1_32_solid.csv"),
         ((64,1), r"C:\Users\gabri\OneDrive\Documentos\LBM\output\Scale_1_64_solid.csv"),
         ((128,1), r"C:\Users\gabri\OneDrive\Documentos\LBM\output\Scale_1_128_solid.csv")]

files_x = [((32,1), r"C:\Users\gabri\OneDrive\Documentos\LBM\output\Scale_1_32_X.csv"),
         ((64,1), r"C:\Users\gabri\OneDrive\Documentos\LBM\output\Scale_1_64_X.csv"),
         ((128,1), r"C:\Users\gabri\OneDrive\Documentos\LBM\output\Scale_1_128_X.csv")]

files_y = [((32,1), r"C:\Users\gabri\OneDrive\Documentos\LBM\output\Scale_1_32_Y.csv"),
         ((64, 1), r"C:\Users\gabri\OneDrive\Documentos\LBM\output\Scale_1_64_Y.csv"),
         ((128, 1), r"C:\Users\gabri\OneDrive\Documentos\LBM\output\Scale_1_128_Y.csv")]

# Physical parameters
visc        = 1e-3 # Physical viscosity
L           = 1.0  # Physical domain size
fforce_x    = 0.01 # Physical field force


for (shape, file_x), (_, file_y), (_, file_solid) in zip(files_x, files_y, files_solid):
    dim_y, dim_x = shape  # Grid length: dim_y = rows, dim_x = columns

    # Load data
    time, vel_x_hist    = process_single_matrix_file(file_x, (dim_y, dim_x))
    _,    vel_y_hist    = process_single_matrix_file(file_y, (dim_y, dim_x))
    __,   solid         = process_single_matrix_file(file_solid, (dim_y, dim_x))

    # Calculate viscosity (from your C++ code)
    delta_t     = time[1] - time[0]
    delta_h     = L / dim_y
    u0          = ((L ** 2) * 1 * fforce_x) / (visc * 8)
    T           = 20.0
    print(f"\n=== Analyzing: {dim_y}x{dim_x} grid ===")
    print(f"delta_h: {delta_h:.6f}, Grid: {dim_y} rows x {dim_x} columns, u0 = {u0}")

    # Plot results
    #plot_vertical_velocity_profile(vel_x_hist, dim_y, delta_h, f"X-Velocity Profile - {dim_y}x{dim_x}")

    # Plot center cell evolution
    #center_row, center_col = dim_y // 2, dim_x // 2
    #plot_cell_velocity_time_series(vel_x_hist, center_row, center_col, time, "X-Velocity Evolution")

    # Compare with analytical solution
    plot_analytical_vs_simulated(vel_x_hist[-1], dim_y, delta_h, delta_t, fforce_x, visc)