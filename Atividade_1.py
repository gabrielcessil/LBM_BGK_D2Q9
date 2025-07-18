import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

def process_single_matrix_file(file_path, matrix_shape):

    # Read CSV file (no header, each column is a time step)
    data = pd.read_csv(file_path, header=None).values

    # Reshape each time step
    time_steps = data.shape[0]
    matrixes_t = [data[t, 1:].reshape(matrix_shape) for t in range(time_steps)]
    time = np.array([data[t, 0] for t in range(time_steps)])
    return time, matrixes_t


# Example usage - replace with your actual file path and dimensions
# Use raw string (r"") or forward slashes for paths
files = [((32,32), r"/home/gabriel/Desktop/Implementando_LBM/Scale_32_32.csv"),
         ((64,64), r"/home/gabriel/Desktop/Implementando_LBM/Scale_64_64.csv"),
         ((128,128), r"/home/gabriel/Desktop/Implementando_LBM/Scale_128_128.csv")]


def plot_vel_hist(vel_simu, vel_anal):

    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(vel_simu, marker='o', label='lbm')
    plt.plot(vel_anal, marker='*', label='analítico')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xlabel("Time step")
    plt.ylabel("Relative $L_2$ error")
    plt.title("Velocity in a single cell")
    plt.show()



scale_error = []
scale_dim_x = []
for shape, file in files:
    time, vel_hist = process_single_matrix_file(file, shape)

    visc    = 10**-3
    L       = 1.0
    u0      = 0.05
    T       = 20.0
    dim_x   = shape[0]
    dim_y   = shape[1]
    delta_h = L / dim_x
    
    uy_analytical = lambda time, posX, posY: (u0 * np.sin(2 * np.pi * posX / L) * np.exp(-visc * time * (2* np.pi / L) ** 2))

    
    # Plot velocidade inicial em uma linha (row=0)
    plt.figure(figsize=(10, 6))
    velocity = vel_hist[0][0,:]
    plt.plot(np.arange(dim_x)*delta_h, vel_hist[0][0,:], color='blue', alpha = 1)
    for i, vel in enumerate(vel_hist[1:]):    
        plt.plot(np.arange(dim_x)*delta_h, vel[0,:], color='blue', alpha = 0.1*(len(vel_hist)-i) / len(vel_hist))
    plt.grid(True)
    plt.show()

    # Plot em certa celula ao longo do tempo
    col_posX    = int(dim_x / 4)
    posX        = delta_h*col_posX+  delta_h/2
    row_posY    = int(dim_y / 4)
    posY        = delta_h * row_posY+ delta_h/2
    uy_posX_anal = uy_analytical(time, posX, posY)
    uy_posX_simu = np.array([mat[row_posY][col_posX] for mat in vel_hist])
    plot_vel_hist(uy_posX_simu, uy_posX_anal)
    

    # Error order analysis
    time_step   = time[1] - time[0]
    k_analyzed  = int(np.log(0.5) / (-visc * time_step * (2 * np.pi / L) ** 2))
    t_analyzed  = k_analyzed * time_step
    v           = vel_hist[k_analyzed]
    posX_values = np.arange(0, dim_x)* delta_h + delta_h/2  # From 0 to L
    posY_values = np.arange(0, dim_y)* delta_h + delta_h/2  # From 0 to L
    
    posX_grid, posY_grid = np.meshgrid(posX_values, posY_values)
    v_anal      = uy_analytical(t_analyzed, posX_grid, posY_grid)
    L2_norm     = np.sqrt( np.mean( (v  -v_anal)**2))
    print("L2_norm: ", L2_norm)

    scale_error.append(L2_norm)
    scale_dim_x.append(dim_x)

# Convert to log-log
log_dim_x = np.log10(scale_dim_x)
log_error = np.log10(scale_error)

# Linear regression in log-log space: log(error) = slope * log(N) + intercept
slope, intercept = np.polyfit(log_dim_x, log_error, 1)

# Generate fitted line
fit_error = 10**(slope * log_dim_x + intercept)

# Plot data and regression
plt.figure(figsize=(6, 4),dpi=300)
plt.loglog(scale_dim_x, scale_error, 'o-', label='Measured error')
plt.loglog(scale_dim_x, fit_error, '--', label=f'Fit: slope = {slope:.6f}')
plt.xlabel("Grid size (N)")
plt.ylabel("Relative $L_2$ error")
plt.title("Convergence plot")
plt.grid(True, which="both", ls="--")
plt.legend()
for x, y in zip(scale_dim_x, scale_error):
    text = f"Grid Size={x}\n Error={y:.3e}"
    plt.annotate(text,
                 xy=(x, y),
                 xytext=(5, 5), textcoords='offset points',
                 fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.2", edgecolor='black', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()
