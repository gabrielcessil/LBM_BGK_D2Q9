#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <tuple>
#include <fstream>
#include <iomanip>
#include <string>


LBM_BGK_D2Q9::LBM_BGK_D2Q9(double Tau_, double spatial_step, double time_step, int dim_x_, int dim_y_)
        : Tau(Tau_), delta_t(time_step), delta_h(spatial_step), dim_x(dim_x_), dim_y(dim_y_) {
        // Initialize probability matrices with zeros
        particles_eq_matrixes.resize(9 * dim_x * dim_y, 0.0);
        particles_prob_matrixes.resize(2, std::vector<double>(9 * dim_x * dim_y, 0.0));


        // Initialize macroscopic matrixes with zeros
        velocity_X_matrix.resize(dim_y, std::vector<double>(dim_x, 0.0));
        velocity_Y_matrix.resize(dim_y, std::vector<double>(dim_x, 0.0));
        momentum_X_matrix.resize(dim_y, std::vector<double>(dim_x, 0.0));
        momentum_Y_matrix.resize(dim_y, std::vector<double>(dim_x, 0.0));
        density_matrix.resize(dim_y, std::vector<double>(dim_x, 1.0)); // Default density = 1.0

        e = {
            {0.0, 0.0}, {1.0, 0.0}, {-1.0, 0.0}, {0.0, 1.0}, {0.0, -1.0},
            {1.0, 1.0}, {-1.0, -1.0}, {-1.0, 1.0}, {1.0, -1.0}
        };

        c.resize(9, std::vector<double>(2));
        for (int i = 0; i < 9; ++i) {
            c[i][0] = e[i][0];
            c[i][1] = e[i][1];
        }

        weights = {4.0 / 9.0,
                   1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
                   1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};

        cs = 1 / std::sqrt(3.0);
}

// Handle uni-dimensional probability matrixes
int LBM_BGK_D2Q9::get_index(int direction, int row, int col) {
    return (row * dim_x + col) * 9 + direction;
}
double LBM_BGK_D2Q9::access_matrix(const std::vector<double>& matrix, int direction, int row, int col) {
    return matrix[get_index(direction, row, col)];
}
void LBM_BGK_D2Q9::set_matrix(double value, std::vector<double>& matrix, int direction, int row, int col) {
    matrix[get_index(direction, row, col)] = value;
}

// Handle macroscopic matrixes
void LBM_BGK_D2Q9::update_density() {
    for (int col = 0; col < dim_x; col++) {
        for (int row = 0; row < dim_y; row++) {
            density_matrix[row][col] = 0.0;
            for (int i = 0; i < 9; i++) {
                density_matrix[row][col] += access_matrix(particles_prob_matrixes[prob_scheduler], i, row, col);
            }
        }
    }
}
void LBM_BGK_D2Q9::update_momentum() {
    for (int col = 0; col < dim_x; col++) {
        for (int row = 0; row < dim_y; row++) {
            momentum_X_matrix[row][col] = 0.0;
            momentum_Y_matrix[row][col] = 0.0;
            for (int i = 0; i < 9; i++) {
                double f_i = access_matrix(particles_prob_matrixes[prob_scheduler], i, row, col);
                momentum_X_matrix[row][col] += f_i * c[i][0];
                momentum_Y_matrix[row][col] += f_i * c[i][1];
            }
        }
    }
}

void LBM_BGK_D2Q9::update_feq() {
    for (int col = 0; col < dim_x; col++) {
        for (int row = 0; row < dim_y; row++) {
            double rho  = density_matrix[row][col];
            double ux   = velocity_X_matrix[row][col];
            double uy   = velocity_Y_matrix[row][col];


            for (int i = 0; i < 9; i++) {
                double ci_dot_u     = c[i][0] * ux + c[i][1] * uy;
                double u_dot_u      = ux * ux + uy * uy;
                double feq          = rho * weights[i] * (1.0 + 3.0 * ci_dot_u + 4.5 * ci_dot_u * ci_dot_u - 1.5 * u_dot_u);
                set_matrix(feq, particles_eq_matrixes, i, row, col);
            }
        }
    }
}

void LBM_BGK_D2Q9::collide() {
    for (int col = 0; col < dim_x; col++) {
        for (int row = 0; row < dim_y; row++) {
            for (int i = 0; i < 9; ++i) {
                double f_i      = access_matrix(particles_prob_matrixes[prob_scheduler], i, row, col);
                double feq_i    = access_matrix(particles_eq_matrixes, i, row, col);
                double updated  = f_i - (f_i - feq_i) / Tau ;
                set_matrix(updated, particles_prob_matrixes[prob_scheduler], i, row, col);
            }
        }
    }
}

void LBM_BGK_D2Q9::propagate(){

    for (int col = 0; col < dim_x; col++) {
        for (int row = 0; row < dim_y; row++) {
            for (int i = 0; i < 9; i++) {
                int dx = e[i][0];
                int dy = e[i][1];
                int new_col = (col + dx + dim_x) % dim_x;
                int new_row = (row + dy + dim_y) % dim_y;
                double val  = access_matrix(particles_prob_matrixes[prob_scheduler], i, row, col); // Get value from propagated cell
                set_matrix(val, particles_prob_matrixes[(prob_scheduler+1) % 2], i, new_row, new_col);    // Set value into the new cell
            }
        }
    }
    prob_scheduler = (prob_scheduler+1)%2;

}

void LBM_BGK_D2Q9::update_velocity(){
    for (int col = 0; col < dim_x; col++) {
        for (int row = 0; row < dim_y; row++) {
            velocity_X_matrix[row][col] = momentum_X_matrix[row][col] /  density_matrix[row][col];
            velocity_Y_matrix[row][col] = momentum_Y_matrix[row][col] /  density_matrix[row][col];
        }
    }
}
void LBM_BGK_D2Q9::run(int n_timesteps, const std::string& filename = "") {

    // Restart the file if filename is provided
    if (!filename.empty()) {
        std::ofstream clear_file(filename, std::ios::trunc);
        clear_file.close();
    }

    // 2. Set momentum to match initial velocity and density
    for (int col = 0; col < dim_x; col++) {
        for (int row = 0; row < dim_y; row++) {
            momentum_X_matrix[row][col] = density_matrix[row][col] * velocity_X_matrix[row][col];
            momentum_Y_matrix[row][col] = density_matrix[row][col] * velocity_Y_matrix[row][col];
        }
    }

    // Initialize f_eq
    update_feq();

    // Initialize f as f_eq
    for (int i = 0; i < 9; i++) {
        for (int col = 0; col < dim_x; col++) {
            for (int row = 0; row < dim_y; row++) {
                double val = access_matrix(particles_eq_matrixes, i, row, col); // Get value from propagated cell
                set_matrix(val, particles_prob_matrixes[prob_scheduler], i, row, col);    // Set value into the new cell
            }
        }
    }

    for (int t_time_step = 0; t_time_step < n_timesteps; t_time_step++) {
        // Save current timestep data if filename is provided
        if (!filename.empty()) {
            save_velocity_y_csv(filename, t_time_step*delta_t);
        }

        // Mesoscopic operations
        collide();
        propagate();
        // Set macroscopic consequences
        update_density();
        update_momentum();
        update_velocity();


        // Update Equilibrium state
        update_feq();



    }
}

void LBM_BGK_D2Q9::save_velocity_y_csv(const std::string& filename, double time_instant) {
    std::ofstream file(filename, std::ios::app);  // Open in append mode

    // Write time instant as first column
    file << std::scientific << std::setprecision(6) << time_instant << ",";

    // Write flattened velocity data
    for (int row = 0; row < dim_y; ++row) {
        for (int col = 0; col < dim_x; ++col) {
            file << std::scientific << std::setprecision(9) << velocity_Y_matrix[row][col] * delta_h / delta_t;
            if (row != dim_y - 1 || col != dim_x - 1) {
                file << ",";
            }
        }
    }
    file << "\n";
    file.close();
}
