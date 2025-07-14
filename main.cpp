#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <tuple>
#include <fstream>
#include <iomanip>
#include <string>

class LBM_BGK_D2Q9 {
public:
    // Inputs
    int dim_x, dim_y;
    double Tau, delta_t, delta_h, cs;

    // Declare matrixes operated in loop
    std::vector<double> particles_eq_matrixes;
    std::vector<std::vector<double>> particles_prob_matrixes;
    int prob_scheduler = 0; // 0 or 1
    std::vector<std::vector<double>> velocity_X_matrix;
    std::vector<std::vector<double>> velocity_Y_matrix;
    std::vector<std::vector<double>> density_matrix;
    std::vector<std::vector<double>> momentum_X_matrix;
    std::vector<std::vector<double>> momentum_Y_matrix;
    std::vector<std::vector<int>> e;
    std::vector<std::vector<double>> c;
    std::vector<double> weights;

    // Constructor remains unchanged
    LBM_BGK_D2Q9(double Tau_, double spatial_step, double time_step, int dim_x_, int dim_y_)
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
            {0, 0}, {1, 0}, {-1, 0}, {0, 1}, {0, -1},
            {1, 1}, {-1, -1}, {-1, 1}, {1, -1}
        };

        c.resize(9, std::vector<double>(2));
        for (int i = 0; i < 9; ++i) {
            c[i][0] = e[i][0] * delta_h / delta_t;
            c[i][1] = e[i][1] * delta_h / delta_t;
        }

        weights = {4.0 / 9.0,
                   1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
                   1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};

        cs = (delta_h / delta_t) / std::sqrt(3.0);
    }

    // Handle uni-dimensional probability matrixes
    int get_index(int direction, int row, int col) {
        return (row * dim_x + col) * 9 + direction;
    }
    double access_matrix(const std::vector<double>& matrix, int direction, int row, int col) {
        return matrix[get_index(direction, row, col)];
    }
    void set_matrix(double value, std::vector<double>& matrix, int direction, int row, int col) {
        matrix[get_index(direction, row, col)] = value;
    }

    // Handle macroscopic matrixes
    void update_density() {
        for (int col = 0; col < dim_x; col++) {
            for (int row = 0; row < dim_y; row++) {
                density_matrix[row][col] = 0.0;
                for (int i = 0; i < 9; i++) {
                    density_matrix[row][col] += access_matrix(particles_prob_matrixes[prob_scheduler], i, row, col);
                }
            }
        }
    }
    void update_momentum() {
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

    void update_feq() {
        for (int col = 0; col < dim_x; col++) {
            for (int row = 0; row < dim_y; row++) {
                double rho  = density_matrix[row][col];
                double ux   = velocity_X_matrix[row][col];
                double uy   = velocity_Y_matrix[row][col];
                

                for (int i = 0; i < 9; i++) {
                    double ci_dot_u     = c[i][0] * ux + c[i][1] * uy;
                    double u_dot_u      = ux * ux + uy * uy;
                    double feq          = rho * weights[i] * (1 + 3 * ci_dot_u + 4.5 * ci_dot_u * ci_dot_u - 1.5 * u_dot_u);
                    set_matrix(feq, particles_eq_matrixes, i, row, col);
                }
            }
        }
    }

    void collide() {
        for (int col = 0; col < dim_x; col++) {
            for (int row = 0; row < dim_y; row++) {
                for (int i = 0; i < 9; ++i) {
                    double f_i      = access_matrix(particles_prob_matrixes[prob_scheduler], i, row, col);
                    double feq_i    = access_matrix(particles_eq_matrixes, i, row, col);
                    double updated  = f_i - (f_i - feq_i) / (Tau );
                    set_matrix(updated, particles_prob_matrixes[prob_scheduler], i, row, col);
                }
            }
        }
    }

    void propagate(){
        int new_scheduler = (prob_scheduler+1) % 2;
        for (int i = 0; i < 9; i++) {
                int dx = e[i][0];
                int dy = e[i][1];
                for (int col = 0; col < dim_x; col++) {
                    for (int row = 0; row < dim_y; row++) {
                        int new_col = (col + dx + dim_x) % dim_x;
                        int new_row = (row + dy + dim_y) % dim_y;
                        double val = access_matrix(particles_prob_matrixes[prob_scheduler], i, row, col); // Get value from propagated cell
                        set_matrix(val, particles_prob_matrixes[new_scheduler], i, new_row, new_col);    // Set value into the new cell
                    }
                }
            }
        prob_scheduler = new_scheduler;
    }

    void update_velocity(){
        for (int col = 0; col < dim_x; col++) {
            for (int row = 0; row < dim_y; row++) {
                velocity_X_matrix[row][col] = momentum_X_matrix[row][col] /  density_matrix[row][col];
                velocity_Y_matrix[row][col] = momentum_Y_matrix[row][col] /  density_matrix[row][col];
            }
        }
    }
    void run(int n_timesteps, const std::string& filename = "") {

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

    void save_velocity_y_csv(const std::string& filename, double time_instant) {
        std::ofstream file(filename, std::ios::app);  // Open in append mode
        
        // Write time instant as first column
        file << std::scientific << std::setprecision(6) << time_instant << ",";
        
        // Write flattened velocity data
        for (int row = 0; row < dim_y; ++row) {
            for (int col = 0; col < dim_x; ++col) {
                file << std::scientific << std::setprecision(6) << velocity_Y_matrix[row][col];
                if (row != dim_y - 1 || col != dim_x - 1) {
                    file << ",";
                }
            }
        }
        file << "\n";
        file.close();
    }
};


// Helper function to initialize velocity field
void initialize_velocity_field(LBM_BGK_D2Q9& lbm, double u0) {
    const double PI = 3.14159265358979323846; 

    for (int col = 0; col < lbm.dim_x; col++) {
        double val = u0 * sin(2.0 * PI * col*lbm.delta_h / (lbm.dim_x*lbm.delta_h));
        for (int row = 0; row < lbm.dim_y; row++) {
            lbm.velocity_X_matrix[row][col] = 0.0;
            lbm.velocity_Y_matrix[row][col] = val;
        }
    }
}

int main() {
    // Physical parameters
    const double visc       = 1 * std::pow(10,-3);  // Oil visc
    const double L          = 1;                    // Characteristic length
    const double u0         = 0.05;                // Characteristic velocity
    const double T_real     = 20;                   // Simulated time
    //const double Re         = 50;                 // Reynolds number

    // Simulation parameters
    const double u_star     = 0.05;    // 
    const int dim_x_base    = 32;       // Base resolution (scaled in the loop)
    const int n_scales      = 4;        // Number of scales to test
    //const double Tau        = 0.8;      // Relaxation time

    std::vector<double> errors;
    std::vector<int> scales;

    for (int i = 0; i < n_scales; i++) {
        int scale = pow(2, i);
        scales.push_back(scale);
        std::cout << "\ni: "<< i <<", Scale: " << scale <<std::endl;

        int dim_x = dim_x_base * scale;
        int dim_y = dim_x_base * scale;

        std::cout << "dim_x: " << dim_x << ", dim_y: " << dim_y << std::endl;

        double spatial_step = L / dim_x;
        std::cout << "spatial_step: " << spatial_step << std::endl;
        
        double time_step    = (u_star/u0)*spatial_step;
        std::cout << "time_step: " << time_step << std::endl;
        
        double cs           = (spatial_step/time_step)/std::sqrt(3.0);
        std::cout << "cs: " << cs << std::endl;

        double Tau = visc/(std::pow(cs,2)*time_step) + 0.5;
        std::cout << "Tau: " << Tau << std::endl;

        int n_timesteps     = static_cast<int>(T_real / time_step);
        std::cout << "n_timesteps: " << n_timesteps << std::endl;

        // Sanity check
        double Re = u0*L/visc;
        std::cout << "Re: " << Re << std::endl;



        // Create and initialize simulation
        LBM_BGK_D2Q9 lbm(Tau, spatial_step, time_step, dim_x, dim_y);

        initialize_velocity_field(lbm, u0);
        
        // Run simulation
        lbm.run(n_timesteps, "Scale_" + std::to_string(dim_x)+"_"+std::to_string(dim_y)+".csv");
    }



    return 0;
}