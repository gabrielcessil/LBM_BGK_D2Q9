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
    double Tau, delta_t, delta_h, cs, u_star;

    // Declare matrixes operated in loop
    std::vector<double> particles_eq_matrixes;
    std::vector<std::vector<double>> particles_prob_matrixes;
    int prob_scheduler = 0; // 0 or 1
    std::vector<std::vector<double>> velocity_X_matrix;
    std::vector<std::vector<double>> velocity_Y_matrix;
    std::vector<std::vector<double>> density_matrix;
    std::vector<std::vector<double>> momentum_X_matrix;
    std::vector<std::vector<double>> momentum_Y_matrix;
    std::vector<std::vector<double>> e;
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
            {0.0, 0.0},
            {1.0, 0.0}, {-1.0, 0.0}, {0.0, 1.0}, {0.0, -1.0},
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
    int get_index(int direction, int row, int col) {
        return direction + row*(9*dim_x) + col*9;
    }
    double access_matrix(const std::vector<double>& matrix, int direction, int row, int col) {
        return matrix[get_index(direction, row, col)];
    }
    void set_matrix(double value, std::vector<double>& matrix, int direction, int row, int col) {
        matrix[get_index(direction, row, col)] = value;
    }

    // Handle macroscopic matrixes
    void update_density() {
        for (int row = 0; row < dim_y; row++) {
            for (int col = 0; col < dim_x; col++) {
                density_matrix[row][col] = 0.0;
                for (int i = 0; i < 9; i++) {
                    density_matrix[row][col] += access_matrix(particles_prob_matrixes[prob_scheduler], i, row, col);
                }
            }
        }
    }
    void update_momentum() {
        for (int row = 0; row < dim_y; row++) {
            for (int col = 0; col < dim_x; col++) {
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
        for (int row = 0; row < dim_y; row++) {
            for (int col = 0; col < dim_x; col++) {
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

    void collide() {
        for (int row = 0; row < dim_y; row++) {
            for (int col = 0; col < dim_x; col++) {        
                for (int i = 0; i < 9; ++i) {
                    double f_i      = access_matrix(particles_prob_matrixes[prob_scheduler], i, row, col);
                    double feq_i    = access_matrix(particles_eq_matrixes, i, row, col);
                    double updated  = f_i - (f_i - feq_i) / Tau ;
                    set_matrix(updated, particles_prob_matrixes[prob_scheduler], i, row, col);
                }
            }
        }
    }


    void propagate(){


        for (int row = 0; row < dim_y; row++) {
            for (int col = 0; col < dim_x; col++) {
                for (int i = 0; i < 9; i++) {
                    int dx = e[i][0];
                    int dy = e[i][1];
                    int new_col = (col + dx + dim_x) % dim_x;
                    int new_row = (row - dy + dim_y) % dim_y; // Aumentar row eh diminuir y,
                    double val  = access_matrix(particles_prob_matrixes[prob_scheduler], i, row, col);          // Get value from propagated cell
                    set_matrix(val, particles_prob_matrixes[(prob_scheduler+1) % 2], i, new_row, new_col);      // Set value into the new cell
                }
            }
        }
        prob_scheduler = (prob_scheduler+1)%2;

    }

    void update_velocity(){
        for (int row = 0; row < dim_y; row++) {
            for (int col = 0; col < dim_x; col++) {
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
                density_matrix[row][col] = 1;
                momentum_X_matrix[row][col] = density_matrix[row][col] * velocity_X_matrix[row][col];
                momentum_Y_matrix[row][col] = density_matrix[row][col] * velocity_Y_matrix[row][col];
                
            }
        }

        // Initialize f_eq
        update_feq();


        // Initialize f as f_eq
        for (int row = 0; row < dim_y; row++) {
            for (int col = 0; col < dim_x; col++) {
                for (int i = 0; i < 9; i++) {
                    double val = access_matrix(particles_eq_matrixes, i, row, col); // Get value from propagated cell
                    set_matrix(val, particles_prob_matrixes[prob_scheduler], i, row, col);    // Set value into the new cell
                }
            }
        }

        verify_momentum(); // DEBUG

        for (int t_time_step = 0; t_time_step < n_timesteps; t_time_step++) {
            // Save current timestep data if filename is provided
            if (!filename.empty()) {
                save_velocity_y_csv(filename, t_time_step*delta_t);
            }

            //std::cout << compute_total_mass() << std::endl;
            //std::cout << compute_total_momentum_x() << std::endl;
            
            // Mesoscopic operations
            collide();
            propagate();
            // Update macroscopic consequences
            update_density();
            update_momentum();            
            update_velocity();

            // verify_momentum(); // DEBUG

            // Update Equilibrium state
            update_feq();



        }
    }

    void save_velocity_y_csv(const std::string& filename, double time_instant) {
        std::ofstream file(filename, std::ios::app);  // Open in append mode

        // Write time instant as first column
        file << std::scientific << std::setprecision(6) << time_instant << ",";

        // Write flattened velocity data
        for (int row = 0; row < dim_y; row++) {
            for (int col = 0; col < dim_x; col++) {
                file << std::scientific << std::setprecision(9) << velocity_Y_matrix[row][col] * delta_h / delta_t;
                if (row != dim_y - 1 || col != dim_x - 1) {
                    file << ",";
                }
            }
        }
        file << "\n";
        file.close();
    }

    double compute_total_mass() {
        double total_mass = 0.0;
        for (int row = 0; row < dim_y; row++) {
            for (int col = 0; col < dim_x; col++) {
                total_mass += density_matrix[row][col];
            }
        }
        return total_mass;
    }

    void verify_momentum() {
        double total_yy = 0.0, total_xx = 0.0;
        for (int row = 0; row < dim_y; row++) {
            for (int col = 0; col < dim_x; col++) {
                total_xx += momentum_X_matrix[row][col];
                total_yy += momentum_Y_matrix[row][col];
            }
        }

        double total_x = 0.0;
        double total_y = 0.0; 
        for (int col = 0; col < dim_x; col++) {
            for (int row = 0; row < dim_y; row++) {
                total_x += density_matrix[row][col] * velocity_X_matrix[row][col];
                total_y += density_matrix[row][col] * velocity_Y_matrix[row][col];
            }
        }
        std::cout << "Y-Momentum: " << total_yy << ", " << total_y << "." << std::endl;
        std::cout << "X-Momentum: " << total_xx << ", " << total_x << "." << std::endl << std::endl;
    }

    double compute_total_momentum_x() {
        double total = 0.0;
        for (int row = 0; row < dim_y; row++) {
            for (int col = 0; col < dim_x; col++) {
                total += momentum_X_matrix[row][col];
            }
        }
        return total;
    }

};


// Helper function to initialize velocity field
void initialize_velocity_field(LBM_BGK_D2Q9& lbm, double u_star) {
    const double PI = 3.14159265358979323846;
    lbm.u_star = u_star;

    for (int col = 0; col < lbm.dim_x; col++) {
        double val = u_star * sin(2.0 * PI * (col*lbm.delta_h+lbm.delta_h/2) / (lbm.dim_x*lbm.delta_h));
        for (int row = 0; row < lbm.dim_y; row++) {
            lbm.velocity_X_matrix[row][col] = 0.0;
            lbm.velocity_Y_matrix[row][col] = val;
        }
    }
}

void propagation_test() {
    int dim_x = 3, dim_y=3;
    // Create and initialize simulation
    LBM_BGK_D2Q9 lbm(0.8, 1, 1, dim_x, dim_y);
    for (int i_selected = 0; i_selected < 9; i_selected++){
        std::cout <<std::endl<<std::endl << "Direction "<< i_selected <<std::endl;

        for (int row = 0; row < dim_y; row++) {
            for (int col = 0; col < dim_x; col++) {
                for (int i = 0; i < 9; i++) {
                    if (i == i_selected){
                        lbm.set_matrix((row+1)*std::pow(10,col),lbm.particles_prob_matrixes[lbm.prob_scheduler], i, row, col);
                    }
                    else{
                        lbm.set_matrix(0,lbm.particles_prob_matrixes[lbm.prob_scheduler], i, row, col);
                    }
                }

            }
        }
        for (int row = 0; row < dim_y; row++) {
            for (int col = 0; col < dim_x; col++) {
                for (int i = 0; i < 9; i++) {
                    std::cout << lbm.access_matrix(lbm.particles_prob_matrixes[lbm.prob_scheduler], i, row, col) << ", ";
                }
                std::cout << " | ";
            }
            std::cout << std::endl;
        }
        lbm.propagate();
        std::cout <<"Propagated." <<std::endl;

        for (int row = 0; row < dim_y; row++) {
            for (int col = 0; col < dim_x; col++) {
                for (int i = 0; i < 9; i++) {
                    std::cout << lbm.access_matrix(lbm.particles_prob_matrixes[lbm.prob_scheduler], i, row, col) << ", ";
                }
                std::cout << " | ";
            }
            std::cout << std::endl;
        }
        lbm.propagate();
        std::cout <<"Propagated." <<std::endl;

        for (int row = 0; row < dim_y; row++) {
            for (int col = 0; col < dim_x; col++) {
                for (int i = 0; i < 9; i++) {
                    std::cout << lbm.access_matrix(lbm.particles_prob_matrixes[lbm.prob_scheduler], i, row, col) << ", ";
                }
                std::cout << " | ";
            }
            std::cout << std::endl;
        }
        lbm.propagate();
        std::cout <<"Propagated." <<std::endl;

        for (int row = 0; row < dim_y; row++) {
            for (int col = 0; col < dim_x; col++) {
                for (int i = 0; i < 9; i++) {
                    std::cout << lbm.access_matrix(lbm.particles_prob_matrixes[lbm.prob_scheduler], i, row, col) << ", ";
                }
                std::cout << " | ";
            }
            std::cout << std::endl;
        }
    }
}

int main() {
    // Physical parameters
    const double visc       = 1.0 * std::pow(10,-3);  // Oil visc
    const double D          = 1.0;                    // Characteristic length
    const double L          = 1.0;                    // Domain length
    const double u0         = 0.05;                   // Characteristic velocity

    std::cout << "Physical parameters: " << std::endl;
    std::cout << "visc: " << visc << std::endl;
    std::cout << "D: " << D << std::endl;
    std::cout << "L: " << L << std::endl;
    std::cout << "u0: " << u0 << std::endl;

    // Simulation parameters
    const double T_real     = 20.0;       // Simulated time
    const int dim_x_base    = 32;       // Base resolution (scaled in the loop)
    const int n_scales      = 4.0;        // Number of scales to test

    std::vector<double> errors;
    std::vector<int> scales;

    for (int i = 0; i < n_scales; i++) {
        int scale = pow(2, i);
        scales.push_back(scale);
        std::cout << "\ni: "<< i <<", Scale: " << scale <<std::endl;

        // Increasing grid  
        int dim_x = dim_x_base * scale;
        int dim_y = dim_x_base * scale;

        // Diffusion scalling
        const double spatial_step    = L/dim_x;
        const double tau_star        = 0.8;
        const double visc_star       = (1.0/3.0)*(tau_star-0.5);
        const double u_star          = (u0*D/visc)*(visc_star/dim_x);
        const double time_step       = (u_star/u0)*(L/dim_x);

        // Setting simulation
        int n_timesteps     = static_cast<int>(T_real / time_step);

        std::cout << "dim_x: " << dim_x << ", dim_y: " << dim_y << std::endl;
        std::cout << "spatial_step: " << spatial_step << std::endl;
        std::cout << "tau_star: " << tau_star << std::endl;
        std::cout << "visc_star: " << visc_star << std::endl;
        std::cout << "u_star: " << u_star << std::endl;
        std::cout << "time_step: " << time_step << std::endl;
        // Sanity check
        std::cout << "Re: "         << u0*D/visc << std::endl;
        std::cout << "Re_star: "    << u_star * dim_x / visc_star << std::endl;
        std::cout << "u_star: "     << u0 / (spatial_step / time_step) << std::endl;
        std::cout << "visc_star: "  << visc * time_step / (spatial_step*spatial_step) << std::endl;
        std::cout << "n_timesteps: " << n_timesteps << std::endl;


        // Create and initialize simulation
        LBM_BGK_D2Q9 lbm(tau_star, spatial_step, time_step, dim_x, dim_y);

        // Run simulation
        initialize_velocity_field(lbm, u_star);
        lbm.run(n_timesteps); // , "Scale_" + std::to_string(dim_x)+"_"+std::to_string(dim_y)+".csv");

        //propagation_test();

    }

    return 0;
}