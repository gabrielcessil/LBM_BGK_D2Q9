#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <tuple>
#include <fstream>
#include <iomanip>
#include <string>
#include <iomanip> 

class LBM_BGK_D2Q9 {
public:
    // Inputs
    int dim_x, dim_y;
    double Tau, delta_t, delta_h, cs, u_star;

    // Declare matrixes operated in loop
    std::vector<double> particles_eq_matrixes;
    std::vector<std::vector<double>> particles_prob_matrixes;
    int prob_scheduler = 0; // 0 or 1
    std::vector<std::vector<double>> solid_matrix; // 0 for solid, 1 for void
    std::vector<std::vector<double>> velocity_X_matrix;
    std::vector<std::vector<double>> velocity_Y_matrix;
    std::vector<std::vector<double>> density_matrix;
    std::vector<std::vector<double>> momentum_X_matrix;
    std::vector<std::vector<double>> momentum_Y_matrix;
    std::vector<std::vector<double>> e;
    std::vector<std::vector<double>> c;
    std::vector<double> weights;
    std::vector<int> opp_dir;

    double acc_x = 0, acc_y = 0;

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
        
        // Initialize domain as voids
        solid_matrix.resize(dim_y, std::vector<double>(dim_x, 1.0));

        e = {
            {0.0, 0.0},
            {1.0, 0.0}, {-1.0, 0.0}, 
            {0.0, 1.0}, {0.0, -1.0},
            {1.0, 1.0}, {-1.0, -1.0}, 
            {-1.0, 1.0}, {1.0, -1.0}
        };

        opp_dir = {
            0,
            2, 1, 
            4, 3,
            6, 5, 
            8, 7
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
                if(solid_matrix[row][col] != 0.0){
                    for (int i = 0; i < 9; i++) {
                        density_matrix[row][col] += access_matrix(particles_prob_matrixes[prob_scheduler], i, row, col);
                    }
                }
            }
        }
    }
    void update_momentum() {
        for (int row = 0; row < dim_y; row++) {
            for (int col = 0; col < dim_x; col++){
                if(solid_matrix[row][col] != 0.0){
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
    }

    void update_feq() {
        for (int row = 0; row < dim_y; row++) {
            for (int col = 0; col < dim_x; col++) {
                double rho  = density_matrix[row][col];
                double ux   = velocity_X_matrix[row][col] + Tau*acc_x;
                double uy   = velocity_Y_matrix[row][col] + Tau*acc_y;

                
                if(solid_matrix[row][col] != 0.0){
                    for (int i = 0; i < 9; i++) {
                        double ci_dot_u     = c[i][0] * ux + c[i][1] * uy;
                        double u_dot_u      = ux * ux + uy * uy;
                        double feq          = rho * weights[i] * (1.0 + 3.0 * ci_dot_u + 4.5 * ci_dot_u * ci_dot_u - 1.5 * u_dot_u);
                        set_matrix(feq, particles_eq_matrixes, i, row, col);
                    }
                }
            }
        }
    }

    void collide() {
        for (int row = 0; row < dim_y; row++) {
            for (int col = 0; col < dim_x; col++) {        

                // No collisions in solid cell
                if(solid_matrix[row][col] != 0.0){
                    for (int i = 0; i < 9; ++i) {
                        double f_i      = access_matrix(particles_prob_matrixes[prob_scheduler], i, row, col);
                        double feq_i    = access_matrix(particles_eq_matrixes, i, row, col);
                        double updated  = f_i - (f_i - feq_i) / Tau ;
                        set_matrix(updated, particles_prob_matrixes[prob_scheduler], i, row, col);
                    }
                }
            }
        }
    }


    void propagate(){

        int next_prop_scheduler = (prob_scheduler+1) % 2;
        for (int prop_row = 0; prop_row < dim_y; prop_row++) {
            for (int prop_col = 0; prop_col < dim_x; prop_col++) {
                
                
                // Evaluate each one of the 9 Directions for given propagated cell
                for (int dir = 0; dir < 9; dir++) {
                    int dx = e[dir][0];
                    int dy = e[dir][1];
                
                    // Periodic condition
                    int target_col = (prop_col + dx + dim_x) % dim_x; 
                    int target_row = (prop_row - dy + dim_y) % dim_y; // Increase row is decrease y
                    double propagated_cell_val  = access_matrix(particles_prob_matrixes[prob_scheduler], dir, prop_row, prop_col);          // Get value from propagated cell
                   

                    // Fullway bounce-back: If the target cell is solid: store particles in target node but with inverted directions
                    if(solid_matrix[target_row][target_col] == 0.0){
                        set_matrix(propagated_cell_val, particles_prob_matrixes[next_prop_scheduler], opp_dir[dir], target_row, target_col); // Set value into the new cell, but in opposed directions
                    }
                    else set_matrix(propagated_cell_val, particles_prob_matrixes[next_prop_scheduler], dir, target_row, target_col);      // Set value into the new cell
                    
                    // Halfway bounce-back: If the target cell is solid: store particles in the propagated node but with inverted directions. Do not propagate solid cells content.
                    //if(solid_matrix[prop_row][prop_col] != 0.0){
                    //    if(solid_matrix[target_row][target_col] == 0.0){
                    //        set_matrix(propagated_cell_val, particles_prob_matrixes[next_prop_scheduler], opp_dir[dir], prop_row, prop_col);
                    //    }
                    //    else{
                    //        set_matrix(propagated_cell_val, particles_prob_matrixes[next_prop_scheduler], dir, target_row, target_col);
                    //    }
                    //}
                }
            }
        }

        // After all propagations, update the current matrix
        prob_scheduler = next_prop_scheduler;

    }

    void update_velocity(){
        for (int row = 0; row < dim_y; row++) {
            for (int col = 0; col < dim_x; col++) {
                if(solid_matrix[row][col] != 0.0){
                    velocity_X_matrix[row][col] = momentum_X_matrix[row][col] /  density_matrix[row][col];
                    velocity_Y_matrix[row][col] = momentum_Y_matrix[row][col] /  density_matrix[row][col];
                }
            }
        }
    }

    void initilize_f(){
        for (int row = 0; row < dim_y; row++) {
            for (int col = 0; col < dim_x; col++) {
                for (int i = 0; i < 9; i++) {
                    if(solid_matrix[row][col] != 0.0){
                        double val = access_matrix(particles_eq_matrixes, i, row, col); // Get value from propagated cell
                        set_matrix(val, particles_prob_matrixes[prob_scheduler], i, row, col);    // Set value into the new cell
                    }
                    else{
                        double val = density_matrix[row][col] * weights[i];
                        set_matrix(val, particles_prob_matrixes[prob_scheduler], i, row, col);    // Set value into the new cell
                    }
                }
               
            }
        }
    }
    void initilize_momentum(){
        for (int col = 0; col < dim_x; col++) {
            for (int row = 0; row < dim_y; row++) {
                if(solid_matrix[row][col] != 0.0){
                    density_matrix[row][col] = 1;
                    momentum_X_matrix[row][col] = density_matrix[row][col] * velocity_X_matrix[row][col];
                    momentum_Y_matrix[row][col] = density_matrix[row][col] * velocity_Y_matrix[row][col];
                }
            }
        }
    }


    void run(int n_timesteps, const std::string& filename = "", const int N_saves_percent = 5) {

        // Restart the file if filename is provided
        std::string file_x          = filename+"_X"+".csv";
        std::string file_y          = filename+"_Y"+".csv";
        std::string file_solid      = filename+"_solid.csv";
        std::string file_infos       = filename+"_infos.csv";
        if (!filename.empty()) {
            std::ofstream clear_file_y(file_y, std::ios::trunc);
            clear_file_y.close();
            
            std::ofstream clear_file_x(file_x, std::ios::trunc);
            clear_file_x.close();

            std::ofstream clear_file_solid(file_solid, std::ios::trunc);
            save_solid_csv(file_solid);
            clear_file_solid.close();

            std::ofstream clear_file_infos(file_infos, std::ios::trunc);
            clear_file_infos << "time, mass, momentum x, momentum y, total momentum \n"; 
            clear_file_infos.close();
        }

        // 2. Set momentum to match initial velocity and density
        initilize_momentum();

        // Initialize f_eq
        update_feq();

        // Initialize f as f_eq
        initilize_f();

        // Run timesteps
        for (int t_time_step = 0; t_time_step < n_timesteps; t_time_step++) {

            // Save current timestep data if filename is provided
            int save_interval_timesteps = n_timesteps / (100/N_saves_percent);
            if (!filename.empty() && t_time_step % save_interval_timesteps  == 0) {
                save_velocity_y_csv(file_y, t_time_step*delta_t);
                save_velocity_x_csv(file_x, t_time_step*delta_t);
                save_mass_csv(file_infos, t_time_step*delta_t);
            }

            // Mesoscopic operations
            collide();
            propagate();
            // Update macroscopic consequences
            update_density();
            update_momentum();            
            update_velocity();
            // Update Equilibrium state
            update_feq();
        }
    }
    

    // AUXILIARY FUNCTIONS

    void save_velocity_y_csv(const std::string& filename, double time_instant) {
        std::ofstream file(filename, std::ios::app);  // Open in append mode

        // Write time instant as first column
        file << std::scientific << std::setprecision(16) << time_instant << ",";

        // Write flattened velocity data
        for (int row = 0; row < dim_y; row++) {
            for (int col = 0; col < dim_x; col++) {
                file << std::scientific << std::setprecision(16) << velocity_Y_matrix[row][col] * delta_h / delta_t;
                if (row != dim_y - 1 || col != dim_x - 1) {
                    file << ",";
                }
            }
        }
        file << "\n";
        file.close();
    }
    void save_velocity_x_csv(const std::string& filename, double time_instant) {
        std::ofstream file(filename, std::ios::app);  // Open in append mode

        // Write time instant as first column
        file << std::scientific << std::setprecision(16) << time_instant << ",";

        // Write flattened velocity data
        for (int row = 0; row < dim_y; row++) {
            for (int col = 0; col < dim_x; col++) {
                file << std::scientific << std::setprecision(16) << velocity_X_matrix[row][col] * delta_h / delta_t;
                if (row != dim_y - 1 || col != dim_x - 1) {
                    file << ",";
                }
            }
        }
        file << "\n";
        file.close();
    }

    void save_solid_csv(const std::string& filename, double time_instant = 0) {
        std::ofstream file(filename, std::ios::app);  // Open in append mode

        // Write time instant as first column
        file << std::scientific << std::setprecision(4) << time_instant << ",";

        // Write flattened data
        for (int row = 0; row < dim_y; row++) {
            for (int col = 0; col < dim_x; col++) {
                file << std::scientific << std::setprecision(4) << solid_matrix[row][col];
                if (row != dim_y - 1 || col != dim_x - 1) {
                    file << ",";
                }
            }
        }
        file << "\n";
        file.close();
    }

    void save_mass_csv(const std::string& filename, double time_instant) {
        std::ofstream file(filename, std::ios::app);  // Open in append mode
        // Write time instant as first column
        double mx   = compute_total_momentum_x();
        double my   = compute_total_momentum_y();
        double mass = compute_total_mass();
        file << std::scientific << std::setprecision(16) << time_instant << ",";
        // Write  data
        file << std::scientific << std::setprecision(16) << mass << "," << mx << "," << my << "," << mx+my;
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
        // Shows that momentum is consistent with mass and velocity
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
    double compute_total_momentum_y() {
        double total = 0.0;
        for (int row = 0; row < dim_y; row++) {
            for (int col = 0; col < dim_x; col++) {
                total += momentum_Y_matrix[row][col];
            }
        }
        return total;
    }

};


// Helper function to initialize velocity field
void initialize_velocity_field(LBM_BGK_D2Q9& lbm, double u_star) {
    lbm.u_star = u_star;

    for (int col = 0; col < lbm.dim_x; col++) {
        for (int row = 0; row < lbm.dim_y; row++){
            lbm.velocity_X_matrix[row][col] = 0.0;
            lbm.velocity_Y_matrix[row][col] = 0.0;
        }
    }
}

void initialize_solid_matrix(LBM_BGK_D2Q9& lbm) {
    for (int col = 0; col < lbm.dim_x; col++) {
        for (int row = 0; row < lbm.dim_y; row++) {
            // Set solid in the upper and bottom rows
            if(row ==  lbm.dim_y-1 || row == 0){
                lbm.solid_matrix[row][col] = 0.0;
                lbm.velocity_X_matrix[row][col] = 0.0;
                lbm.velocity_Y_matrix[row][col] = 0.0;
            }
            
            // Set void on middle rows
            else lbm.solid_matrix[row][col] = 1.0;
            
        }
    }
}

void propagation_test(int dir_selected){
    int dim_x = 3, dim_y=5;
    LBM_BGK_D2Q9 lbm(0.8, 1, 1, dim_x, dim_y);
    initialize_solid_matrix(lbm);
    // Initializing
    
    std::cout <<std::endl<<std::endl << "Testing Propagation of Direction: "<< dir_selected <<std::endl;
    for (int row = 0; row < dim_y; row++) {
        for (int col = 0; col < dim_x; col++) {

            if(lbm.solid_matrix[row][col] != 0.0){
                for (int i = 0; i < 9; i++) {
                    if (i == dir_selected){
                        lbm.set_matrix((row+1)*std::pow(10,col),lbm.particles_prob_matrixes[lbm.prob_scheduler], i, row, col);
                    }
                    else{
                        lbm.set_matrix(0,lbm.particles_prob_matrixes[lbm.prob_scheduler], i, row, col);
                    }
                }
            }
        }
    }
    
    for (int prop = 0; prop < 6; prop++){
        // Printing
        for (int col = 0; col < dim_x; col++) {
                std::cout<< "################";
        }
        std::cout <<std::endl;
        for (int row = 0; row < dim_y; row++) {
            for (int col = 0; col < dim_x; col++) {
                std::cout << " || ";
                std::cout << std::setw(5) << std::left << lbm.access_matrix(lbm.particles_prob_matrixes[lbm.prob_scheduler], 7, row, col);
                std::cout << std::setw(5) << std::left << lbm.access_matrix(lbm.particles_prob_matrixes[lbm.prob_scheduler], 3, row, col);
                std::cout << std::setw(5) << std::left << lbm.access_matrix(lbm.particles_prob_matrixes[lbm.prob_scheduler], 5, row, col);
                std::cout << "|| ";
            }
            std::cout <<std::endl;
            for (int col = 0; col < dim_x; col++) {
                std::cout << " || ";
                std::cout << std::setw(5) << std::left << lbm.access_matrix(lbm.particles_prob_matrixes[lbm.prob_scheduler], 2, row, col);
                std::cout << std::setw(5) << std::left << lbm.access_matrix(lbm.particles_prob_matrixes[lbm.prob_scheduler], 0, row, col);
                std::cout << std::setw(5) << std::left << lbm.access_matrix(lbm.particles_prob_matrixes[lbm.prob_scheduler], 1, row, col);
                std::cout << "|| ";
            }
            std::cout <<std::endl;
            for (int col = 0; col < dim_x; col++) {
                std::cout << " || ";
                std::cout << std::setw(5) << std::left << lbm.access_matrix(lbm.particles_prob_matrixes[lbm.prob_scheduler], 6, row, col) ;
                std::cout << std::setw(5) << std::left << lbm.access_matrix(lbm.particles_prob_matrixes[lbm.prob_scheduler], 4, row, col) ;
                std::cout << std::setw(5) << std::left << lbm.access_matrix(lbm.particles_prob_matrixes[lbm.prob_scheduler], 8, row, col) ;
                std::cout << "|| ";
            }
            std::cout <<std::endl;
            for (int col = 0; col < dim_x; col++) {
                std::cout<< "   ----------------   ";
            }
            std::cout <<std::endl;
        }
        for (int col = 0; col < dim_x; col++) {
                std::cout<< "################";
        }
        std::cout <<std::endl<<std::endl;

        lbm.propagate();
    }
}


int main() {
    // Physical parameters


    const double visc       = 1.0 * std::pow(10,-3);  // Oil visc 
    const double D          = 1.0;                    // Characteristic length [m]
    const double L          = 1.0;                    // Domain length [m]
    const double u0         = 0.05;                   // Characteristic velocity

    std::cout << "Physical parameters: " << std::endl;
    std::cout << "visc: " << visc << std::endl;
    std::cout << "D: " << D << std::endl;
    std::cout << "L: " << L << std::endl;
    std::cout << "u0: " << u0 << std::endl;

    // Simulation parameters
    const double T_real     = 1500.0;       // Simulated time
    const int dim_base      = 16;       // Base resolution (scaled in the loop)
    const int n_scales      = 3;        // Number of scales to test

    const double acc_x      = u0*visc*8/(pow(L,2)); // Field force in x direction [1 mm/s /s]

    std::vector<double> errors;
    std::vector<int> scales;

    for (int i = 0; i < n_scales; i++) {
        int scale = pow(2, i);
        scales.push_back(scale);
        std::cout << "\ni: "<< i <<", Scale: " << scale <<std::endl;

        // Increasing grid  
        int dim_y = dim_base * scale;
        int dim_x = 1;

        // Diffusion scalling
        const double spatial_step    = L/dim_y;
        const double tau_star        = 0.8;
        const double visc_star       = (1.0/3.0)*(tau_star-0.5);
        const double u_star          = (u0*D/visc)*(visc_star/dim_y);
        const double time_step       = (u_star/u0)*(L/dim_y);

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
        std::cout << "Re_star: "    << u_star * dim_y / visc_star << std::endl;
        std::cout << "u_star: "     << u0 / (spatial_step / time_step) << std::endl;
        std::cout << "visc_star: "  << visc * time_step / (spatial_step*spatial_step) << std::endl;
        std::cout << "n_timesteps: " << n_timesteps << std::endl;


        // Create and initialize simulation
        LBM_BGK_D2Q9 lbm(tau_star, spatial_step, time_step, dim_x, dim_y);
        lbm.acc_x = acc_x  * pow(time_step,2) / spatial_step ;   // Add force in lattice units

        initialize_velocity_field(lbm, u_star);
        initialize_solid_matrix(lbm);

        std::string outputfile_basename = "Scale_" + std::to_string(dim_x)+"_"+std::to_string(dim_y);

        // Run simulation
        lbm.run(n_timesteps, outputfile_basename); 
        //propagation_test(6);

    }

    return 0;
}