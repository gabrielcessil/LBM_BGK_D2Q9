#include <iostream>
#include "LBM_BGK_D2Q9.h"  // Include the header file

int main() {
    // Physical parameters
    const double Tau_ = 0.8;  // Example Tau value
    const double spatial_step = 1.0;  // Example spatial step (delta_h)
    const double time_step = 0.1;  // Example time step (delta_t)
    const int dim_x = 32;  // Grid dimension in x-direction
    const int dim_y = 32;  // Grid dimension in y-direction

    // Create an object of the LBM_BGK_D2Q9 class
    LBM_BGK_D2Q9 lbm(Tau_, spatial_step, time_step, dim_x, dim_y);

    // Example: Initialize velocity field
    double u0 = 0.05;  // Initial velocity (example)
    lbm.initialize_velocity_field(u0);

    // Run the simulation for 100 timesteps
    lbm.run(100, "simulation_output.csv");

    return 0;
}
