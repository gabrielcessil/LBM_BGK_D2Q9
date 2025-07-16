#ifndef LBM_BGK_D2Q9_H
#define LBM_BGK_D2Q9_H

#include <vector>
#include <string>
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
    // Class members and methods (as you provided in the question)
    LBM_BGK_D2Q9(double Tau_, double spatial_step, double time_step, int dim_x_, int dim_y_);
    void run(int n_timesteps, const std::string& filename = "");
    void initialize_velocity_field(double u0);

private:
    int dim_x, dim_y;
    double Tau, delta_t, delta_h, cs;
    std::vector<double> particles_eq_matrixes;
    std::vector<std::vector<double>> particles_prob_matrixes;
    int prob_scheduler;
    std::vector<std::vector<double>> velocity_X_matrix;
    std::vector<std::vector<double>> velocity_Y_matrix;
    std::vector<std::vector<double>> density_matrix;
    std::vector<std::vector<double>> momentum_X_matrix;
    std::vector<std::vector<double>> momentum_Y_matrix;
    std::vector<std::vector<double>> e;
    std::vector<std::vector<double>> c;
    std::vector<double> weights;

    void update_density();
    void update_momentum();
    void update_feq();
    void collide();
    void propagate();
    void update_velocity();
    void save_velocity_y_csv(const std::string& filename, double time_instant);
    int get_index(int direction, int row, int col);
    double access_matrix(const std::vector<double>& matrix, int direction, int row, int col);
    void set_matrix(double value, std::vector<double>& matrix, int direction, int row, int col);
};

#endif // LBM_BGK_D2Q9_H
