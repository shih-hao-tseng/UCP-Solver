/*
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 * 
 * Author: Shih-Hao Tseng <shtseng@caltech.edu>
 */
#ifndef CHANNEL_CODING_CUH
#define CHANNEL_CODING_CUH

#include "UCP_solver.cuh"
#include "local_updates/gradient_momentum.cuh"
#include "local_updates/modified_newton.cuh"

class ChannelCoding : public UCPSolver {
public:
    virtual double get_J_value (void);

    void set_lambda (double lambda);
    void set_sigma_x (double sigma_x);

    double test_normalization_x (void);

//protected:
    virtual void initialize_variables (void);
    virtual void destroy_variables (void);

    // for parallelization
    virtual void compute_u_denoise (int stage);
    virtual void prepare_gpu_copy (void);

    virtual void additional_report (std::ostream& os);
    virtual void additional_log (std::ostream& os);

    double _lambda  {0.04};
    double _sigma_x {1.0};

    double* _x_prob    {nullptr};
    double* _x_prob_dx {nullptr};

    double _unif_w_prob_dw {0.0};

    //double _deviation_cost {0.0};
    double _power_cost     {0.0};

    // parallelize J computation
    double* _comp_x_cost     {nullptr};
    double* _comp_power_cost {nullptr};
};

class ChannelCoding_GradientMomentum : public GradientMomentum {
public:
    virtual void compute_dC_du (int stage);
protected:
    virtual void prepare_gpu_copy (void);
};

class ChannelCoding_ModifiedNewton : public ModifiedNewton {
public:
    virtual void compute_derivatives (int stage);
protected:
    virtual void prepare_gpu_copy (void);
};

__global__
void parallel_get_J_value (ChannelCoding* base);

__global__
void parallel_initialize_variables (
    ChannelCoding* base,
    const double total_samples,
    const double sample_coord_step_size,
    double* sample_coord,

    const double sigma_x,
    double* x_prob,  double* x_prob_dx
);

__device__
double compute_C_value (ChannelCoding* base, int stage, double u_m, double y_m);

__device__
double get_w_prob_value (ChannelCoding* base, double coord);

__global__
void
parallel_compute_dC_du (
    ChannelCoding* base,
    ChannelCoding_GradientMomentum* algo,
    const int stage
);

__global__
void
parallel_compute_derivatives (
    ChannelCoding* base,
    ChannelCoding_ModifiedNewton* algo,
    const int stage
);

#endif // CHANNEL_CODING_CUH