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
#ifndef CHANNEL_CODING_H
#define CHANNEL_CODING_H

#include "UCP_solver.h"
#include "local_updates/gradient_momentum.h"
#include "local_updates/modified_newton.h"

class ChannelCoding : public UCPSolver {
public:
    virtual double get_J_value (void);

    void set_lambda (double lambda);
    void set_sigma_x (double sigma_x);

    double test_normalization_x (void);

//protected:
    virtual void initialize_variables (void);
    virtual void destroy_variables (void);

    virtual void additional_report (std::ostream& os);
    virtual void additional_log (std::ostream& os);

    virtual double compute_C_value (int stage, double u_m, double y_m);

    double _lambda  {0.04};
    double _sigma_x {1.0};

    // computational helpers
    double get_w_prob_value (double coord);
    double* _x_prob    {nullptr};
    double* _x_prob_dx {nullptr};

    double _unif_w_prob_dw {0.0};

    double _deviation_cost {0.0};
    double _power_cost     {0.0};
};

class ChannelCoding_GradientMomentum : public GradientMomentum {
public:
    virtual void compute_dC_du (int stage);
};

class ChannelCoding_ModifiedNewton : public ModifiedNewton {
public:
    virtual void compute_derivatives (int stage);
};

#endif // CHANNEL_CODING_H