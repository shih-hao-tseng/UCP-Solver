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
#ifndef WITSENHAUSEN_H
#define WITSENHAUSEN_H

#include "UCP_solver.h"
#include "local_updates/gradient_momentum.h"
#include "local_updates/modified_newton.h"

class Witsenhausen : public UCPSolver {
public:
    virtual double get_J_value (void);

    void set_k (double k);
    void set_sigma_x (double sigma_x);
    void set_sigma_w (double sigma_w);

    double test_normalization_x (void);
    double test_normalization_w (void);

//protected:
    virtual void initialize_variables (void);
    virtual void destroy_variables (void);

    double get_w_prob_value (double coord);

    virtual double compute_C_value (int stage, double u_m, double y_m);

    double _k_2     {0.04};  // k square
    double _sigma_x {5.0};
    double _sigma_w {1.0};
    double _sigma_w_to_x_ratio {0.2};

    // computational helpers
    double* _x_prob    {nullptr};
    double* _w_prob    {nullptr};
    double* _x_prob_dx {nullptr};
    double* _w_prob_dw {nullptr};
};

class Witsenhausen_GradientMomentum : public GradientMomentum {
public:
    virtual void compute_dC_du (int stage);
};

class Witsenhausen_ModifiedNewton : public ModifiedNewton {
public:
    virtual void compute_derivatives (int stage);
};

#endif // WITSENHAUSEN_H