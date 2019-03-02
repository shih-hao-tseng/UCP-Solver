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
#ifndef INVENTORY_CONTROL_H
#define INVENTORY_CONTROL_H

#include "UCP_solver.h"
#include "local_updates/gradient_momentum.h"
#include "local_updates/modified_newton.h"

class InventoryControl : public UCPSolver {
public:
    virtual double get_J_value (void);

    void set_h (double h);
    void set_l (double l);
    void set_xi (double xi);

//protected:
    virtual void initialize_variables (void);

    double gamma (double a);
    double d_gamma (double a);

    // J_m: summation from stage to _total_stages - 1
    double get_J_m_value (int stage, double x_m);
    double get_dJ_m_value (int stage, double x_m, double dx_m_du0);

    virtual double compute_C_value (int stage, double u_m, double y_m);

    double _h  {2.0};
    double _l  {2.0};
    double _xi {1.0};

    // computational helpers
    double _unif_w_prob_dw {0.0};
};

class InventoryControl_GradientMomentum : public GradientMomentum {
public:
    virtual void update (int stage);
    virtual void compute_dC_du (int stage);
};

class InventoryControl_ModifiedNewton : public ModifiedNewton {
public:
    virtual void update (int stage);
    virtual void compute_derivatives (int stage);
};

#endif // INVENTORY_CONTROL_H