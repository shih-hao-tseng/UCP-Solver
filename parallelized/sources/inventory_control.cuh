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
#ifndef INVENTORY_CONTROL_CUH
#define INVENTORY_CONTROL_CUH

#include "UCP_solver.cuh"
#include "local_updates/gradient_momentum.cuh"

class InventoryControl : public UCPSolver {
public:
    virtual double get_J_value (void);

    void set_h (double h);
    void set_l (double l);
    void set_xi (double xi);

//protected:
    virtual void initialize_variables (void);
    virtual void destroy_variables (void);

    // for parallelization
    virtual void compute_u_denoise (int stage);
    virtual void prepare_gpu_copy (void);

    double _h  {2.0};
    double _l  {2.0};
    double _xi {1.0};

    // computational helpers
    double _unif_w_prob_dw {0.0};

    // parallelize J computation
    double* _comp_x_cost {nullptr};
};

class InventoryControl_GradientMomentum : public GradientMomentum {
public:
    virtual void update (int stage);
    virtual void compute_dC_du (int stage);
protected:
    virtual void prepare_gpu_copy (void);
};

__global__
void parallel_get_J_value (InventoryControl* base);

// J_m: summation from stage to _total_stages - 1
__device__
double get_J_m_value (InventoryControl* base, int stage, double x_m);

__device__
double get_dJ_m_value (InventoryControl* base, int stage, double x_m, double dx_m_du0);

__device__
double gamma (InventoryControl* base, double a);

__device__
double d_gamma (InventoryControl* base, double a);

__device__
double compute_C_value (InventoryControl* base, int stage, double u_m, double y_m);

__global__
void
parallel_update(
    InventoryControl* base,
    InventoryControl_GradientMomentum* algo,
    const int stage
);

__global__
void
parallel_compute_dC_du (
    InventoryControl* base,
    InventoryControl_GradientMomentum* algo,
    const int stage
);

#endif // INVENTORY_CONTROL_CUH