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
#include "inventory_control.cuh"
#include <math.h>

double
InventoryControl::get_J_value (void) {
    if (_sample_coord == nullptr)  return 0.0;

    parallel_get_J_value<<<(_total_samples+255)/256, 256>>> (
        (InventoryControl*)_prob_at_gpu
    );
    cudaDeviceSynchronize ();

    double retval = 0.0;
    for (int sample_x = 0; sample_x < _total_samples; ++sample_x) {
        retval += _comp_x_cost[sample_x];
    }
    retval *= _unif_w_prob_dw;
    return retval;
}

__global__
void
parallel_get_J_value (InventoryControl* base) {
    int sample_x = blockIdx.x*blockDim.x + threadIdx.x;
    if(sample_x >= base->_total_samples)  return;
    double x_0 = -1.0 + base->_sample_coord_step_size/2 + sample_x * base->_sample_coord_step_size;

    base->_comp_x_cost[sample_x] = get_J_m_value (base, 0, x_0);
}

void
InventoryControl::set_h (double h) {
    _h = h;
}

void
InventoryControl::set_l (double l) {
    _l = l;
}

void
InventoryControl::set_xi (double xi) {
    _xi = xi;
}

void
InventoryControl::initialize_variables (void) {
    UCPSolver::initialize_variables ();

    _unif_w_prob_dw = _sample_coord_step_size / 2.0;

    cudaMallocManaged(&_comp_x_cost, _total_samples*sizeof(double));
}

void
InventoryControl::destroy_variables (void) {
    if ( _output_coord == nullptr ) {
        return;
    }

    cudaFree(_comp_x_cost);
    _comp_x_cost = nullptr;

    UCPSolver::destroy_variables ();
}

__device__
double
gamma (InventoryControl* base, double a) {
    if (a > 0.0) {
        return a * base->_h;
    }
    if (a < 0.0) {
        return - a * base->_l;
    }
    return 0.0;
}

__device__
double
d_gamma (InventoryControl* base, double a) {
    if (a > 0.0) {
        return base->_h;
    }
    if (a < 0.0) {
        return -base->_l;
    }
    return 0.0;
}

__device__
double
get_J_m_value (
    InventoryControl* base,
    int stage,
    double x_m
) {
    //   E[ \sum\limits_{m=0}^M \xi u_m + \gamma(x_m + u_m - w_m) ]
    // = E[ \sum\limits_{m=0}^M \xi u_m + \gamma(x_{m+1}) ]
    if( (stage < 0) || (stage > base->_total_stages - 1) ) {
        return 0.0;
    }
    // compute \xi u_m(x_m) + E[ \gamma(x_{m+1}) ]
    double retval = 0.0;
    double u_m = get_u_value (base, stage, x_m);
    //retval += base->_xi * u_m;
    double x_m_and_u_m = u_m + x_m;
    double x_m1;
    for (double w_m = -1.0 + base->_sample_coord_step_size/2; w_m <= 1.0; w_m += base->_sample_coord_step_size) {
        x_m1 = x_m_and_u_m - w_m;
        retval += (gamma(base, x_m1) + get_J_m_value(base,stage + 1, x_m1));
    }
    retval *= base->_unif_w_prob_dw;
    retval += base->_xi * u_m;
    return retval;
}

__device__
double
get_dJ_m_value (
    InventoryControl* base,
    int stage, double x_m, double dx_m_du0
) {
    if( (stage < 0) || (stage > base->_total_stages - 1) ) {
        return 0.0;
    }
    double du1 = get_du_value (base, stage, x_m);
    double retval = 0.0;
    //retval += base->_xi * du1 * dx_m_du0;
    double x_m_and_u_m = x_m + get_u_value(base, stage, x_m);
    double x_m1;
    double dx_m1_du0 = (1.0 + du1) * dx_m_du0;
    for (double w_m = -1.0 + base->_sample_coord_step_size/2; w_m <= 1.0; w_m += base->_sample_coord_step_size) {
        x_m1 = x_m_and_u_m - w_m;
        retval += (d_gamma(base, x_m1) * dx_m1_du0 + get_dJ_m_value(base, stage + 1, x_m1, dx_m1_du0));
    }
    retval *= base->_unif_w_prob_dw;
    retval += base->_xi * du1 * dx_m_du0;
    return retval;
}

__device__
double
compute_C_value (InventoryControl* base, int stage, double u_m, double y_m) {
    double C_value = 0.0; 
    //C_value += base->_xi * u_m;
    // scaling is not important under the same y_m
    // and hence we can consider only the effect after the current stage 
    double x_m_and_u_m = y_m + u_m;
    double x_m1;
    for (double w_m = -1.0 + base->_sample_coord_step_size/2; w_m <= 1.0; w_m += base->_sample_coord_step_size) {
        x_m1 = x_m_and_u_m - w_m;
        C_value += (gamma(base,x_m1) + get_J_m_value(base,stage + 1, x_m1));
    }
    C_value *= base->_unif_w_prob_dw;
    C_value += base->_xi * u_m;
    return C_value;
}

PROB_PARALLEL_SUITE(InventoryControl);

void
InventoryControl_GradientMomentum::update (int stage) {
    // compute dC_du
    compute_dC_du (stage);

    // update by gradient with momentum
    parallel_update<<<(_base->_total_samples+255)/256, 256>>> (
        (InventoryControl*) _base->_prob_at_gpu,
        (InventoryControl_GradientMomentum*) _algo_at_gpu,
        stage
    );
    cudaDeviceSynchronize ();
}

__global__
void
parallel_update(
    InventoryControl* base,
    InventoryControl_GradientMomentum* algo,
    const int stage
) {
    int sample = blockIdx.x*blockDim.x + threadIdx.x;
    if(sample >= base->_total_samples)  return;
    sample += stage * base->_total_samples;

    algo->_v[sample] = algo->_beta*algo->_v[sample] - algo->_tau*algo->_dC_du[sample];
    base->_u[sample] = base->_u[sample] + algo->_v[sample];
    if(base->_u[sample] < 0.0) {
        // non negative constraints
        base->_u[sample] = 0.0;
    }
}

void
InventoryControl_GradientMomentum::compute_dC_du (int stage) {
    parallel_compute_dC_du<<<(_base->_total_samples+255)/256, 256>>> (
        (InventoryControl*)_base->_prob_at_gpu,
        (InventoryControl_GradientMomentum*) _algo_at_gpu,
        stage
    );
    cudaDeviceSynchronize ();
}

__global__
void
parallel_compute_dC_du (
    InventoryControl* base,
    InventoryControl_GradientMomentum* algo,
    const int stage
) {
    int sample_y = blockIdx.x*blockDim.x + threadIdx.x;
    if(sample_y >= base->_total_samples)  return;

    if((stage < 0) || (stage > base->_total_stages - 1)) {
        return;
    }
    sample_y += stage * base->_total_samples;

    double x_m_and_u_m, x_m1;
    x_m_and_u_m = base->_sample_coord[sample_y] + base->_u[sample_y];
    algo->_dC_du[sample_y] = 0.0;
    //algo->_dC_du[sample_y] += base->_xi;
    // stage
    for (double w_m = -1.0 + base->_sample_coord_step_size/2; w_m <= 1.0; w_m += base->_sample_coord_step_size) {
        x_m1 = x_m_and_u_m - w_m;
        algo->_dC_du[sample_y] += (d_gamma(base,x_m1) + get_dJ_m_value(base, stage + 1, x_m1, 1));
    }
    algo->_dC_du[sample_y] *= base->_unif_w_prob_dw;
    algo->_dC_du[sample_y] += base->_xi;
}

ALGO_PARALLEL_SUITE(InventoryControl_GradientMomentum)