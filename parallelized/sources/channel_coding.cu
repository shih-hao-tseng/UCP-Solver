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
#include "channel_coding.cuh"
#include "helper_functions.cuh"
#include <iostream>

double
ChannelCoding::get_J_value (void) {
    if (_sample_coord == nullptr)  return 0.0;

    parallel_get_J_value<<<(_total_samples+255)/256, 256>>> (
        (ChannelCoding*)_prob_at_gpu
    );
    cudaDeviceSynchronize ();

    _power_cost = 0.0;
    double cost = 0.0;
    for (int sample_x = 0; sample_x < _total_samples; ++sample_x) {
        _power_cost += _comp_power_cost[sample_x];
        cost += _comp_x_cost[sample_x];
    }

    return cost;
}

__global__
void
parallel_get_J_value (ChannelCoding* base) {
    int sample_x = blockIdx.x*blockDim.x + threadIdx.x;
    if(sample_x >= base->_total_samples)  return;

    double x0, x1, u0, u1, dev;
    base->_comp_power_cost[sample_x] = 0.0;
    base->_comp_x_cost[sample_x] = 0.0;

    x0 = base->_sample_coord[sample_x];
    u0 = base->_u[sample_x];  //get_u_value(0,x0);
    base->_comp_power_cost[sample_x] = u0 * u0 * base->_x_prob_dx[sample_x];
    base->_comp_x_cost[sample_x] += base->_lambda * base->_comp_power_cost[sample_x];

    for (double w = -1.0 + base->_sample_coord_step_size/2; w <= 1.0; w += base->_sample_coord_step_size) {
        x1 = u0 + w;
        u1 = get_u_value(base, 1,x1);
        dev = (u1 - x0);
        base->_comp_x_cost[sample_x] += dev * dev * base->_x_prob_dx[sample_x] * base->_unif_w_prob_dw;
    }
}

void
ChannelCoding::set_lambda (double lambda) {
    _lambda = lambda;
}

void
ChannelCoding::set_sigma_x (double sigma_x) {
    _sigma_x = sigma_x;
}

double
ChannelCoding::test_normalization_x (void) {
    if (_x_prob_dx == nullptr)  return 0.0;

    double* x_prob_dx = new double [_total_samples];
    cudaMemcpy(x_prob_dx,_x_prob_dx,_total_samples*sizeof(double),cudaMemcpyDeviceToHost);

    double sum = 0.0;
    for (int sample = 0; sample < _total_samples; ++sample) {
        sum += x_prob_dx[sample];
    }

    delete [] x_prob_dx;
    return sum;
}

void
ChannelCoding::initialize_variables (void) {
    UCPSolver::initialize_variables ();

    _total_stages = 2;

    cudaMalloc(&_x_prob   , _total_samples*sizeof(double));
    cudaMalloc(&_x_prob_dx, _total_samples*sizeof(double));

    parallel_initialize_variables<<<(_total_samples+255)/256, 256>>> (
        this,
        _total_samples,
        _sample_coord_step_size,
        _sample_coord,
        _sigma_x,  _x_prob,  _x_prob_dx
    );
    cudaDeviceSynchronize ();

    _unif_w_prob_dw = _sample_coord_step_size / 2;

    cudaMallocManaged(&_comp_x_cost, _total_samples*sizeof(double));
    cudaMallocManaged(&_comp_power_cost, _total_samples*sizeof(double));
}

__global__
void
parallel_initialize_variables (
    ChannelCoding* base,
    const double total_samples,
    const double sample_coord_step_size,
    double* sample_coord,

    const double sigma_x,
    double* x_prob,  double* x_prob_dx
) {
    int sample = blockIdx.x*blockDim.x + threadIdx.x;
    if(sample >= total_samples)  return;

    x_prob[sample] =
        (normpdf(sample_coord[sample]-sample_coord_step_size/2.0,sigma_x) +
         normpdf(sample_coord[sample]+sample_coord_step_size/2.0,sigma_x)) / 2.0;
    x_prob_dx[sample] = x_prob[sample] * sample_coord_step_size;
}

void
ChannelCoding::destroy_variables (void) {
    if ( _output_coord == nullptr ) {
        return;
    }

    cudaFree(_x_prob);
    cudaFree(_x_prob_dx);
    _x_prob    = nullptr;
    _x_prob_dx = nullptr;

    cudaFree(_comp_x_cost);
    _comp_x_cost = nullptr;

    UCPSolver::destroy_variables ();
}

void
ChannelCoding::additional_report (std::ostream& os) {
    os << "\tgamma = " << 10 * log10(_power_cost*3.0) << " dB";
    return;
}

void
ChannelCoding::additional_log (std::ostream& os) {
    os << "\t" << 10 * log10(_power_cost*3.0);
    return;
}

PROB_PARALLEL_SUITE(ChannelCoding);

__device__
double
compute_C_value (ChannelCoding* base, int stage, double u_m, double y_m) {
    double C_value = 0.0;
    double x0, x1, u0, u1, dev;
    switch (stage) {
        case 0:
        {
            x0 = y_m;
            u0 = u_m;
            C_value += base->_lambda * u0 * u0;

            for (double w = -1.0 + base->_sample_coord_step_size/2; w <= 1.0; w += base->_sample_coord_step_size) {
                x1 = u0 + w;
                u1 = get_u_value(base, 1, x1);
                dev = (u1 - x0);
                C_value += dev * dev * base->_unif_w_prob_dw;
            }
            break;
        }
        case 1:
        {
            u1 = u_m;
            for (int sample_x = 0; sample_x < base->_total_samples; ++sample_x) {
                x0 = base->_sample_coord[sample_x];
                u0 = base->_u[sample_x];
                dev = (u1 - x0);
                // w = x1 - u0 = y1 - u0
                C_value += dev * dev * base->_x_prob_dx[sample_x] * get_w_prob_value(base, y_m - u0);
            }
            break;
        }
        default:
            break;
    }
    return C_value;
}

__device__
double
get_w_prob_value (ChannelCoding* base, double coord) {
    if(coord > 1.0) {
        return 0.0;
    }
    if(coord < -1.0) {
        return 0.0;
    }
    return base->_unif_w_prob_dw;
}

void
ChannelCoding_GradientMomentum::compute_dC_du (int stage) {
    parallel_compute_dC_du<<<(_base->_total_samples+255)/256, 256>>> (
        (ChannelCoding*)_base->_prob_at_gpu,
        (ChannelCoding_GradientMomentum*) _algo_at_gpu,
        stage
    );
    cudaDeviceSynchronize ();
}

__global__
void
parallel_compute_dC_du (
    ChannelCoding* base,
    ChannelCoding_GradientMomentum* algo,
    const int stage
) {
    int sample_y = blockIdx.x*blockDim.x + threadIdx.x;
    if(sample_y >= base->_total_samples)  return;

    double x0, x1, u0, u1, dev;
    switch (stage) {
        case 0:
        {
            double du1_du0;
            x0 = base->_sample_coord[sample_y];
            u0 = base->_u[sample_y];
            algo->_dC_du[sample_y] = 2 * base->_lambda * u0 * base->_x_prob[sample_y];

            for (double w = -1.0 + base->_sample_coord_step_size/2; w <= 1.0; w += base->_sample_coord_step_size) {
                x1 = u0 + w;
                u1 = get_u_value(base, 1,x1);
                du1_du0 = get_du_value(base, 1, x1);
                dev = (u1 - x0);
                algo->_dC_du[sample_y] += 2 * du1_du0 * dev * base->_x_prob[sample_y] * base->_unif_w_prob_dw;
            }
            break;
        }
        case 1:
        {
            x1 = base->_sample_coord[sample_y];

            sample_y += base->_total_samples;
            algo->_dC_du[sample_y] = 0.0;

            u1 = base->_u[sample_y];
            for (int sample_x = 0; sample_x < base->_total_samples; ++sample_x) {
                x0 = base->_sample_coord[sample_x];
                u0 = base->_u[sample_x];
                dev = (u1 - x0);
                // w = x1 - u0 = y1 - u0
                algo->_dC_du[sample_y] += 2 * dev * base->_x_prob_dx[sample_x] * get_w_prob_value(base, x1 - u0);
            }
            break;
        }
        default:
            break;
    }
}

ALGO_PARALLEL_SUITE(ChannelCoding_GradientMomentum)

void
ChannelCoding_ModifiedNewton::compute_derivatives (int stage) {
    parallel_compute_derivatives<<<(_base->_total_samples+255)/256, 256>>> (
        (ChannelCoding*)_base->_prob_at_gpu,
        (ChannelCoding_ModifiedNewton*) _algo_at_gpu,
        stage
    );
    cudaDeviceSynchronize ();
}

__global__
void
parallel_compute_derivatives (
    ChannelCoding* base,
    ChannelCoding_ModifiedNewton* algo,
    const int stage
) {
    int sample_y = blockIdx.x*blockDim.x + threadIdx.x;
    if(sample_y >= base->_total_samples)  return;

    double x0, x1, u0, u1, dev;
    switch (stage) {
        case 0:
        {
            double du1_du0, d2u1_du02;
            x0 = base->_sample_coord[sample_y];
            u0 = base->_u[sample_y];
            algo->_dC_du  [sample_y] = 2 * base->_lambda * u0 * base->_x_prob[sample_y];
            algo->_d2C_du2[sample_y] = 2 * base->_lambda * base->_x_prob[sample_y];

            for (double w = -1.0 + base->_sample_coord_step_size/2; w <= 1.0; w += base->_sample_coord_step_size) {
                x1 = u0 + w;
                u1 = get_u_value(base, 1, x1);
                du1_du0 = get_du_value(base, 1, x1);
                d2u1_du02 = get_d2u_value(base, 1, x1);
                dev = (u1 - x0);
                algo->_dC_du  [sample_y] += 2 * du1_du0 * dev * base->_x_prob[sample_y] * base->_unif_w_prob_dw;
                algo->_d2C_du2[sample_y] += 2 * (d2u1_du02 * dev + du1_du0 * du1_du0) * base->_x_prob[sample_y] * base->_unif_w_prob_dw;
            }
            break;
        }
        case 1:
        {
            x1 = base->_sample_coord[sample_y];

            sample_y += base->_total_samples;
            algo->_dC_du  [sample_y] = 0.0;
            algo->_d2C_du2[sample_y] = 0.0;

            u1 = base->_u[sample_y];
            for (int sample_x = 0; sample_x < base->_total_samples; ++sample_x) {
                x0 = base->_sample_coord[sample_x];
                u0 = base->_u[sample_x];
                dev = (u1 - x0);
                // w = x1 - u0 = y1 - u0
                algo->_dC_du  [sample_y] += 2 * dev * base->_x_prob_dx[sample_x] * get_w_prob_value(base, x1 - u0);
                algo->_d2C_du2[sample_y] += 2 * base->_x_prob_dx[sample_x] * get_w_prob_value(base, x1 - u0);
            }
            break;
        }
        default:
            break;
    }
}

ALGO_PARALLEL_SUITE(ChannelCoding_ModifiedNewton)