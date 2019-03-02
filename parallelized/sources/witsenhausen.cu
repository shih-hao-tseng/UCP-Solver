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
#include "witsenhausen.cuh"
#include "helper_functions.cuh"

double
Witsenhausen::get_J_value (void) {
    if (_output_coord == nullptr)  return 0.0;

    parallel_get_J_value<<<(_total_samples+255)/256, 256>>> (
        (Witsenhausen*)_prob_at_gpu
    );
    cudaDeviceSynchronize ();

    double cost = 0.0;
    for (int sample_x = 0; sample_x < _total_samples; ++sample_x) {
        cost += _comp_x_cost[sample_x];
    }

    return cost;
}

__global__
void
parallel_get_J_value (Witsenhausen* base) {
    int sample_x = blockIdx.x*blockDim.x + threadIdx.x;
    if(sample_x >= base->_total_samples)  return;

    double x0, u0, x1, w, u1, x2;
    base->_comp_x_cost[sample_x] = 0.0;

    x0 = base->_sample_coord[sample_x];
    u0 = base->_u[sample_x];  //get_u_value(this,0,x0);
    base->_comp_x_cost[sample_x] += base->_k_2 * u0 * u0 * base->_x_prob_dx[sample_x];

    x1 = x0 + u0;
    for (int sample_w = 0; sample_w < base->_total_samples; ++sample_w) {
        w = base->_sample_coord[sample_w] * base->_sigma_w_to_x_ratio;
        u1 = get_u_value(base, 1, x1 + w);
        x2 = x1 - u1;
        base->_comp_x_cost[sample_x] += x2 * x2 * base->_x_prob_dx[sample_x] * base->_w_prob_dw[sample_w];
    }
}

void
Witsenhausen::set_k (double k) {
    _k_2 = k*k;
}

void
Witsenhausen::set_sigma_x (double sigma_x) {
    _sigma_x = sigma_x;
}

void
Witsenhausen::set_sigma_w (double sigma_w) {
    _sigma_w = sigma_w;
}

double
Witsenhausen::test_normalization_x (void) {
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

double
Witsenhausen::test_normalization_w (void) {
    if (_w_prob_dw == nullptr)  return 0.0;

    double* w_prob_dw = new double [_total_samples];
    cudaMemcpy(w_prob_dw,_w_prob_dw,_total_samples*sizeof(double),cudaMemcpyDeviceToHost);

    double sum = 0.0;
    for (int sample = 0; sample < _total_samples; ++sample) {
        sum += w_prob_dw[sample];
    }

    delete [] w_prob_dw;
    return sum;
}

void
Witsenhausen::initialize_variables (void) {
    UCPSolver::initialize_variables ();

    _total_stages = 2;

    cudaMalloc(&_x_prob   , _total_samples*sizeof(double));
    cudaMalloc(&_x_prob_dx, _total_samples*sizeof(double));
    cudaMalloc(&_w_prob   , _total_samples*sizeof(double));
    cudaMalloc(&_w_prob_dw, _total_samples*sizeof(double));

    _sigma_w_to_x_ratio = _sigma_w/_sigma_x;
    double w_step_size = _sample_coord_step_size * _sigma_w_to_x_ratio;

    parallel_initialize_variables<<<(_total_samples+255)/256, 256>>> (
        this,
        _total_samples,
        _sample_coord_step_size,
        _sigma_w_to_x_ratio,
        w_step_size,
        _sample_coord,

        _sigma_x,  _x_prob,  _x_prob_dx,
        _sigma_w,  _w_prob,  _w_prob_dw
    );
    cudaDeviceSynchronize ();

    cudaMallocManaged(&_comp_x_cost, _total_samples*sizeof(double));
}

__global__
void
parallel_initialize_variables (
    Witsenhausen* base,
    const double total_samples,
    const double sample_coord_step_size,
    const double sigma_w_to_x_ratio,
    const double w_step_size,
    double* sample_coord,

    const double sigma_x,
    double* x_prob,  double* x_prob_dx,

    const double sigma_w,
    double* w_prob,  double* w_prob_dw
) {
    int sample = blockIdx.x*blockDim.x + threadIdx.x;
    if(sample >= total_samples)  return;

    x_prob[sample] =
        (normpdf(sample_coord[sample]-sample_coord_step_size/2.0,sigma_x) +
         normpdf(sample_coord[sample]+sample_coord_step_size/2.0,sigma_x)) / 2.0;
    x_prob_dx[sample] = x_prob[sample] * sample_coord_step_size;

    w_prob[sample] = 
        (normpdf(sample_coord[sample]*sigma_w_to_x_ratio-w_step_size/2.0,sigma_w) +
         normpdf(sample_coord[sample]*sigma_w_to_x_ratio+w_step_size/2.0,sigma_w)) / 2.0;
    w_prob_dw[sample] = w_prob[sample] * w_step_size;
}

void
Witsenhausen::destroy_variables (void) {
    if ( _output_coord == nullptr ) {
        return;
    }

    cudaFree(_x_prob);
    cudaFree(_x_prob_dx);
    cudaFree(_w_prob);
    cudaFree(_w_prob_dw);
    _x_prob    = nullptr;
    _x_prob_dx = nullptr;
    _w_prob    = nullptr;
    _w_prob_dw = nullptr;

    cudaFree(_comp_x_cost);
    _comp_x_cost = nullptr;

    UCPSolver::destroy_variables ();
}

PROB_PARALLEL_SUITE(Witsenhausen);

__device__
double
compute_C_value (Witsenhausen* base, int stage, double u_m, double y_m) {
    double C_value = 0.0;
    double x0, u0, x1, w, u1, x2;
    switch (stage) {
        case 0:
        {
            // scaling gives the same result
            //int sample_x = (y_m + _sample_range) / _sample_coord_step_size;
            x0 = y_m;
            u0 = u_m;
            C_value += base->_k_2 * u0 * u0;// * _x_prob[sample_x];

            x1 = x0 + u0;
            for (int sample_w = 0; sample_w < base->_total_samples; ++sample_w) {
                w = base->_sample_coord[sample_w] * base->_sigma_w_to_x_ratio;
                u1 = get_u_value(base, 1, x1 + w);
                x2 = x1 - u1;
                //C_value += x2 * x2 * _x_prob[sample_x] * _w_prob_dw[sample_w];
                C_value += x2 * x2 * base->_w_prob_dw[sample_w];
            }
            break;
        }
        case 1:
        {
            u1 = u_m;
            for (int sample_x = 0; sample_x < base->_total_samples; ++sample_x) {
                x0 = base->_sample_coord[sample_x];
                u0 = base->_u[sample_x];  //get_u_value(0,x0);
                x1 = x0 + u0;
                x2 = x1 - u1;
                C_value += x2 * x2 * base->_x_prob_dx[sample_x] * get_w_prob_value(base, y_m - x1);
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
get_w_prob_value (Witsenhausen* base, double coord) {
    if ( base->_w_prob == nullptr )  return 0.0;

    if(coord + base->_sample_range * base->_sigma_w_to_x_ratio < 0.0) {
        return 0.0;
    } else if (coord > base->_sample_range * base->_sigma_w_to_x_ratio) {
        return 0.0;
    } else {
        double index = (coord/base->_sigma_w_to_x_ratio + base->_sample_range) / base->_sample_coord_step_size;
        int id_max = ceil(index);
        int id_min = floor(index);
        if (id_max == id_min) {
            return base->_w_prob[id_max];
        } else {
            double max = base->_w_prob[id_max];
            double min = base->_w_prob[id_min];
            double coord_max = base->_sample_coord[id_max] * base->_sigma_w_to_x_ratio;
            double coord_min = base->_sample_coord[id_min] * base->_sigma_w_to_x_ratio;
            return (max*(coord - coord_min) + min*(coord_max - coord))/(coord_max - coord_min);
        }
    }
}

void
Witsenhausen_GradientMomentum::compute_dC_du (int stage) {
    // the precision is slightly worse because the GPU supports float only rather than double
    parallel_compute_dC_du<<<(_base->_total_samples+255)/256, 256>>> (
        (Witsenhausen*)_base->_prob_at_gpu,
        (Witsenhausen_GradientMomentum*) _algo_at_gpu,
        stage
    );
    cudaDeviceSynchronize ();
}

__global__
void
parallel_compute_dC_du (
    Witsenhausen* base,
    Witsenhausen_GradientMomentum* algo,
    const int stage
) {
    int sample_y = blockIdx.x*blockDim.x + threadIdx.x;
    if(sample_y >= base->_total_samples)  return;

    double x0, u0, x1, w, u1, x2, y1;
    switch (stage) {
        case 0:
        {
            double du1_du0;
            x0 = base->_sample_coord[sample_y];
            u0 = base->_u[sample_y];
            algo->_dC_du  [sample_y] = base->_k_2 * 2 * u0 * base->_x_prob[sample_y];

            x1 = x0 + u0;
            for (int sample_w = 0; sample_w < base->_total_samples; ++sample_w) {
                w = base->_sample_coord[sample_w] * base->_sigma_w_to_x_ratio;
                y1 = x1 + w;
                u1 = get_u_value(base, 1, y1);
                du1_du0 = get_du_value(base, 1, y1);
                x2 = x1 - u1;
                algo->_dC_du  [sample_y] += 2 * (1 - du1_du0) * x2 * base->_x_prob[sample_y] * base->_w_prob_dw[sample_w];
            }
            break;
        }
        case 1:
        {
            y1 = base->_sample_coord[sample_y];

            sample_y += base->_total_samples;
            algo->_dC_du  [sample_y] = 0.0;

            u1 = base->_u[sample_y];
            for (int sample_x = 0; sample_x < base->_total_samples; ++sample_x) {
                x0 = base->_sample_coord[sample_x];
                u0 = base->_u[sample_x];
                x1 = x0 + u0;
                w  = y1 - x1;
                x2 = x1 - u1;
                algo->_dC_du  [sample_y] += - 2 * x2 * base->_x_prob_dx[sample_x] * get_w_prob_value(base, w);
            }
            break;
        }
        default:
            break;
    }
}

ALGO_PARALLEL_SUITE(Witsenhausen_GradientMomentum)

void
Witsenhausen_ModifiedNewton::compute_derivatives (int stage) {
    // the precision is slightly worse because the GPU supports float only rather than double
    parallel_compute_derivatives<<<(_base->_total_samples+255)/256, 256>>> (
        (Witsenhausen*)_base->_prob_at_gpu,
        (Witsenhausen_ModifiedNewton*) _algo_at_gpu,
        stage
    );
    cudaDeviceSynchronize ();
}

__global__
void
parallel_compute_derivatives (
    Witsenhausen* base,
    Witsenhausen_ModifiedNewton* algo,
    const int stage
) {
    int sample_y = blockIdx.x*blockDim.x + threadIdx.x;
    if(sample_y >= base->_total_samples)  return;

    // Also compute d2C_du2
    double x0, u0, x1, w, u1, x2, y1;
    switch (stage) {
        case 0:
        {
            double du1_du0, d2u1_du02;
            x0 = base->_sample_coord[sample_y];
            u0 = base->_u[sample_y];
            algo->_dC_du  [sample_y] = base->_k_2 * 2 * u0 * base->_x_prob[sample_y];
            algo->_d2C_du2[sample_y] = base->_k_2 * 2 * base->_x_prob[sample_y];

            x1 = x0 + u0;
            for (int sample_w = 0; sample_w < base->_total_samples; ++sample_w) {
                w = base->_sample_coord[sample_w] * base->_sigma_w_to_x_ratio;
                y1 = x1 + w;
                u1 = get_u_value(base, 1, y1);
                du1_du0 = get_du_value(base, 1, y1);
                d2u1_du02 = get_d2u_value(base, 1, y1);
                x2 = x1 - u1;
                algo->_dC_du  [sample_y] += 2 * (1 - du1_du0) * x2 * base->_x_prob[sample_y] * base->_w_prob_dw[sample_w];
                algo->_d2C_du2[sample_y] += 2 * ( -d2u1_du02 * x2 + (1 - du1_du0) * (1 - du1_du0)) * base->_x_prob[sample_y] * base->_w_prob_dw[sample_w];
            }
            break;
        }
        case 1:
        {
            y1 = base->_sample_coord[sample_y];

            sample_y += base->_total_samples;
            algo->_dC_du  [sample_y] = 0.0;
            algo->_d2C_du2[sample_y] = 0.0;

            u1 = base->_u[sample_y];
            for (int sample_x = 0; sample_x < base->_total_samples; ++sample_x) {
                x0 = base->_sample_coord[sample_x];
                u0 = base->_u[sample_x];
                x1 = x0 + u0;
                w  = y1 - x1;
                x2 = x1 - u1;
                algo->_dC_du  [sample_y] += - 2 * x2 * base->_x_prob_dx[sample_x] * get_w_prob_value(base, w);
                algo->_d2C_du2[sample_y] += 2 * base->_x_prob_dx[sample_x] * get_w_prob_value(base, w);
            }
            break;
        }
        default:
            break;
    }
}

ALGO_PARALLEL_SUITE(Witsenhausen_ModifiedNewton)