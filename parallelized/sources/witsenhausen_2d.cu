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
#include "witsenhausen_2d.cuh"
#include "helper_functions.cuh"
#include <iostream>
#include <fstream>

#define INDEX(x,y) x*base->_samples_per_direction+y

double
Witsenhausen2D::get_J_value (void) {
    if (_output_coord == nullptr)  return 0.0;

    parallel_get_J_value<<<(_total_samples+255)/256, 256>>> (
        (Witsenhausen2D*)_prob_at_gpu
    );
    cudaDeviceSynchronize ();

    double cost = 0.0;
    for (int sample_x = 0; sample_x < _total_samples; ++sample_x) {
        cost += _comp_x_cost[sample_x];
    }

    // need to normalize
    return cost / 2.0;
}

__device__
TwoDimensions
get_u_value (Witsenhausen2D* base, int stage, TwoDimensions coord) {
    if ( base->_u == nullptr )  return TwoDimensions(0.0);
    // omit stage checking
    double* u = &base->_u[stage*base->_total_samples];

    if(coord._x + base->_sample_range <= 0.0) {
        coord._x = -base->_sample_range;
    } else if (coord._x >= base->_sample_range) {
        coord._x = base->_sample_range;
    }
    if(coord._y + base->_sample_range <= 0.0) {
        coord._y = -base->_sample_range;
    } else if (coord._y >= base->_sample_range) {
        coord._y = base->_sample_range;
    }

    double index_x = (coord._x + base->_sample_range) / base->_sample_coord_step_size;
    double index_y = (coord._y + base->_sample_range) / base->_sample_coord_step_size;
    int id_x_max = ceil(index_x);
    int id_x_min = floor(index_x);
    int id_y_max = ceil(index_y);
    int id_y_min = floor(index_y);

    TwoDimensions coord_x_max_y_max = base->_sample_coord[INDEX(id_x_max, id_y_max)];
    TwoDimensions coord_x_min_y_min = base->_sample_coord[INDEX(id_x_min, id_y_min)];

    TwoDimensions u_x_max_y_max = u[INDEX(id_x_max, id_y_max)];
    TwoDimensions u_x_max_y_min = u[INDEX(id_x_max, id_y_min)];
    TwoDimensions u_x_min_y_max = u[INDEX(id_x_min, id_y_max)];
    TwoDimensions u_x_min_y_min = u[INDEX(id_x_min, id_y_min)];

    // interpolation
    TwoDimensions u_y_max, u_y_min;
    if (id_x_max == id_x_min) {
        u_y_max = u_x_max_y_max;
        u_y_min = u_x_max_y_min;
    } else {
        u_y_max = (u_x_max_y_max * (coord._x - coord_x_min_y_min._x) + 
                   u_x_min_y_max * (coord_x_max_y_max._x - coord._x))
                  /(coord_x_max_y_max._x - coord_x_min_y_min._x);
        u_y_min = (u_x_max_y_min * (coord._x - coord_x_min_y_min._x) + 
                   u_x_min_y_min * (coord_x_max_y_max._x - coord._x))
                  /(coord_x_max_y_max._x - coord_x_min_y_min._x);
    }

    if (id_y_max == id_y_min) {
        return u_y_max;
    } else {
        return (u_y_max * (coord._y - coord_x_min_y_min._y) + 
                u_y_min * (coord_x_max_y_max._y - coord._y))
               /(coord_x_max_y_max._y - coord_x_min_y_min._y);
    }
}

__device__
TwoDimensions
get_du_dx_value (Witsenhausen2D* base, int stage, TwoDimensions coord) {
    TwoDimensions dx;
    dx._x = base->_sample_coord_step_size/20;
    dx._y = 0;

    return 
      ( get_u_value(base, stage, coord + dx) -
        get_u_value(base, stage, coord - dx) )
        / base->_sample_coord_step_size*10;
}

__device__
TwoDimensions
get_du_dy_value (Witsenhausen2D* base, int stage, TwoDimensions coord) {
    TwoDimensions dy;
    dy._x = 0;
    dy._y = base->_sample_coord_step_size/20;

    return 
      ( get_u_value(base, stage, coord + dy) -
        get_u_value(base, stage, coord - dy) )
        / base->_sample_coord_step_size*10;
}

__device__
TwoDimensions
get_d2u_dx2_value (Witsenhausen2D* base, int stage, TwoDimensions coord) {
    TwoDimensions dx;
    dx._x = base->_sample_coord_step_size/20;
    dx._y = 0;

    return 
      ( get_u_value(base, stage, coord + dx) +
        get_u_value(base, stage, coord - dx) -
        get_u_value(base, stage, coord) * 2
      ) / (base->_sample_coord_step_size*base->_sample_coord_step_size)*400;
}

__device__
TwoDimensions
get_d2u_dy2_value (Witsenhausen2D* base, int stage, TwoDimensions coord) {
    TwoDimensions dy;
    dy._x = 0;
    dy._y = base->_sample_coord_step_size/20;

    return 
      ( get_u_value(base, stage, coord + dy) +
        get_u_value(base, stage, coord - dy) -
        get_u_value(base, stage, coord) * 2
      ) / (base->_sample_coord_step_size*base->_sample_coord_step_size)*400;
}

__device__
TwoDimensions
get_d2u_dxdy_value (Witsenhausen2D* base, int stage, TwoDimensions coord) {
    TwoDimensions dx;
    dx._x = base->_sample_coord_step_size/20;
    dx._y = 0;

    TwoDimensions dy;
    dy._x = 0;
    dy._y = base->_sample_coord_step_size/20;

    return 
      ( get_u_value(base, stage, coord + dx + dy) +
        get_u_value(base, stage, coord - dx - dy) -
        get_u_value(base, stage, coord + dx - dy) -
        get_u_value(base, stage, coord - dx + dy)
      ) / (base->_sample_coord_step_size*base->_sample_coord_step_size)*400;
}

__global__
void
parallel_get_J_value (Witsenhausen2D* base) {
    int sample_x = blockIdx.x*blockDim.x + threadIdx.x;
    if(sample_x >= base->_total_samples)  return;

    TwoDimensions x0, u0, x1, w, u1, x2;
    base->_comp_x_cost[sample_x] = 0.0;

    x0 = base->_sample_coord[sample_x];
    u0 = base->_u[sample_x];  //get_u_value(this,0,x0);
    base->_comp_x_cost[sample_x] += base->_k_2 * u0.norm_square() * base->_x_prob_dx[sample_x];

    x1 = x0 + u0;
    for (int sample_w = 0; sample_w < base->_total_samples; ++sample_w) {
        w = base->_sample_coord[sample_w];
        w *= base->_sigma_w_to_x_ratio;

        u1 = get_u_value(base, 1, x1 + w);
        x2 = x1 - u1;
        base->_comp_x_cost[sample_x] += x2.norm_square() * base->_x_prob_dx[sample_x] * base->_w_prob_dw[sample_w];
    }
}

void
Witsenhausen2D::set_k (double k) {
    _k_2 = k*k;
}

void
Witsenhausen2D::set_sigma_x (double sigma_x) {
    _sigma_x = sigma_x;
}

void
Witsenhausen2D::set_sigma_w (double sigma_w) {
    _sigma_w = sigma_w;
}

double
Witsenhausen2D::test_normalization_x (void) {
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
Witsenhausen2D::test_normalization_w (void) {
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
Witsenhausen2D::log_u_to_file(std::ofstream& output_file, double* u) {
    for(int sample = 0; sample < _total_samples; ++sample) {
        TwoDimensions coord = _output_coord[sample];
        TwoDimensions u_value = u[sample];

        output_file << coord._x << "\t" << coord._y << "\t"
                    << u_value._x << "\t" << u_value._y << std::endl;
    }
}

void
Witsenhausen2D::u_initializer_2d (void) {
    int start = 0;
    TwoDimensions output_coord;
    TwoDimensions output_u;
    if (_u_initialized_from_input && (_input_file_names != nullptr)) {
        for(int stage = 0; stage < _total_stages; ++stage) {
            std::ifstream input_file (_input_file_names[stage].c_str());
            for(int sample = 0; sample < _total_samples; ++sample) {
                input_file >> output_coord._x >> output_coord._y >> output_u._x >> output_u._y;
                _output_coord[sample] = output_coord._number;
                _output_u[start+sample] = output_u._number;
            }
            start += _total_samples;
            input_file.close();
        }
        cudaMemcpy(_u,_output_u,_total_stages*_total_samples*sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(_sample_coord,_output_coord,_total_samples*sizeof(double),cudaMemcpyHostToDevice);
    } else {
        // sample around the center
        for(int stage = 0; stage < _total_stages; ++stage) {
            cudaMemcpy(&_u[start],_sample_coord,_total_samples*sizeof(double),cudaMemcpyDeviceToDevice);
            start += _total_samples;
        }
        cudaMemcpy(_output_u,_u,_total_stages*_total_samples*sizeof(double),cudaMemcpyDeviceToHost);
        cudaMemcpy(_output_coord,_sample_coord,_total_samples*sizeof(double),cudaMemcpyDeviceToHost);
    }
    return;
}


void
Witsenhausen2D::initialize_variables (void) {
    _samples_per_direction = sqrt(_total_samples);
    destroy_variables ();

    cudaMalloc(&_sample_coord, _total_samples*sizeof(double));

    _sample_coord_step_size = _sample_range*2.0/(_samples_per_direction-1);

    _denoise_index_radius = floor(_denoise_radius / _sample_coord_step_size);

    cudaMalloc(&_u,  _total_stages*_total_samples*sizeof(double));
    cudaMalloc(&_u_denoise, _total_stages*_total_samples*sizeof(double));

    _output_u     = new double [_total_stages*_total_samples];
    _output_coord = new double [_total_samples];

    if(_local_update != nullptr) {
        _local_update->initialize_variables ();
    }

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
        _samples_per_direction,
        _sample_range,
        _sample_coord_step_size,
        _sigma_w_to_x_ratio,
        w_step_size,
        _sample_coord,

        _sigma_x,  _x_prob,  _x_prob_dx,
        _sigma_w,  _w_prob,  _w_prob_dw
    );
    cudaDeviceSynchronize ();
    
    // The operations below force the value to sustain, otherwise, the value can go wrong (wierd)
    test_normalization_x();
    test_normalization_w();

    cudaMallocManaged(&_comp_x_cost, _total_samples*sizeof(double));

    u_initializer_2d();
}

__global__
void
parallel_initialize_variables (
    Witsenhausen2D* base,
    const int    total_samples,
    const int    samples_per_direction,
    const double sample_range,
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

    int id_x = sample / samples_per_direction;
    int id_y = sample % samples_per_direction;

    TwoDimensions coord;
    coord._x = -sample_range + sample_coord_step_size * id_x;
    coord._y = -sample_range + sample_coord_step_size * id_y;

    sample_coord[sample] = coord._number;

    double dx = sample_coord_step_size/2.0;
    double dw = w_step_size/2.0;

    x_prob[sample] =
        (normpdf(double(coord._x)-dx,sigma_x) +
         normpdf(double(coord._x)+dx,sigma_x))
      * (normpdf(double(coord._y)-dx,sigma_x) +
         normpdf(double(coord._y)+dx,sigma_x)) / 4.0;

    x_prob_dx[sample] = x_prob[sample] * sample_coord_step_size * sample_coord_step_size;

    coord *= sigma_w_to_x_ratio;

    w_prob[sample] = 
        (normpdf(double(coord._x)-dw,sigma_w) +
         normpdf(double(coord._x)+dw,sigma_w))
      * (normpdf(double(coord._y)-dw,sigma_w) +
         normpdf(double(coord._y)+dw,sigma_w)) / 4.0;

    w_prob_dw[sample] = w_prob[sample] * w_step_size * w_step_size;
}

void
Witsenhausen2D::destroy_variables (void) {
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

void
Witsenhausen2D::prepare_gpu_copy (void) {
    if (_prob_at_gpu == nullptr) {
        cudaMalloc(&_prob_at_gpu, sizeof(Witsenhausen2D));
        cudaMemcpy(_prob_at_gpu, this, sizeof(Witsenhausen2D), cudaMemcpyHostToDevice);
    }
}

__global__
void witsenhausen_2d_parallel_compute_u_denoise (
    Witsenhausen2D* base,
    const int stage
) {
    int sample_y = blockIdx.x*blockDim.x + threadIdx.x;
    if(sample_y >= base->_total_samples)  return;

    double C_value, best_value;
    TwoDimensions y_m, u_m;

    double* u         = &base->_u[stage*base->_total_samples];
    double* u_denoise = &base->_u_denoise[stage*base->_total_samples];

    y_m = base->_sample_coord[sample_y];
    // initial value
    u_denoise[sample_y] = u[sample_y];

    best_value = compute_C_value(base,stage,u[sample_y],y_m);

    int id_x = sample_y / base->_samples_per_direction;
    int id_y = sample_y % base->_samples_per_direction;

    int current_sample, current_sample_x, current_sample_y;

    for(int denoise_index_x = -base->_denoise_index_radius; 
            denoise_index_x <= base->_denoise_index_radius; ++denoise_index_x ) {
        if(denoise_index_x == 0)  continue;  // initial condition
        current_sample_x = id_x + denoise_index_x;
        if((current_sample_x < 0) ||
           (current_sample_x > base->_samples_per_direction - 1))  continue;

        for(int denoise_index_y = -base->_denoise_index_radius; 
                denoise_index_y <= base->_denoise_index_radius; ++denoise_index_y ) {
            if(denoise_index_y == 0)  continue;  // initial condition
            current_sample_y = id_y + denoise_index_y;
            if((current_sample_y < 0) ||
               (current_sample_y > base->_samples_per_direction - 1))  continue;
            
            current_sample = INDEX(current_sample_x,current_sample_y);
            C_value = compute_C_value(base,stage,u[current_sample],y_m);
            if(best_value > C_value) {
                best_value = C_value;
                u_denoise[sample_y] = u[current_sample];
            }
        }
    }
    return;
}

void
Witsenhausen2D::compute_u_denoise (int stage) {
    witsenhausen_2d_parallel_compute_u_denoise<<<(_total_samples+255)/256, 256>>> (
        (Witsenhausen2D*)_prob_at_gpu,
        stage
    );
    cudaDeviceSynchronize();
}



__device__
double
compute_C_value (Witsenhausen2D* base, int stage, TwoDimensions u_m, TwoDimensions y_m) {
    double C_value = 0.0;
    TwoDimensions x0, u0, x1, w, u1, x2;
    switch (stage) {
        case 0:  // u1
        {
            // scaling gives the same result
            //int sample_x = (y_m + _sample_range) / _sample_coord_step_size;
            x0 = y_m;
            u0 = u_m;
            C_value += base->_k_2 * u0.norm_square();// * _x_prob[sample_x];

            x1 = x0 + u0;
            for (int sample_w = 0; sample_w < base->_total_samples; ++sample_w) {
                w = base->_sample_coord[sample_w];
                w *= base->_sigma_w_to_x_ratio;
                u1 = get_u_value(base, 1, x1 + w);
                x2 = x1 - u1;
                //C_value += x2 * x2 * _x_prob[sample_x] * _w_prob_dw[sample_w];
                C_value += x2.norm_square() * base->_w_prob_dw[sample_w];
            }
            break;
        }
        case 1:  // u2
        {
            u1 = u_m;
            for (int sample_x = 0; sample_x < base->_total_samples; ++sample_x) {
                x0 = base->_sample_coord[sample_x];
                u0 = base->_u[sample_x];  //get_u_value(0,x0);
                x1 = x0 + u0;
                x2 = x1 - u1;
                C_value += x2.norm_square() * base->_x_prob_dx[sample_x] * get_w_prob_value(base, y_m - x1);
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
get_w_prob_value (Witsenhausen2D* base, TwoDimensions coord) {
    if ( base->_w_prob == nullptr )  return 0.0;

    double w_range = base->_sample_range * base->_sigma_w_to_x_ratio;

    if(coord._x + w_range <= 0.0) {
        return 0.0;
    } else if (coord._x >= w_range) {
        return 0.0;
    }
    if(coord._y + w_range <= 0.0) {
        return 0.0;
    } else if (coord._y >= w_range) {
        return 0.0;
    }

    double index_x = (coord._x/base->_sigma_w_to_x_ratio + base->_sample_range) / base->_sample_coord_step_size;
    double index_y = (coord._y/base->_sigma_w_to_x_ratio + base->_sample_range) / base->_sample_coord_step_size;
    int id_x_max = ceil(index_x);
    int id_x_min = floor(index_x);
    int id_y_max = ceil(index_y);
    int id_y_min = floor(index_y);

    TwoDimensions coord_x_max_y_max = base->_sample_coord[INDEX(id_x_max, id_y_max)];
    TwoDimensions coord_x_min_y_min = base->_sample_coord[INDEX(id_x_min, id_y_min)];

    double w_prob_x_max_y_max = base->_w_prob[INDEX(id_x_max, id_y_max)];
    double w_prob_x_max_y_min = base->_w_prob[INDEX(id_x_max, id_y_min)];
    double w_prob_x_min_y_max = base->_w_prob[INDEX(id_x_min, id_y_max)];
    double w_prob_x_min_y_min = base->_w_prob[INDEX(id_x_min, id_y_min)];

    // interpolation
    double w_prob_y_max, w_prob_y_min;
    if (id_x_max == id_x_min) {
        w_prob_y_max = w_prob_x_max_y_max;
        w_prob_y_min = w_prob_x_max_y_min;
    } else {
        w_prob_y_max = (w_prob_x_max_y_max * (coord._x - coord_x_min_y_min._x) + 
                        w_prob_x_min_y_max * (coord_x_max_y_max._x - coord._x))
                       /(coord_x_max_y_max._x - coord_x_min_y_min._x);
        w_prob_y_min = (w_prob_x_max_y_min * (coord._x - coord_x_min_y_min._x) + 
                        w_prob_x_min_y_min * (coord_x_max_y_max._x - coord._x))
                       /(coord_x_max_y_max._x - coord_x_min_y_min._x);
    }

    if (id_y_max == id_y_min) {
        return w_prob_y_max;
    } else {
        return (w_prob_y_max * (coord._y - coord_x_min_y_min._y) + 
                w_prob_y_min * (coord_x_max_y_max._y - coord._y))
               /(coord_x_max_y_max._y - coord_x_min_y_min._y);
    }
}

void
Witsenhausen2D_Gradient::initialize_variables (void) {
    destroy_variables ();

    cudaMalloc(&_dC_du, _base->_total_stages*_base->_total_samples*sizeof(TwoDimensions));

    return;
}

void
Witsenhausen2D_Gradient::destroy_variables (void) {
    cudaFree(_dC_du);
    _dC_du = nullptr;
    return;
}

__global__
void
parallel_update(
    Witsenhausen2D* base,
    Witsenhausen2D_Gradient* algo,
    const int stage
) {
    int sample = blockIdx.x*blockDim.x + threadIdx.x;
    if(sample >= base->_total_samples)  return;
    sample += stage * base->_total_samples;

    TwoDimensions u = base->_u[sample];
    
    u -= algo->_dC_du[sample] * algo->_tau;

    base->_u[sample] = u._number;
}

void
Witsenhausen2D_Gradient::update (int stage) {
    // compute dC_du
    compute_dC_du(stage);

    // update by gradient with momentum
    // again, we should update after computing dC_du
    parallel_update<<<(_base->_total_samples+255)/256, 256>>> (
        (Witsenhausen2D*) _base->_prob_at_gpu,
        (Witsenhausen2D_Gradient*) _algo_at_gpu,
        stage
    );
    cudaDeviceSynchronize ();

    return;
}

void
Witsenhausen2D_Gradient::compute_dC_du (int stage) {
    // the precision is slightly worse because the GPU supports float only rather than double
    parallel_compute_dC_du<<<(_base->_total_samples+255)/256, 256>>> (
        (Witsenhausen2D*)_base->_prob_at_gpu,
        (Witsenhausen2D_Gradient*) _algo_at_gpu,
        stage
    );
    cudaDeviceSynchronize ();
}

__global__
void
parallel_compute_dC_du (
    Witsenhausen2D* base,
    Witsenhausen2D_Gradient* algo,
    const int stage
) {
    int sample_y = blockIdx.x*blockDim.x + threadIdx.x;
    if(sample_y >= base->_total_samples)  return;

    TwoDimensions x0, u0, x1, w, u1, x2, y1;
    double prob;

    switch (stage) {
        case 0:
        {
            TwoDimensions du1_du0x, du1_du0y;
            x0 = base->_sample_coord[sample_y];
            u0 = base->_u[sample_y];
            algo->_dC_du  [sample_y] = u0 * base->_k_2 * 2 * base->_x_prob[sample_y];

            x1 = x0 + u0;
            for (int sample_w = 0; sample_w < base->_total_samples; ++sample_w) {
                w = base->_sample_coord[sample_w];
                w *= base->_sigma_w_to_x_ratio;
                y1 = x1 + w;
                u1 = get_u_value(base, 1, y1);

                du1_du0x = get_du_dx_value(base, 1, y1);
                du1_du0y = get_du_dy_value(base, 1, y1);

                x2 = x1 - u1;
                prob = 2 * base->_x_prob[sample_y] * base->_w_prob_dw[sample_w];

                algo->_dC_du[sample_y]._x += (x2._x * (1 - du1_du0x._x) - x2._y * du1_du0x._y) * prob;
                algo->_dC_du[sample_y]._y += (x2._y * (1 - du1_du0y._y) - x2._x * du1_du0y._x) * prob;
            }
            break;
        }
        case 1:
        {
            y1 = base->_sample_coord[sample_y];

            sample_y += base->_total_samples;
            algo->_dC_du[sample_y]._x = 0.0;
            algo->_dC_du[sample_y]._y = 0.0;

            u1 = base->_u[sample_y];
            for (int sample_x = 0; sample_x < base->_total_samples; ++sample_x) {
                x0 = base->_sample_coord[sample_x];
                u0 = base->_u[sample_x];
                x1 = x0 + u0;
                w  = y1 - x1;
                x2 = x1 - u1;
                algo->_dC_du[sample_y] += x2 * (-2) * base->_x_prob_dx[sample_x] * get_w_prob_value(base, w);
            }
            break;
        }
        default:
            break;
    }
}

ALGO_PARALLEL_SUITE(Witsenhausen2D_Gradient)

void
Witsenhausen2D_ModifiedNewton::initialize_variables (void) {
    destroy_variables ();

    cudaMalloc(&_dC_du,   _base->_total_stages*_base->_total_samples*sizeof(TwoDimensions));
    cudaMalloc(&_d2C_du2, _base->_total_stages*_base->_total_samples*sizeof(Hessian));

    return;
}

void
Witsenhausen2D_ModifiedNewton::destroy_variables (void) {
    cudaFree(_dC_du);
    cudaFree(_d2C_du2);
    _dC_du   = nullptr;
    _d2C_du2 = nullptr;
    return;
}

__global__
void
parallel_update(
    Witsenhausen2D* base,
    Witsenhausen2D_ModifiedNewton* algo,
    const int stage
) {
    int sample = blockIdx.x*blockDim.x + threadIdx.x;
    if(sample >= base->_total_samples)  return;
    sample += stage * base->_total_samples;

    TwoDimensions u = base->_u[sample];
    TwoDimensions du = algo->_d2C_du2[sample].inverse_times(algo->_dC_du[sample]);

    if(du._x * algo->_dC_du[sample]._x > 0.0) {
        u._x -= du._x;
    } else {
        u._x -= algo->_dC_du[sample]._x * algo->_tau;
    }
    if(du._y * algo->_dC_du[sample]._y > 0.0) {
        u._y -= du._y;
    } else {
        u._y -= algo->_dC_du[sample]._y * algo->_tau;
    }

    base->_u[sample] = u._number;
}

void
Witsenhausen2D_ModifiedNewton::update (int stage) {
    // compute dC_du and d2C_du2
    compute_derivatives (stage);

    // update by gradient with momentum
    // again, we should update after computing dC_du
    parallel_update<<<(_base->_total_samples+255)/256, 256>>> (
        (Witsenhausen2D*) _base->_prob_at_gpu,
        (Witsenhausen2D_ModifiedNewton*) _algo_at_gpu,
        stage
    );
    cudaDeviceSynchronize ();

    return;
}

__global__
void
parallel_compute_derivatives (
    Witsenhausen2D* base,
    Witsenhausen2D_ModifiedNewton* algo,
    const int stage
) {
    int sample_y = blockIdx.x*blockDim.x + threadIdx.x;
    if(sample_y >= base->_total_samples)  return;

    TwoDimensions x0, u0, x1, w, u1, x2, y1;
    TwoDimensions d2u1_du0x2, d2u1_du0xy, d2u1_du0y2;

    double prob;
    float tmpx, tmpy;

    switch (stage) {
        case 0:
        {
            TwoDimensions du1_du0x, du1_du0y;
            x0 = base->_sample_coord[sample_y];
            u0 = base->_u[sample_y];
            prob = base->_k_2 * 2 * base->_x_prob[sample_y];
            algo->_dC_du  [sample_y] = u0 * prob;
            algo->_d2C_du2[sample_y]._xx = prob;
            algo->_d2C_du2[sample_y]._xy = 0.0;
            algo->_d2C_du2[sample_y]._yy = prob;

            x1 = x0 + u0;
            for (int sample_w = 0; sample_w < base->_total_samples; ++sample_w) {
                w = base->_sample_coord[sample_w];
                w *= base->_sigma_w_to_x_ratio;
                y1 = x1 + w;
                u1 = get_u_value(base, 1, y1);

                du1_du0x = get_du_dx_value(base, 1, y1);
                du1_du0y = get_du_dy_value(base, 1, y1);

                x2 = x1 - u1;
                prob = 2 * base->_x_prob[sample_y] * base->_w_prob_dw[sample_w];

                tmpx = 1 - du1_du0x._x;
                tmpy = 1 - du1_du0y._y;

                algo->_dC_du[sample_y]._x += (x2._x * tmpx - x2._y * du1_du0x._y) * prob;
                algo->_dC_du[sample_y]._y += (x2._y * tmpy - x2._x * du1_du0y._x) * prob;

                d2u1_du0x2 = get_d2u_dx2_value(base, 1, y1);
                d2u1_du0xy = get_d2u_dxdy_value(base, 1, y1);
                d2u1_du0y2 = get_d2u_dy2_value(base, 1, y1);

                algo->_d2C_du2[sample_y]._xx +=
                    (tmpx * tmpx + x2._x * (1 - d2u1_du0x2._x) + du1_du0x._y*du1_du0x._y - d2u1_du0x2._y) * prob;
                algo->_d2C_du2[sample_y]._xy +=
                    (-du1_du0y._x * tmpx - x2._x * d2u1_du0xy._x - tmpy * du1_du0x._y - x2._y * d2u1_du0xy._y) * prob;
                algo->_d2C_du2[sample_y]._yy += 
                    (tmpy * tmpy + x2._y * (1 - d2u1_du0y2._y) + du1_du0y._x*du1_du0y._x - d2u1_du0y2._x) * prob;
            }
            break;
        }
        case 1:
        {
            y1 = base->_sample_coord[sample_y];

            sample_y += base->_total_samples;
            algo->_dC_du[sample_y]._x = 0.0;
            algo->_dC_du[sample_y]._y = 0.0;

            algo->_d2C_du2[sample_y]._xx = 0.0;
            algo->_d2C_du2[sample_y]._xy = 0.0;
            algo->_d2C_du2[sample_y]._yy = 0.0;

            u1 = base->_u[sample_y];
            for (int sample_x = 0; sample_x < base->_total_samples; ++sample_x) {
                x0 = base->_sample_coord[sample_x];
                u0 = base->_u[sample_x];
                x1 = x0 + u0;
                w  = y1 - x1;
                x2 = x1 - u1;

                prob = 2 * base->_x_prob_dx[sample_x] * get_w_prob_value(base, w);
                algo->_dC_du[sample_y] += x2 * (-1) * prob;

                algo->_d2C_du2[sample_y]._xx += prob;
                algo->_d2C_du2[sample_y]._yy += prob;
            }
            break;
        }
        default:
            break;
    }
}

void
Witsenhausen2D_ModifiedNewton::compute_derivatives (int stage) {
    // the precision is slightly worse because the GPU supports float only rather than double
    parallel_compute_derivatives<<<(_base->_total_samples+255)/256, 256>>> (
        (Witsenhausen2D*)_base->_prob_at_gpu,
        (Witsenhausen2D_ModifiedNewton*) _algo_at_gpu,
        stage
    );
    cudaDeviceSynchronize ();
}

ALGO_PARALLEL_SUITE(Witsenhausen2D_ModifiedNewton)