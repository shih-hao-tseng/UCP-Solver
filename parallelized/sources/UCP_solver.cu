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
#include "UCP_solver.cuh"
#include <cstring>
#include <math.h>
#include <iostream>
#include <fstream>

void
LocalUpdate::set_base (UCPSolver* base) {
    _base = base;
}

void
LocalUpdate::initialize_variables (void) {
    return;
}

void
LocalUpdate::destroy_variables (void) {
    if(_algo_at_gpu != nullptr) {
        cudaFree (_algo_at_gpu);
        _algo_at_gpu = nullptr;
    }
    return;
}

void
LocalUpdate::update (int stage) {
    return;
}

void
LocalUpdate::denoised (int stage) {
    return;
}

ALGO_PARALLEL_SUITE(LocalUpdate);

double
UCPSolver::get_J_value (void) {
    return 0.0;
}

double*
UCPSolver::get_u (int stage) {
    if ( _output_u == nullptr )  return nullptr;
    if ( ( stage >= 0 ) && ( stage < _total_stages ) ) {
        return &_output_u[stage*_total_samples];
    }
    return nullptr;
}

void
UCPSolver::compute_result (bool output_file, bool silent) {
    /*******************************
     * Initialization
     *******************************/
    initialize_variables ();

    prepare_gpu_copy ();
    if(_local_update != nullptr) {
        _local_update->prepare_gpu_copy ();
    }

    double local_update_improvement;
    double partial_exhaustion_improvement;
    double improvement = 1.0 + _error_bound;
    double current_J_value = get_J_value();
    int total_iteration_local_update = _total_iterations / 2;
    int total_iteration_partial_exhaustion = _total_iterations - total_iteration_local_update;

    int step = 0;

    /*******************************
     * Set up log file
     *******************************/
    std::ofstream flog;
    if(!silent && !_log_file_name.empty()) {
        flog.open(_log_file_name.c_str());
        if(flog) {
            flog.precision(17);
            flog << "step\tcurrent_J\t"
                 << "total_iteration_local_update\tlocal_update_improvement\t"
                 << "total_iteration_partial_exhaustion\tpartial_exhaustion_improvement" << std::endl;
        }
    }

    while(improvement > _error_bound) {
        /*******************************
         * Local Update
         *******************************/
        local_update_improvement = current_J_value;
        if(_local_update != nullptr) {
            for(int i = 0; i < total_iteration_local_update; ++i) {
                for(int stage = 0; stage < _total_stages; ++stage) {
                    _local_update->update (stage);
                }
            }
            current_J_value = get_J_value ();
        }
        local_update_improvement -= current_J_value;
        local_update_improvement = abs(local_update_improvement);
        improvement = local_update_improvement;

        /*****************************************
         * Partial Exhaustion: Local Denoising
         *****************************************/
        partial_exhaustion_improvement = current_J_value;
        for(int i = 0; i < total_iteration_partial_exhaustion; ++i) {
            for(int stage = 0; stage < _total_stages; ++stage) {
                // compute u_denoise
                u_denoise(stage);
            }
        }
        current_J_value = get_J_value ();
        partial_exhaustion_improvement -= current_J_value;
        partial_exhaustion_improvement = abs(partial_exhaustion_improvement);
        improvement += partial_exhaustion_improvement;

        if(output_file && (_output_file_names != nullptr)) {
            cudaMemcpy(_output_u,_u,_total_stages*_total_samples*sizeof(double),cudaMemcpyDeviceToHost);
            cudaMemcpy(_output_coord,_sample_coord,_total_samples*sizeof(double),cudaMemcpyDeviceToHost);
            for(int stage = 0; stage < _total_stages; ++stage) {
                std::ofstream output_file (_output_file_names[stage].c_str());
                output_file.precision(17);
                double* u = get_u(stage);
                for(int sample = 0; sample < _total_samples; ++sample) {
                    output_file << _output_coord[sample] << "\t" << u[sample] << std::endl;
                }
                output_file.close();
            }
        }

        if(!silent) {
            std::cout << step << "\tJ = " << current_J_value << "\t"
                      << "LU = " << total_iteration_local_update << ": " << local_update_improvement << "\t"
                      << "PE = " << total_iteration_partial_exhaustion << ": " << partial_exhaustion_improvement;
            additional_report (std::cout);
            std::cout << std::endl;
            if(flog) {
                flog  << step << "\t" << current_J_value << "\t"
                      << total_iteration_local_update << "\t" << local_update_improvement << "\t"
                      << total_iteration_partial_exhaustion << "\t" << partial_exhaustion_improvement;
                additional_log (flog);
                flog  << std::endl;
            }
            ++step;
        }

        /*******************************
         * Adjusting iteration weights
         *******************************/
        total_iteration_local_update = (local_update_improvement / improvement) * _total_iterations;
        if (total_iteration_local_update < 1) {
            total_iteration_local_update = 1;
        }
        if (total_iteration_local_update > _total_iterations - 1) {
            total_iteration_local_update = _total_iterations - 1;
        }
        total_iteration_partial_exhaustion = _total_iterations - total_iteration_local_update;
    }

    if(flog) {
        flog.close();
    }
}

void
UCPSolver::set_error_bound (double error_bound) {
    _error_bound = error_bound;
}

void
UCPSolver::set_total_iterations (int total_iterations) {
    _total_iterations = total_iterations;
}

void
UCPSolver::set_total_stages (int total_stages) {
    _total_stages = total_stages;
}

void
UCPSolver::set_total_samples (int total_samples) {
    if(total_samples < 1) {
        // at least two samples for the control functions
        _total_samples = 2;
    } else {
        _total_samples = total_samples;
    }
}

void
UCPSolver::set_sample_range (double sample_range){
    _sample_range = sample_range;
}

void
UCPSolver::set_denoise_radius (double denoise_radius){
    _denoise_radius = denoise_radius;
}

void
UCPSolver::set_local_update (LocalUpdate* local_update) {
    _local_update = local_update;
    _local_update->set_base (this);
}

void
UCPSolver::set_output_file_names (std::string* output_file_names) {
    _output_file_names = output_file_names;
}

void
UCPSolver::initialize_u_from_input_file (std::string* input_file_names, bool forced_initialize) {
    _input_file_names = input_file_names;
    _u_initialized_from_input = true;
    if (forced_initialize) {
        initialize_variables ();
    }
}

void
UCPSolver::set_log_file_name (std::string log_file_name) {
    _log_file_name = log_file_name;
}

void
UCPSolver::initialize_variables (void) {
    destroy_variables ();

    cudaMalloc(&_sample_coord, _total_samples*sizeof(double));
    _sample_coord_step_size = _sample_range*2.0/(_total_samples-1);

    parallel_initialize_variables<<<(_total_samples+255)/256, 256>>> (
        this,
        _total_samples,
        _sample_range,
        _sample_coord_step_size,
        _sample_coord
    );
    cudaDeviceSynchronize ();

    _denoise_index_radius = floor(_denoise_radius / _sample_coord_step_size);

    cudaMalloc(&_u,  _total_stages*_total_samples*sizeof(double));
    cudaMalloc(&_u_denoise, _total_stages*_total_samples*sizeof(double));

    _output_u     = new double [_total_stages*_total_samples];
    _output_coord = new double [_total_samples];

    u_initializer();

    if(_local_update != nullptr) {
        _local_update->initialize_variables ();
    }
}

__global__
void
parallel_initialize_variables (
    UCPSolver* base, // dummy, for manual polymorphism
    const double total_samples,
    const double sample_range,
    const double sample_coord_step_size,
    double* sample_coord
) {
    int sample = blockIdx.x*blockDim.x + threadIdx.x;
    if(sample >= total_samples)  return;

    sample_coord[sample] = -sample_range + sample*sample_coord_step_size;
}

void
UCPSolver::destroy_variables (void) {
    if ( _output_coord == nullptr ) {
        return;
    }
    if(_local_update != nullptr) {
        _local_update->destroy_variables ();
    }

    cudaFree(_u);
    cudaFree(_u_denoise);
    cudaFree(_sample_coord);
    _sample_coord = nullptr;
    _u = nullptr;
    _u_denoise = nullptr;

    delete [] _output_u;
    delete [] _output_coord;
    _output_u     = nullptr;
    _output_coord = nullptr;

    if(_prob_at_gpu != nullptr) {
        cudaFree(_prob_at_gpu);
        _prob_at_gpu = nullptr;
    }
}

void
UCPSolver::additional_report (std::ostream& os) {
    // designed for derived classes
    return;
}

void
UCPSolver::additional_log (std::ostream& os) {
    // designed for derived classes
    return;
}

void
UCPSolver::u_initializer (void) {
    int start = 0;
    if (_u_initialized_from_input && (_input_file_names != nullptr)) {
        for(int stage = 0; stage < _total_stages; ++stage) {
            std::ifstream input_file (_input_file_names[stage].c_str());
            for(int sample = 0; sample < _total_samples; ++sample) {
                input_file >> _output_coord[sample] >> _output_u[start+sample];
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
UCPSolver::u_denoise (int stage) {
    if(_output_u == nullptr)  return;

    compute_u_denoise (stage);

    cudaMemcpy(&_u[stage*_total_samples],&_u_denoise[stage*_total_samples],_total_samples*sizeof(double),cudaMemcpyDeviceToDevice);
    if(_local_update != nullptr) {
        _local_update->denoised (stage);
    }

    return;
}

// for manual polymorphism
PROB_PARALLEL_SUITE(UCPSolver);

__device__
double
get_u_value (UCPSolver* base, int stage, double coord) {
    if ( base->_u == nullptr )  return 0.0;
    // omit stage checking
    double* u = &base->_u[stage*base->_total_samples];

    if(coord + base->_sample_range <= 0.0) {
        return u[0];
    } else if (coord >= base->_sample_range) {
        return u[base->_total_samples - 1];
    } else {
        double index = (coord + base->_sample_range) / base->_sample_coord_step_size;
        int id_max = ceil(index);
        int id_min = floor(index);
        if (id_max == id_min) {
            return u[id_max];
        } else {
            double max = u[id_max];
            double min = u[id_min];
            double coord_max = base->_sample_coord[id_max];
            double coord_min = base->_sample_coord[id_min];
            return (max*(coord - coord_min) + min*(coord_max - coord))/(coord_max - coord_min);
        }
    }
}

__device__
double
get_du_value (UCPSolver* base, int stage, double coord) {
    return 
      ( get_u_value(base, stage, coord + base->_sample_coord_step_size/20) -
        get_u_value(base, stage, coord - base->_sample_coord_step_size/20) )
        / base->_sample_coord_step_size*10;
}

__device__
double
get_d2u_value (UCPSolver* base, int stage, double coord) {
    return 
      ( get_u_value(base, stage, coord + base->_sample_coord_step_size/20) + 
        get_u_value(base, stage, coord - base->_sample_coord_step_size/20) - 
        get_u_value(base, stage, coord) * 2 )
        /(base->_sample_coord_step_size*base->_sample_coord_step_size)*400;
}

__device__
double
compute_C_value (UCPSolver* base, int stage, double u_m, double y_m) {
    double C_value = 0.0;
    return C_value;
}