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
#include "UCP_solver.h"
#include <cstring>
#include <math.h>
#include <iostream>
#include <fstream>

double
UCPSolver::get_J_value (void) {
    return 0.0;
}

double*
UCPSolver::get_u (int stage) {
    if ( _u == nullptr )  return nullptr;
    if ( ( stage >= 0 ) && ( stage < _total_stages ) ) {
        return _u[stage];
    }
    return nullptr;
}

void
UCPSolver::compute_result (bool output_file, bool silent) {
    /*******************************
     * Initialization
     *******************************/
    initialize_variables ();

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
        /******************************************
         * Local Update
         ******************************************/
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

        /******************************************
         * Partial Exhaustion: Local Denoising
         ******************************************/
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
            for(int stage = 0; stage < _total_stages; ++stage) {
                std::ofstream output_file (_output_file_names[stage].c_str());
                output_file.precision(17);
                for(int sample = 0; sample < _total_samples; ++sample) {
                    output_file << _sample_coord[sample] << "\t" << _u[stage][sample] << std::endl;
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

    _u = new double*[_total_stages];
    _u_denoise = new double*[_total_stages];

    _sample_coord = new double [_total_samples];
    _sample_coord_step_size = _sample_range*2.0/(_total_samples-1);
    _sample_coord[0] = -_sample_range;
    for(int sample = 1; sample < _total_samples; ++sample) {
        _sample_coord[sample] = _sample_coord[sample-1] + _sample_coord_step_size;
    }

    _denoise_index_radius = floor(_denoise_radius / _sample_coord_step_size);

    for(int stage = 0; stage < _total_stages; ++stage) {
        _u[stage] = new double [_total_samples];
        _u_denoise[stage] = new double [_total_samples];
    }

    u_initializer();

    if(_local_update != nullptr) {
        _local_update->initialize_variables ();
    }
}

void
UCPSolver::destroy_variables (void) {
    if ( _sample_coord == nullptr ) {
        return;
    }
    if(_local_update != nullptr) {
        _local_update->destroy_variables ();
    }

    for(int stage = 0; stage < _total_stages; ++stage) {
        delete [] _u[stage];
        delete [] _u_denoise[stage];
    }
    delete [] _sample_coord;
    delete [] _u;
    delete [] _u_denoise;
    _sample_coord = nullptr;
    _u = nullptr;
    _u_denoise = nullptr;
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
    if (_u_initialized_from_input && (_input_file_names != nullptr)) {
        for(int stage = 0; stage < _total_stages; ++stage) {
            std::ifstream input_file (_input_file_names[stage].c_str());
            for(int sample = 0; sample < _total_samples; ++sample) {
                input_file >> _sample_coord[sample] >> _u[stage][sample];
            }
            input_file.close();
        }
    } else {
        // sample around the center
        for(int stage = 0; stage < _total_stages; ++stage) {
            memcpy(&_u[stage][0],&_sample_coord[0],_total_samples*sizeof(double));
        }
    }
    return;
}

double
UCPSolver::get_u_value (int stage, double coord) {
    if ( _u == nullptr )  return 0.0;
    if ( ( stage < 0 ) || ( stage >= _total_stages ) ) {
        return 0.0;
    }

    if(coord + _sample_range <= 0.0) {
        return _u[stage][0];
    } else if (coord >= _sample_range) {
        return _u[stage][_total_samples - 1];
    } else {
        double index = (coord + _sample_range) / _sample_coord_step_size;
        int id_max = ceil(index);
        int id_min = floor(index);
        if (id_max == id_min) {
            return _u[stage][id_max];
        } else {
            double max = _u[stage][id_max];
            double min = _u[stage][id_min];
            double coord_max = _sample_coord[id_max];
            double coord_min = _sample_coord[id_min];
            return (max*(coord - coord_min) + min*(coord_max - coord))/(coord_max - coord_min);
        }
    }
}

double
UCPSolver::get_du_value (int stage, double coord) {
    return (get_u_value(stage,coord+_sample_coord_step_size/20)
           -get_u_value(stage,coord-_sample_coord_step_size/20))
           /_sample_coord_step_size*10;
}

double
UCPSolver::get_d2u_value (int stage, double coord) {
    return (get_u_value(stage,coord+_sample_coord_step_size/20)
           +get_u_value(stage,coord-_sample_coord_step_size/20)
           -get_u_value(stage,coord)*2)
           /(_sample_coord_step_size*_sample_coord_step_size)*400;
}

bool
UCPSolver::u_denoise (int stage) {
    if(_u == nullptr)  return false;

    bool stage_denoised = false;
    // initial conditions
    memcpy(&_u_denoise[stage][0],&_u[stage][0],_total_samples*sizeof(double));

    double C_value, best_value, y_m;
    int current_sample;
    for(int sample_y = 0; sample_y < _total_samples; ++sample_y) {
        y_m = _sample_coord[sample_y];
        // initial value
        best_value = compute_C_value(stage,_u[stage][sample_y],y_m);
        for(int denoise_index = -_denoise_index_radius; 
                denoise_index <= _denoise_index_radius; ++denoise_index ) {
            if(denoise_index == 0)  continue;  // initial condition
            current_sample = sample_y + denoise_index;
            if((current_sample < 0) ||
               (current_sample > _total_samples - 1))  continue;

            C_value = compute_C_value(stage,_u[stage][current_sample],y_m);
            if(best_value > C_value) {
                best_value = C_value;
                _u_denoise[stage][sample_y] = _u[stage][current_sample];
                // force the algorithm to search again
                stage_denoised = true;
            }
/*
            // check two more points: the average
            if((denoise_index == 1) || (denoise_index == -1)) {
                double u_value = (_u[stage][sample_y] + _u[stage][current_sample])/2;
                C_value = compute_C_value(stage,u_value,y_m);
                if(best_value > C_value) {
                    best_value = C_value;
                    _u_denoise[stage][sample_y] = u_value;
                    // force the algorithm to search again
                    stage_denoised = true;
                }
            }
*/
        }
    }

    // update
    // we should update after checking all the values
    if(stage_denoised) {
        memcpy(&_u[stage][0],&_u_denoise[stage][0],_total_samples*sizeof(double));
        _local_update->denoised (stage);
    }

    return stage_denoised;
}

double
UCPSolver::compute_C_value (int stage, double u_m, double y_m) {
    double C_value = 0.0;
    return C_value;
}

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