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
#ifndef UCP_SOLVER_CUH
#define UCP_SOLVER_CUH
#include <string>

class UCPSolver;

class LocalUpdate {
public:
    ~LocalUpdate () {
        destroy_variables ();
    }

    void set_base (UCPSolver* base);

    virtual void initialize_variables (void);
    virtual void destroy_variables (void);

    virtual void update (int stage);
    virtual void denoised (int stage);

    virtual void prepare_gpu_copy (void);

protected:
    UCPSolver* _base {nullptr};
    void*  _algo_at_gpu {nullptr};
};

#define ALGO_PARALLEL_SUITE(T) \
void \
T::prepare_gpu_copy (void) { \
    if (_algo_at_gpu == nullptr) { \
        cudaMalloc(&_algo_at_gpu, sizeof(T)); \
        cudaMemcpy(_algo_at_gpu, this, sizeof(T), cudaMemcpyHostToDevice); \
        cudaDeviceSynchronize(); \
    } \
}

class UCPSolver {
public:
    ~UCPSolver () {
        destroy_variables ();
    }

    virtual double get_J_value (void);
    double* get_u (int stage);

    void compute_result (bool output_file = false, bool silent = true);

    void set_error_bound (double);
    void set_total_iterations (int);
    void set_total_stages (int);
    void set_total_samples (int);

    void set_sample_range (double);
    void set_denoise_radius (double);

    void set_local_update (LocalUpdate* local_update);

    void set_output_file_names (std::string* output_file_names);
    void initialize_u_from_input_file (std::string* input_file_names, bool forced_initialize = false);
    void set_log_file_name (std::string log_file_name);

//protected:
    virtual void initialize_variables (void);
    virtual void destroy_variables (void);

    // output some additional information, designed for the derived classes
    virtual void additional_report (std::ostream& os);
    virtual void additional_log (std::ostream& os);

    virtual void u_initializer (void);

    // return true if u_m is locally denoised
    void u_denoise (int stage);
    virtual void compute_u_denoise (int stage);

    // for parallelization
    virtual void prepare_gpu_copy (void);

    double _error_bound      {0.0000000001};
    int    _total_iterations {20};
    int    _total_stages     {2};
    int    _total_samples    {20000};

    double _sample_range         {25.0};
    double _denoise_radius       {0.01};
    int    _denoise_index_radius {0};

    LocalUpdate* _local_update {nullptr};

    double* _sample_coord     {nullptr};  // sampled coordinates
    double  _sample_coord_step_size {1.0};

    // express 2D array in 1D
    double* _u         {nullptr};
    double* _u_denoise {nullptr};

    // for output
    double* _output_u     {nullptr};
    // _output_coord is at the host
    // _sample_coord is at the device 
    // they are essentially the same, but we separately declare them for parallel computation
    double* _output_coord {nullptr};

    std::string* _output_file_names {nullptr};
    std::string* _input_file_names  {nullptr};
    bool _u_initialized_from_input  {false};
    std::string  _log_file_name;

    void* _prob_at_gpu {nullptr};
};

__global__
void parallel_initialize_variables (
    UCPSolver* base,
    const double total_samples,
    const double sample_range,
    const double sample_coord_step_size,
    double* sample_coord
);

// get u value using linear interpolation
__device__
double get_u_value (UCPSolver* base, int stage, double coord);

// compute du/dy
__device__
double get_du_value (UCPSolver* base, int stage, double coord);

// compute d^2u/dy^2
__device__
double get_d2u_value (UCPSolver* base, int stage, double coord);

// for u_denoise
__device__
double compute_C_value (UCPSolver* base, int stage, double u_m, double y_m);

template <class T>
__global__
void parallel_compute_u_denoise (
    T* base, // using T to perform polymorphism
    const int stage
) {
    int sample_y = blockIdx.x*blockDim.x + threadIdx.x;
    if(sample_y >= base->_total_samples)  return;

    double C_value, best_value, y_m;
    int current_sample;
    double* u         = &base->_u[stage*base->_total_samples];
    double* u_denoise = &base->_u_denoise[stage*base->_total_samples];

    y_m = base->_sample_coord[sample_y];
    // initial value
    u_denoise[sample_y] = u[sample_y];
    best_value = compute_C_value(base,stage,u[sample_y],y_m);
    for(int denoise_index = -base->_denoise_index_radius; 
            denoise_index <= base->_denoise_index_radius; ++denoise_index ) {
        if(denoise_index == 0)  continue;  // initial condition
        current_sample = sample_y + denoise_index;
        if((current_sample < 0) ||
           (current_sample > base->_total_samples - 1))  continue;

        C_value = compute_C_value(base,stage,u[current_sample],y_m);
        if(best_value > C_value) {
            best_value = C_value;
            u_denoise[sample_y] = u[current_sample];
        }
    }
    return;
}

// for parallelization
#define PROB_PARALLEL_SUITE(T) \
void \
T::prepare_gpu_copy (void) { \
    if (_prob_at_gpu == nullptr) { \
        cudaMalloc(&_prob_at_gpu, sizeof(T)); \
        cudaMemcpy(_prob_at_gpu, this, sizeof(T), cudaMemcpyHostToDevice); \
    } \
} \
\
void \
T::compute_u_denoise (int stage) { \
    parallel_compute_u_denoise<T><<<(_total_samples+255)/256, 256>>> ( \
        (T*)_prob_at_gpu, \
        stage \
    ); \
    cudaDeviceSynchronize(); \
}

#endif // UCP_SOLVER_CUH