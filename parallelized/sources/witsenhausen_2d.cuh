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
#ifndef WITSENHAUSEN_2D_CUH
#define WITSENHAUSEN_2D_CUH

#include "UCP_solver.cuh"
#include "local_updates/gradient_momentum.cuh"
#include "local_updates/modified_newton.cuh"

class TwoDimensions {
public:
    union{
        double _number;
        struct{
            float _x;
            float _y;
        };
    };
    __host__ __device__ TwoDimensions() {}
    __host__ __device__ TwoDimensions(const TwoDimensions& b) {
        *this = b;
    }
    __host__ __device__ TwoDimensions (const double& b) {
        *this = b;
    }
    __host__ __device__ TwoDimensions& operator =(const TwoDimensions& b) {
        _number = b._number;
        return *this;
    }
    __host__ __device__ TwoDimensions& operator =(const double& b) {
        _number = b;
        return *this;
    }
    __host__ __device__ TwoDimensions& operator +=(const TwoDimensions& b) {
        _x += b._x;
        _y += b._y;
        return *this;
    }
    __host__ __device__ TwoDimensions& operator +=(const double& b) {
        TwoDimensions tmp = b;
        _x += tmp._x;
        _y += tmp._y;
        return *this;
    }
    __host__ __device__ TwoDimensions& operator -=(const TwoDimensions& b) {
        _x -= b._x;
        _y -= b._y;
        return *this;
    }
    __host__ __device__ TwoDimensions& operator -=(const double& b) {
        TwoDimensions tmp = b;
        _x -= tmp._x;
        _y -= tmp._y;
        return *this;
    }
    __host__ __device__ TwoDimensions& operator *=(const double& b) {
        _x *= b;
        _y *= b;
        return *this;
    }
    __host__ __device__ TwoDimensions operator+(const TwoDimensions &b) const {
        TwoDimensions a_plus_b = *this;
        a_plus_b += b;
        return a_plus_b;
    }
    __host__ __device__ TwoDimensions operator+(const double &b) const {
        TwoDimensions a_plus_b = *this;
        a_plus_b += b;
        return a_plus_b;
    }
    __host__ __device__ TwoDimensions operator-(const TwoDimensions &b) const {
        TwoDimensions a_minus_b = *this;
        a_minus_b -= b;
        return a_minus_b;
    }
    __host__ __device__ TwoDimensions operator-(const double &b) const {
        TwoDimensions a_minus_b = *this;
        a_minus_b -= b;
        return a_minus_b;
    }
    __host__ __device__ TwoDimensions operator*(const double &b) const {
        TwoDimensions a_times_b = *this;
        a_times_b._x *= b;
        a_times_b._y *= b;
        return a_times_b;
    }
    __host__ __device__ TwoDimensions operator/(const double &b) const {
        TwoDimensions a_divides_b = *this;
        a_divides_b._x /= b;
        a_divides_b._y /= b;
        return a_divides_b;
    }
    __host__ __device__ double norm_square() const {
        return _x*_x + _y*_y;
    }
};

class Hessian {
public:
    float _xx;
    float _xy;
    float _yy;

    __host__ __device__ TwoDimensions inverse_times (const TwoDimensions &b) {
        TwoDimensions ret;

        float determinant = _xx*_yy - _xy*_xy;
        if (determinant == 0.0){
            ret._x = 0.0;
            ret._y = 0.0;
        } else {
            ret._x = (_yy*b._x -_xy*b._y)/determinant;
            ret._y = (_xx*b._y -_xy*b._x)/determinant;
        }
        return ret;
    }
};

class Witsenhausen2D : public UCPSolver {
public:
    virtual double get_J_value (void);

    void set_k (double k);
    void set_sigma_x (double sigma_x);
    void set_sigma_w (double sigma_w);

    double test_normalization_x (void);
    double test_normalization_w (void);

//protected:
    virtual void log_u_to_file(std::ofstream& output_file, double* u);

    virtual void initialize_variables (void);
    virtual void destroy_variables (void);

    void u_initializer_2d(void);

    // for parallelization
    virtual void compute_u_denoise (int stage);
    virtual void prepare_gpu_copy (void);

    double _k_2     {0.04};  // k square
    double _sigma_x {5.0};
    double _sigma_w {1.0};
    double _sigma_w_to_x_ratio {0.2};

    // computational helpers
    double* _x_prob    {nullptr};
    double* _w_prob    {nullptr};
    double* _x_prob_dx {nullptr};
    double* _w_prob_dw {nullptr};

    double* _cpu_x_prob    {nullptr};
    double* _cpu_w_prob    {nullptr};
    double* _cpu_x_prob_dx {nullptr};
    double* _cpu_w_prob_dw {nullptr};

    // parallelize J computation
    double* _comp_x_cost {nullptr};

    // samples per direction: total samples = samples_per_direction^2
    int _samples_per_direction {0};
};

class Witsenhausen2D_Gradient : public LocalUpdate {
public:
    virtual void initialize_variables (void);
    virtual void destroy_variables (void);

    virtual void update (int stage);
    virtual void denoised (int stage) {  return;  }

    void set_tau (double tau) {  _tau = tau;  }

    virtual void compute_dC_du (int stage);
    virtual void prepare_gpu_copy (void);

    double _tau  {1.0};

    TwoDimensions* _dC_du {nullptr};
};

class Witsenhausen2D_ModifiedNewton : public LocalUpdate {
public:
    virtual void initialize_variables (void);
    virtual void destroy_variables (void);

    virtual void update (int stage);
    virtual void denoised (int stage) {  return;  }

    void set_tau (double tau) {  _tau = tau;  }

    virtual void compute_derivatives (int stage);
    virtual void prepare_gpu_copy (void);

    double _tau  {1.0};

    TwoDimensions* _dC_du   {nullptr};
    Hessian*       _d2C_du2 {nullptr};
};

__global__
void parallel_initialize_variables (
    Witsenhausen2D* base,
    const int total_samples,
    const int samples_per_direction,
    const double sample_range,
    const double sample_coord_step_size,
    const double sigma_w_to_x_ratio,
    const double w_step_size,
    double* sample_coord,

    const double sigma_x,
    double* x_prob,  double* x_prob_dx,

    const double sigma_w,
    double* w_prob,  double* w_prob_dw
);

// for parallelization
// get u value using linear interpolation
__device__
TwoDimensions get_u_value_2d (Witsenhausen2D* base, int stage, TwoDimensions coord);

// for u_denoise
__global__
void parallel_get_J_value (Witsenhausen2D* base);

__device__
double compute_C_value (Witsenhausen2D* base, int stage, TwoDimensions u_m, TwoDimensions y_m);

__global__
void parallel_compute_dC_du (
    Witsenhausen2D* base,
    Witsenhausen2D_Gradient* algo,
    const int stage
);

__device__
double get_w_prob_value (Witsenhausen2D* base, TwoDimensions coord);

#endif // WITSENHAUSEN_2D_CUH