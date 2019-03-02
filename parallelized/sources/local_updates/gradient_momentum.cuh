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
#ifndef GRADIENT_MOMENTUM
#define GRADIENT_MOMENTUM
#include "UCP_solver.cuh"

// Base class for gradient decent method with momentum
class GradientMomentum : public LocalUpdate {
public:
    virtual void initialize_variables (void);
    virtual void destroy_variables (void);

    virtual void update (int stage);
    virtual void denoised (int stage);

    void set_beta (double);
    void set_tau (double);

    virtual void prepare_gpu_copy (void);

//protected:
    void reset_v(int stage);

    virtual void compute_dC_du (int stage);

    double _beta {0.9};
    double _tau  {1.0};

    double* _v     {nullptr};
    double* _dC_du {nullptr};
};

__global__
void
parallel_update(
    UCPSolver* base,
    GradientMomentum* algo,
    const int stage
);

__global__
void
parallel_reset_v(
    const int stage,
    const int total_samples,
    double* v
);

#endif // GRADIENT_MOMENTUM