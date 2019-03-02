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
#ifndef NEWTON
#define NEWTON
#include "UCP_solver.cuh"

// Base class for Newton's method
class Newton : public LocalUpdate {
public:
    virtual void initialize_variables (void);
    virtual void destroy_variables (void);

    virtual void update (int stage);
    virtual void denoised (int stage);

    void set_tau (double);

    virtual void prepare_gpu_copy (void);

//protected:
    virtual void compute_derivatives (int stage);

    double _tau  {1.0};

    double* _dC_du   {nullptr};
    double* _d2C_du2 {nullptr};
};

__global__
void
parallel_update(
    UCPSolver* base,
    Newton* algo,
    const int stage
);

#endif // NEWTON