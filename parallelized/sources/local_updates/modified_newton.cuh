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
#ifndef MODIFIED_NEWTON
#define MODIFIED_NEWTON
#include "newton.cuh"

// Base class for modified Newton's method
class ModifiedNewton : public Newton {
public:
    virtual void update (int stage);

    virtual void prepare_gpu_copy (void);
};

__global__
void
parallel_update(
    UCPSolver* base,
    ModifiedNewton* algo,
    const int stage
);

#endif // MODIFIED_NEWTON