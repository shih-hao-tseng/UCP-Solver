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
#include "modified_newton.cuh"

void
ModifiedNewton::update (int stage) {
    // compute dC_du and d2C_du2
    compute_derivatives (stage);

    parallel_update<<<(_base->_total_samples+255)/256, 256>>> (
        (UCPSolver*) _base->_prob_at_gpu,
        (ModifiedNewton*) _algo_at_gpu,
        stage
    );

    cudaDeviceSynchronize();
/*
    int start = stage * _base->_total_samples;
    for(int sample = start; sample < start + _base->_total_samples; ++sample) {
        if(_d2C_du2[sample] > 0.0) {
            _base->_u[sample] = _base->_u[sample] - _dC_du[sample]/_d2C_du2[sample];
        } else {
            _base->_u[sample] = _base->_u[sample] - _tau*_dC_du[sample];
        }        
    }
*/
}

__global__
void
parallel_update(
    UCPSolver* base,
    ModifiedNewton* algo,
    const int stage
) {
    int sample = blockIdx.x*blockDim.x + threadIdx.x;
    if(sample >= base->_total_samples)  return;
    sample += stage * base->_total_samples;

    if(algo->_d2C_du2[sample] > 0.0) {
        base->_u[sample] = base->_u[sample] - algo->_dC_du[sample]/algo->_d2C_du2[sample];
    } else {
        base->_u[sample] = base->_u[sample] - algo->_tau*algo->_dC_du[sample];
    }
}

ALGO_PARALLEL_SUITE(ModifiedNewton)