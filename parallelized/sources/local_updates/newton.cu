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
#include "newton.cuh"

void
Newton::initialize_variables (void) {
    destroy_variables ();
    cudaMalloc(&_dC_du,   _base->_total_stages*_base->_total_samples*sizeof(double));
    cudaMalloc(&_d2C_du2, _base->_total_stages*_base->_total_samples*sizeof(double));
    return;
}

void
Newton::destroy_variables (void) {
    if (_dC_du == nullptr) {
        return;
    }
    cudaFree(_dC_du);
    cudaFree(_d2C_du2);
    _dC_du   = nullptr;
    _d2C_du2 = nullptr;
    return;
}

void
Newton::update (int stage) {
    // compute dJ_du and d2J_du2
    compute_derivatives (stage);

    // update by Newton's method
    parallel_update<<<(_base->_total_samples+255)/256, 256>>> (
        (UCPSolver*) _base->_prob_at_gpu,
        (Newton*) _algo_at_gpu,
        stage
    );
    cudaDeviceSynchronize();

    return;
}

__global__
void
parallel_update(
    UCPSolver* base,
    Newton* algo,
    const int stage
) {
    int sample = blockIdx.x*blockDim.x + threadIdx.x;
    if(sample >= base->_total_samples)  return;
    sample += stage * base->_total_samples;

    base->_u[sample] = base->_u[sample] - algo->_dC_du[sample]/algo->_d2C_du2[sample];
}

void
Newton::denoised (int stage) {
    return;
}

void
Newton::set_tau (double tau) {
    _tau = tau;
}

void
Newton::compute_derivatives (int stage) {
    return;
}

ALGO_PARALLEL_SUITE(Newton)