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
#include "gradient_momentum.cuh"
#include <cstring>

void
GradientMomentum::initialize_variables (void) {
    destroy_variables ();

    cudaMalloc(&_v,     _base->_total_stages*_base->_total_samples*sizeof(double));
    cudaMalloc(&_dC_du, _base->_total_stages*_base->_total_samples*sizeof(double));

    for (int stage = 0; stage < _base->_total_stages; ++stage) {
        reset_v (stage);
    }

    return;
}

void
GradientMomentum::destroy_variables (void) {
    if (_v == nullptr) {
        return;
    }
    cudaFree(_v);
    cudaFree(_dC_du);
    _v = nullptr;
    _dC_du = nullptr;
    return;
}

void
GradientMomentum::update (int stage) {
    // compute dC_du
    compute_dC_du(stage);

    // update by gradient with momentum
    // again, we should update after computing dC_du
    parallel_update<<<(_base->_total_samples+255)/256, 256>>> (
        (UCPSolver*) _base->_prob_at_gpu,
        (GradientMomentum*) _algo_at_gpu,
        stage
    );
    cudaDeviceSynchronize ();

    return;
}

__global__
void
parallel_update(
    UCPSolver* base,
    GradientMomentum* algo,
    const int stage
) {
    int sample = blockIdx.x*blockDim.x + threadIdx.x;
    if(sample >= base->_total_samples)  return;
    sample += stage * base->_total_samples;

    algo->_v[sample] = algo->_beta * algo->_v[sample] - algo->_tau * algo->_dC_du[sample];
    base->_u[sample] = base->_u[sample] + algo->_v[sample];
}

void
GradientMomentum::denoised (int stage) {
    // reset v
    reset_v(stage);
    return;
}

void
GradientMomentum::set_beta (double beta) {
    _beta = beta;
}

void
GradientMomentum::set_tau (double tau) {
    _tau = tau;
}

void
GradientMomentum::reset_v(int stage) {
    parallel_reset_v<<<(_base->_total_samples+255)/256, 256>>> (
        stage,
        _base->_total_samples,
        _v
    );
    cudaDeviceSynchronize ();
}

__global__
void
parallel_reset_v(
    const int stage,
    const int total_samples,
    double* v
) {
    int sample = blockIdx.x*blockDim.x + threadIdx.x;
    if(sample >= total_samples)  return;
    sample += stage * total_samples;
    v[sample] = 0.0;
}

void
GradientMomentum::compute_dC_du (int stage) {
    return;
}

ALGO_PARALLEL_SUITE(GradientMomentum)