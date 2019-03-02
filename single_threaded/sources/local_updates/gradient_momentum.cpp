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
#include "gradient_momentum.h"
#include <cstring>

void
GradientMomentum::initialize_variables (void) {
    destroy_variables ();
    _v     = new double*[_base->_total_stages];
    _dC_du = new double*[_base->_total_stages];
    for(int stage = 0; stage < _base->_total_stages; ++stage) {
        _v[stage]     = new double [_base->_total_samples];
        _dC_du[stage] = new double [_base->_total_samples];
        memset(&_v[stage][0],0.0,_base->_total_samples*sizeof(double));
    }
    return;
}

void
GradientMomentum::destroy_variables (void) {
    if (_v == nullptr) {
        return;
    }
    for(int stage = 0; stage < _base->_total_stages; ++stage) {
        delete [] _v[stage];
        delete [] _dC_du[stage];
    }
    delete [] _v;
    delete [] _dC_du;
    _v = nullptr;
    _dC_du = nullptr;
    return;
}

void
GradientMomentum::update (int stage) {
    // compute dJ_du
    compute_dC_du(stage);

    // update by gradient with momentum
    // again, we should update after computing dJ_du
    for(int sample = 0; sample < _base->_total_samples; ++sample) {
        _v[stage][sample] = _beta*_v[stage][sample] - _tau*_dC_du[stage][sample];
        _base->_u[stage][sample] = _base->_u[stage][sample] + _v[stage][sample];
    }
    return;
}

void
GradientMomentum::denoised (int stage) {
    // reset v
    memset(&_v[stage][0],0.0,_base->_total_samples*sizeof(double));
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
GradientMomentum::compute_dC_du (int stage) {
    return;
}