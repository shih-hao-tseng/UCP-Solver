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
#include "newton.h"

void
Newton::initialize_variables (void) {
    destroy_variables ();
    _dC_du   = new double*[_base->_total_stages];
    _d2C_du2 = new double*[_base->_total_stages];
    for(int stage = 0; stage < _base->_total_stages; ++stage) {
        _dC_du[stage]   = new double [_base->_total_samples];
        _d2C_du2[stage] = new double [_base->_total_samples];
    }
    return;
}

void
Newton::destroy_variables (void) {
    if (_dC_du == nullptr) {
        return;
    }
    for(int stage = 0; stage < _base->_total_stages; ++stage) {
        delete [] _dC_du[stage];
        delete [] _d2C_du2[stage];
    }
    delete [] _dC_du;
    delete [] _d2C_du2;
    _dC_du   = nullptr;
    _d2C_du2 = nullptr;
    return;
}

void
Newton::update (int stage) {
    // compute dJ_du and d2J_du2
    compute_derivatives (stage);

    // update by Newton's method
    for(int sample = 0; sample < _base->_total_samples; ++sample) {
        _base->_u[stage][sample] = _base->_u[stage][sample] - _dC_du[stage][sample]/_d2C_du2[stage][sample];
    }
    return;
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