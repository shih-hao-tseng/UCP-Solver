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
#include "modified_newton.h"

void
ModifiedNewton::update (int stage) {
    // compute dJ_du and d2J_du2
    compute_derivatives (stage);

    // only update whem the second variation is larger than 0
    for(int sample = 0; sample < _base->_total_samples; ++sample) {
        if(_d2C_du2[stage][sample] > 0.0) {
            _base->_u[stage][sample] = _base->_u[stage][sample] - _dC_du[stage][sample]/_d2C_du2[stage][sample];
        } else {
            _base->_u[stage][sample] = _base->_u[stage][sample] - _tau*_dC_du[stage][sample];
        }
    }
}