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
#include "inventory_control.h"
#include <math.h>

double
InventoryControl::get_J_value (void) {
    if (_sample_coord == nullptr)  return 0.0;
    double retval = 0.0;
    for (double x_0 = -1.0 + _sample_coord_step_size/2; x_0 <= 1.0; x_0 += _sample_coord_step_size) {
        retval += get_J_m_value (0, x_0);
    }
    retval *= _unif_w_prob_dw;
    return retval;
}

void
InventoryControl::set_h (double h) {
    _h = h;
}

void
InventoryControl::set_l (double l) {
    _l = l;
}

void
InventoryControl::set_xi (double xi) {
    _xi = xi;
}

void
InventoryControl::initialize_variables (void) {
    UCPSolver::initialize_variables ();

    _unif_w_prob_dw = _sample_coord_step_size / 2.0;
}

double
InventoryControl::gamma (double a) {
    if (a > 0.0) {
        return a * _h;
    }
    if (a < 0.0) {
        return - a * _l;
    }
    return 0.0;
}

double
InventoryControl::d_gamma (double a) {
    if (a > 0.0) {
        return _h;
    }
    if (a < 0.0) {
        return -_l;
    }
    return 0.0;
}

double
InventoryControl::get_J_m_value (int stage, double x_m) {
    //   E[ \sum\limits_{m=0}^M \xi u_m + \gamma(x_m + u_m - w_m) ]
    // = E[ \sum\limits_{m=0}^M \xi u_m + \gamma(x_{m+1}) ]
    if( (stage < 0) || (stage > _total_stages - 1) ) {
        return 0.0;
    }
    // compute \xi u_m(x_m) + E[ \gamma(x_{m+1}) ]
    double retval = 0.0;
    double u_m = get_u_value (stage, x_m);
    //retval += _xi * u_m;
    double x_m_and_u_m = u_m + x_m;
    double x_m1;
    for (double w_m = -1.0 + _sample_coord_step_size/2; w_m <= 1.0; w_m += _sample_coord_step_size) {
        x_m1 = x_m_and_u_m - w_m;
        retval += (gamma(x_m1) + get_J_m_value(stage + 1, x_m1));
    }
    retval *= _unif_w_prob_dw;
    retval += _xi * u_m;
    return retval;
}

double
InventoryControl::get_dJ_m_value (int stage, double x_m, double dx_m_du0) {
    if( (stage < 0) || (stage > _total_stages - 1) ) {
        return 0.0;
    }
    double du1 = get_du_value (stage, x_m);
    double retval = 0.0;
    //retval += _xi * du1 * dx_m_du0;
    double x_m_and_u_m = x_m + get_u_value(stage, x_m);
    double x_m1;
    double dx_m1_du0 = (1.0 + du1) * dx_m_du0;
    for (double w_m = -1.0 + _sample_coord_step_size/2; w_m <= 1.0; w_m += _sample_coord_step_size) {
        x_m1 = x_m_and_u_m - w_m;
        retval += (d_gamma(x_m1) * dx_m1_du0 + get_dJ_m_value(stage + 1, x_m1, dx_m1_du0));
    }
    retval *= _unif_w_prob_dw;
    retval += _xi * du1 * dx_m_du0;
    return retval;
}

double
InventoryControl::compute_C_value (int stage, double u_m, double y_m) {
    double C_value = 0.0; 
    //C_value += _xi * u_m;
    // scaling is not important under the same y_m
    // and hence we can consider only the effect after the current stage 
    double x_m_and_u_m = y_m + u_m;
    double x_m1;
    for (double w_m = -1.0 + _sample_coord_step_size/2; w_m <= 1.0; w_m += _sample_coord_step_size) {
        x_m1 = x_m_and_u_m - w_m;
        C_value += (gamma(x_m1) + get_J_m_value(stage + 1, x_m1));
    }
    C_value *= _unif_w_prob_dw;
    C_value += _xi * u_m;
    return C_value;
}

void
InventoryControl_GradientMomentum::update (int stage) {
    // compute dC_du
    compute_dC_du (stage);

    // update by gradient with momentum
    for(int sample = 0; sample < _base->_total_samples; ++sample) {
        _v[stage][sample] = _beta*_v[stage][sample] - _tau*_dC_du[stage][sample];
        _base->_u[stage][sample] = _base->_u[stage][sample] + _v[stage][sample];
        if(_base->_u[stage][sample] < 0.0) {
            // non negative constraints
            _base->_u[stage][sample] = 0.0;
        }
    }
}

void
InventoryControl_GradientMomentum::compute_dC_du (int stage) {
    if((stage < 0) || (stage > _base->_total_stages - 1)) {
        return;
    }

    InventoryControl* base = dynamic_cast<InventoryControl*>(_base);

    double x_m_and_u_m, x_m1;
    for(int sample_y = 0; sample_y < base->_total_samples; ++sample_y) {
        x_m_and_u_m = base->_sample_coord[sample_y] + base->_u[stage][sample_y];
        _dC_du[stage][sample_y] = 0.0;
        //_dC_du[stage][sample_y] += base->_xi;
        // stage
        for (double w_m = -1.0 + base->_sample_coord_step_size/2; w_m <= 1.0; w_m += base->_sample_coord_step_size) {
            x_m1 = x_m_and_u_m - w_m;
            _dC_du[stage][sample_y] += (base->d_gamma(x_m1) + base->get_dJ_m_value(stage + 1, x_m1, 1));
        }
        _dC_du[stage][sample_y] *= base->_unif_w_prob_dw;
        _dC_du[stage][sample_y] += base->_xi;
    }
}

void
InventoryControl_ModifiedNewton::update (int stage) {
    // compute dJ_du and d2J_du2
    compute_derivatives (stage);

    // update by Newton's method
    for(int sample = 0; sample < _base->_total_samples; ++sample) {
        _base->_u[stage][sample] = _base->_u[stage][sample] - _dC_du[stage][sample]/_d2C_du2[stage][sample];
        if(_base->_u[stage][sample] < 0.0) {
            // non negative constraints
            _base->_u[stage][sample] = 0.0;
        }
    }
    return;
}

void
InventoryControl_ModifiedNewton::compute_derivatives (int stage) {
    if((stage < 0) || (stage > _base->_total_stages - 1)) {
        return;
    }

    InventoryControl* base = dynamic_cast<InventoryControl*>(_base);

    double x_m, u_m, x_m_and_u_m, x_m1;
    double half_step = base->_sample_coord_step_size/2;
    for(int sample_y = 0; sample_y < base->_total_samples; ++sample_y) {
        x_m = base->_sample_coord[sample_y];
        u_m = base->_u[stage][sample_y];
        /*
        x_m_and_u_m = x_m + u_m;
        _dC_du[stage][sample_y] = 0.0; //base->_xi;

        // stage
        for (double w_m = -1.0 + half_step; w_m <= 1.0; w_m += 2*half_step) {
            x_m1 = x_m_and_u_m - w_m;
            _dC_du[stage][sample_y] += (base->d_gamma(x_m1) + base->get_dJ_m_value(stage + 1, x_m1, 1));
        }
        _dC_du[stage][sample_y] *= base->_unif_w_prob_dw;
        _dC_du[stage][sample_y] += base->_xi;
        */
        _dC_du[stage][sample_y] = (
            base->compute_C_value(stage, u_m + half_step, x_m) 
          - base->compute_C_value(stage, u_m - half_step, x_m) 
        ) / (2*half_step);
        _d2C_du2[stage][sample_y] = (
            base->compute_C_value(stage, u_m + small_step, x_m) 
         +  base->compute_C_value(stage, u_m - small_step, x_m) 
         -2*base->compute_C_value(stage, u_m, x_m)
        );
        // for precision problem
        if(_d2C_du2[stage][sample_y] < 1e-10 ) {
            // dont use modified newton
            _d2C_du2[stage][sample_y] = -1.0;
        }
        _d2C_du2[stage][sample_y] /= (small_step*small_step);
    }
}