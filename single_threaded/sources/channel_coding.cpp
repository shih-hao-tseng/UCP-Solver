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
#include "channel_coding.h"
#include "helper_functions.h"
#include <iostream>
double
ChannelCoding::get_J_value (void) {
    if (_sample_coord == nullptr)  return 0.0;

    _deviation_cost = 0.0;
    _power_cost = 0.0;
    double x0, x1, u0, u1, dev;
    for (int sample_x = 0; sample_x < _total_samples; ++sample_x) {
        x0 = _sample_coord[sample_x];
        u0 = _u[0][sample_x];  //get_u_value(0,x0);
        _power_cost += u0 * u0 * _x_prob_dx[sample_x];

        for (double w = -1.0 + _sample_coord_step_size/2; w <= 1.0; w += _sample_coord_step_size) {
            x1 = u0 + w;
            u1 = get_u_value(1,x1);
            dev = (u1 - x0);
            _deviation_cost += dev * dev * _x_prob_dx[sample_x] * _unif_w_prob_dw;
        }
    }

    // J = E[(x2-x0)^2] + lambda * E[u0^2]
    return _deviation_cost + _lambda * _power_cost;
}

void
ChannelCoding::set_lambda (double lambda) {
    _lambda = lambda;
}

void
ChannelCoding::set_sigma_x (double sigma_x) {
    _sigma_x = sigma_x;
}

double
ChannelCoding::test_normalization_x (void) {
    if (_x_prob_dx == nullptr)  return 0.0;

    double sum = 0.0;
    for (int sample = 0; sample < _total_samples; ++sample) {
        sum += _x_prob_dx[sample];
    }
    return sum;
}

void
ChannelCoding::initialize_variables (void) {
    UCPSolver::initialize_variables ();

    _total_stages = 2;

    _x_prob    = new double [_total_samples];
    _x_prob_dx = new double [_total_samples];

    // for more accurate result
    for(int sample = 0; sample < _total_samples; ++sample) {
        _x_prob[sample] =
            (normpdf(_sample_coord[sample]-_sample_coord_step_size/2.0,_sigma_x) +
             normpdf(_sample_coord[sample]+_sample_coord_step_size/2.0,_sigma_x)) / 2.0;
        _x_prob_dx[sample] = _x_prob[sample] * _sample_coord_step_size;
    }

    _unif_w_prob_dw = _sample_coord_step_size / 2;
}

void
ChannelCoding::destroy_variables (void) {
    if ( _sample_coord == nullptr ) {
        return;
    }

    delete [] _x_prob;
    delete [] _x_prob_dx;
    _x_prob    = nullptr;
    _x_prob_dx = nullptr;

    UCPSolver::destroy_variables ();
}

void
ChannelCoding::additional_report (std::ostream& os) {
    os << "\tgamma = " << 10 * log10(_power_cost*3.0) << " dB";
    return;
}

void
ChannelCoding::additional_log (std::ostream& os) {
    os << "\t" << 10 * log10(_power_cost*3.0);
    return;
}

double
ChannelCoding::compute_C_value (int stage, double u_m, double y_m) {
    double C_value = 0.0;
    double x0, x1, u0, u1, dev;
    switch (stage) {
        case 0:
        {
            x0 = y_m;
            u0 = u_m;
            C_value += _lambda * u0 * u0;

            for (double w = -1.0 + _sample_coord_step_size/2; w <= 1.0; w += _sample_coord_step_size) {
                x1 = u0 + w;
                u1 = get_u_value(1,x1);
                dev = (u1 - x0);
                C_value += dev * dev * _unif_w_prob_dw;
            }
            break;
        }
        case 1:
        {
            u1 = u_m;
            for (int sample_x = 0; sample_x < _total_samples; ++sample_x) {
                x0 = _sample_coord[sample_x];
                u0 = _u[0][sample_x];
                dev = (u1 - x0);
                // w = x1 - u0 = y1 - u0
                C_value += dev * dev * _x_prob_dx[sample_x] * get_w_prob_value(y_m - u0);
            }
            break;
        }
        default:
            break;
    }
    return C_value;
}

double
ChannelCoding::get_w_prob_value (double coord) {
    if(coord > 1.0) {
        return 0.0;
    }
    if(coord < -1.0) {
        return 0.0;
    }
    return _unif_w_prob_dw;
}

void
ChannelCoding_GradientMomentum::compute_dC_du (int stage) {
    ChannelCoding* base = dynamic_cast<ChannelCoding*>(_base);
    double x0, x1, u0, u1, dev;
    switch (stage) {
        case 0:
        {
            double du1_du0;
            for(int sample_y = 0; sample_y < base->_total_samples; ++sample_y) {
                x0 = base->_sample_coord[sample_y];
                u0 = base->_u[0][sample_y];
                _dC_du[0][sample_y] = 2 * base->_lambda * u0 * base->_x_prob[sample_y];

                for (double w = -1.0 + base->_sample_coord_step_size/2; w <= 1.0; w += base->_sample_coord_step_size) {
                    x1 = u0 + w;
                    u1 = base->get_u_value(1,x1);
                    du1_du0 = base->get_du_value(1, x1);
                    dev = (u1 - x0);
                    _dC_du[0][sample_y] += 2 * du1_du0 * dev * base->_x_prob[sample_y] * base->_unif_w_prob_dw;
                }
            }
            break;
        }
        case 1:
        {
            for(int sample_y = 0; sample_y < base->_total_samples; ++sample_y) {
                _dC_du[1][sample_y] = 0.0;

                x1 = base->_sample_coord[sample_y];
                u1 = base->_u[1][sample_y];
                for (int sample_x = 0; sample_x < base->_total_samples; ++sample_x) {
                    x0 = base->_sample_coord[sample_x];
                    u0 = base->_u[0][sample_x];
                    dev = (u1 - x0);
                    // w = x1 - u0 = y1 - u0
                    _dC_du[1][sample_y] += 2 * dev * base->_x_prob_dx[sample_x] * base->get_w_prob_value(x1 - u0);
                }
            }
            break;
        }
        default:
            break;
    }
}

void
ChannelCoding_ModifiedNewton::compute_derivatives (int stage) {
    ChannelCoding* base = dynamic_cast<ChannelCoding*>(_base);
    double x0, x1, u0, u1, dev;
    switch (stage) {
        case 0:
        {
            double du1_du0, d2u1_du02;
            for(int sample_y = 0; sample_y < base->_total_samples; ++sample_y) {
                x0 = base->_sample_coord[sample_y];
                u0 = base->_u[0][sample_y];
                _dC_du  [0][sample_y] = 2 * base->_lambda * u0 * base->_x_prob[sample_y];
                _d2C_du2[0][sample_y] = 2 * base->_lambda * base->_x_prob[sample_y];

                for (double w = -1.0 + base->_sample_coord_step_size/2; w <= 1.0; w += base->_sample_coord_step_size) {
                    x1 = u0 + w;
                    u1 = base->get_u_value(1,x1);
                    du1_du0 = base->get_du_value(1, x1);
                    d2u1_du02 = base->get_d2u_value(1, x1);
                    dev = (u1 - x0);
                    _dC_du  [0][sample_y] += 2 * du1_du0 * dev * base->_x_prob[sample_y] * base->_unif_w_prob_dw;
                    _d2C_du2[0][sample_y] += 2 * (d2u1_du02 * dev + du1_du0 * du1_du0) * base->_x_prob[sample_y] * base->_unif_w_prob_dw;
                }
            }
            break;
        }
        case 1:
        {
            for(int sample_y = 0; sample_y < base->_total_samples; ++sample_y) {
                _dC_du  [1][sample_y] = 0.0;
                _d2C_du2[1][sample_y] = 0.0;

                x1 = base->_sample_coord[sample_y];
                u1 = base->_u[1][sample_y];
                for (int sample_x = 0; sample_x < base->_total_samples; ++sample_x) {
                    x0 = base->_sample_coord[sample_x];
                    u0 = base->_u[0][sample_x];
                    dev = (u1 - x0);
                    // w = x1 - u0 = y1 - u0
                    _dC_du  [1][sample_y] += 2 * dev * base->_x_prob_dx[sample_x] * base->get_w_prob_value(x1 - u0);
                    _d2C_du2[1][sample_y] += 2 * base->_x_prob_dx[sample_x] * base->get_w_prob_value(x1 - u0);
                }
            }
            break;
        }
        default:
            break;
    }
}