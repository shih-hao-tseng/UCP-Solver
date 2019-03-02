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
#include "witsenhausen.h"
#include "helper_functions.h"

double
Witsenhausen::get_J_value (void) {
    if (_sample_coord == nullptr)  return 0.0;

    double stage_0_cost = 0.0;
    double stage_2_cost = 0.0;
    double x0, u0, x1, w, u1, x2;
    for (int sample_x = 0; sample_x < _total_samples; ++sample_x) {
        x0 = _sample_coord[sample_x];
        u0 = _u[0][sample_x];  //get_u_value(0,x0);
        stage_0_cost += u0 * u0 * _x_prob_dx[sample_x];

        x1 = x0 + u0;
        for (int sample_w = 0; sample_w < _total_samples; ++sample_w) {
            w = _sample_coord[sample_w] * _sigma_w_to_x_ratio;
            u1 = get_u_value(1, x1 + w);
            x2 = x1 - u1;
            stage_2_cost += x2 * x2 * _x_prob_dx[sample_x] * _w_prob_dw[sample_w];
        }
    }
    stage_0_cost *= _k_2;

    // J = E[k^2 u_0^2 + x_2^2]
    return stage_0_cost + stage_2_cost;
}

void
Witsenhausen::set_k (double k) {
    _k_2 = k*k;
}

void
Witsenhausen::set_sigma_x (double sigma_x) {
    _sigma_x = sigma_x;
}

void
Witsenhausen::set_sigma_w (double sigma_w) {
    _sigma_w = sigma_w;
}

double
Witsenhausen::test_normalization_x (void) {
    if (_x_prob_dx == nullptr)  return 0.0;

    double sum = 0.0;
    for (int sample = 0; sample < _total_samples; ++sample) {
        sum += _x_prob_dx[sample];
    }
    return sum;
}

double
Witsenhausen::test_normalization_w (void) {
    if (_w_prob_dw == nullptr)  return 0.0;

    double sum = 0.0;
    for (int sample = 0; sample < _total_samples; ++sample) {
        sum += _w_prob_dw[sample];
    }
    return sum;
}

void
Witsenhausen::initialize_variables (void) {
    UCPSolver::initialize_variables ();

    _total_stages = 2;

    _x_prob    = new double [_total_samples];
    _x_prob_dx = new double [_total_samples];
    _w_prob    = new double [_total_samples];
    _w_prob_dw = new double [_total_samples];

    /*
    // ignore the corner effect
    for(int sample = 0; sample < _total_samples; ++sample) {
        _x_prob[sample] = normpdf(_sample_coord[sample],_sigma_x) * step;
        _w_prob[sample] = normpdf(_sample_coord[sample],_sigma_w) * step;
    }
    */
    // for more accurate result
    _sigma_w_to_x_ratio = _sigma_w/_sigma_x;
    double w_step_size = _sample_coord_step_size * _sigma_w_to_x_ratio;
    for(int sample = 0; sample < _total_samples; ++sample) {
        _x_prob[sample] =
            (normpdf(_sample_coord[sample]-_sample_coord_step_size/2.0,_sigma_x) +
             normpdf(_sample_coord[sample]+_sample_coord_step_size/2.0,_sigma_x)) / 2.0;
        _x_prob_dx[sample] = _x_prob[sample] * _sample_coord_step_size;

        _w_prob[sample] = 
            (normpdf(_sample_coord[sample]*_sigma_w_to_x_ratio-w_step_size/2.0,_sigma_w) +
             normpdf(_sample_coord[sample]*_sigma_w_to_x_ratio+w_step_size/2.0,_sigma_w)) / 2.0;
        _w_prob_dw[sample] = _w_prob[sample] * w_step_size;
    }
}

void
Witsenhausen::destroy_variables (void) {
    if ( _sample_coord == nullptr ) {
        return;
    }

    delete [] _x_prob;
    delete [] _x_prob_dx;
    delete [] _w_prob;
    delete [] _w_prob_dw;
    _x_prob    = nullptr;
    _x_prob_dx = nullptr;
    _w_prob    = nullptr;
    _w_prob_dw = nullptr;

    UCPSolver::destroy_variables ();
}

double
Witsenhausen::get_w_prob_value (double coord) {
    if ( _w_prob == nullptr )  return 0.0;

    if(coord + _sample_range * _sigma_w_to_x_ratio < 0.0) {
        return 0.0;
    } else if (coord > _sample_range * _sigma_w_to_x_ratio) {
        return 0.0;
    } else {
        double index = (coord/_sigma_w_to_x_ratio + _sample_range) / _sample_coord_step_size;
        int id_max = ceil(index);
        int id_min = floor(index);
        if (id_max == id_min) {
            return _w_prob[id_max];
        } else {
            double max = _w_prob[id_max];
            double min = _w_prob[id_min];
            double coord_max = _sample_coord[id_max] * _sigma_w_to_x_ratio;
            double coord_min = _sample_coord[id_min] * _sigma_w_to_x_ratio;
            return (max*(coord - coord_min) + min*(coord_max - coord))/(coord_max - coord_min);
        }
    }
}

double
Witsenhausen::compute_C_value (int stage, double u_m, double y_m) {
    double C_value = 0.0;
    double x0, u0, x1, w, u1, x2;
    switch (stage) {
        case 0:
        {
            // scaling gives the same result
            //int sample_x = (y_m + _sample_range) / _sample_coord_step_size;
            x0 = y_m;
            u0 = u_m;
            C_value += _k_2 * u0 * u0;// * _x_prob[sample_x];

            x1 = x0 + u0;
            for (int sample_w = 0; sample_w < _total_samples; ++sample_w) {
                w = _sample_coord[sample_w] * _sigma_w_to_x_ratio;
                u1 = get_u_value(1, x1 + w);
                x2 = x1 - u1;
                //C_value += x2 * x2 * _x_prob[sample_x] * _w_prob_dw[sample_w];
                C_value += x2 * x2 * _w_prob_dw[sample_w];
            }
            break;
        }
        case 1:
        {
            u1 = u_m;
            for (int sample_x = 0; sample_x < _total_samples; ++sample_x) {
                x0 = _sample_coord[sample_x];
                u0 = _u[0][sample_x];  //get_u_value(0,x0);
                x1 = x0 + u0;
                x2 = x1 - u1;
                C_value += x2 * x2 * _x_prob_dx[sample_x] * get_w_prob_value(y_m - x1);
            }
            break;
        }
        default:
            break;
    }
    return C_value;
}

void
Witsenhausen_GradientMomentum::compute_dC_du (int stage) {
    Witsenhausen* base = dynamic_cast<Witsenhausen*>(_base);
    double x0, u0, x1, w, u1, x2, y1;
    switch (stage) {
        case 0:
        {
            double du1_du0;
            for(int sample_y = 0; sample_y < base->_total_samples; ++sample_y) {
                x0 = base->_sample_coord[sample_y];
                u0 = base->_u[0][sample_y];
                _dC_du[0][sample_y] = base->_k_2 * 2 * u0 * base->_x_prob[sample_y];

                x1 = x0 + u0;
                for (int sample_w = 0; sample_w < base->_total_samples; ++sample_w) {
                    w = base->_sample_coord[sample_w] * base->_sigma_w_to_x_ratio;
                    y1 = x1 + w;
                    u1 = base->get_u_value(1, y1);
                    du1_du0 = base->get_du_value(1, y1);
                    x2 = x1 - u1;
                    _dC_du[0][sample_y] += 2 * (1 - du1_du0) * x2 * base->_x_prob[sample_y] * base->_w_prob_dw[sample_w];
                }
            }
            break;
        }
        case 1:
        {
            for(int sample_y = 0; sample_y < base->_total_samples; ++sample_y) {
                _dC_du[1][sample_y] = 0.0;

                y1 = base->_sample_coord[sample_y];
                u1 = base->_u[1][sample_y];
                for (int sample_x = 0; sample_x < base->_total_samples; ++sample_x) {
                    x0 = base->_sample_coord[sample_x];
                    u0 = base->_u[0][sample_x];
                    x1 = x0 + u0;
                    w  = y1 - x1;
                    x2 = x1 - u1;
                    _dC_du[1][sample_y] += - 2 * x2 * base->_x_prob_dx[sample_x] * base->get_w_prob_value(w);
                }
            }
            break;
        }
        default:
            break;
    }
}

void
Witsenhausen_ModifiedNewton::compute_derivatives (int stage) {
    Witsenhausen* base = dynamic_cast<Witsenhausen*>(_base);

    // Also compute d2C_du2
    double x0, u0, x1, w, u1, x2, y1;
    switch (stage) {
        case 0:
        {
            double du1_du0, d2u1_du02;
            for(int sample_y = 0; sample_y < base->_total_samples; ++sample_y) {
                _dC_du  [0][sample_y] = 0.0;
                _d2C_du2[0][sample_y] = 0.0;

                x0 = base->_sample_coord[sample_y];
                u0 = base->_u[0][sample_y];
                _dC_du  [0][sample_y] += base->_k_2 * 2 * u0 * base->_x_prob[sample_y];
                _d2C_du2[0][sample_y] += base->_k_2 * 2 * base->_x_prob[sample_y];

                x1 = x0 + u0;
                for (int sample_w = 0; sample_w < base->_total_samples; ++sample_w) {
                    w = base->_sample_coord[sample_w] * base->_sigma_w_to_x_ratio;
                    y1 = x1 + w;
                    u1 = base->get_u_value(1, y1);
                    du1_du0 = base->get_du_value(1, y1);
                    d2u1_du02 = base->get_d2u_value(1, y1);
                    x2 = x1 - u1;
                    _dC_du  [0][sample_y] += 2 * (1 - du1_du0) * x2 * base->_x_prob[sample_y] * base->_w_prob_dw[sample_w];
                    _d2C_du2[0][sample_y] += 2 * ( -d2u1_du02 * x2 + (1 - du1_du0) * (1 - du1_du0)) * base->_x_prob[sample_y] * base->_w_prob_dw[sample_w];
                }
            }
            break;
        }
        case 1:
        {
            for(int sample_y = 0; sample_y < base->_total_samples; ++sample_y) {
                _dC_du  [1][sample_y] = 0.0;
                _d2C_du2[1][sample_y] = 0.0;

                y1 = base->_sample_coord[sample_y];
                u1 = base->_u[1][sample_y];
                for (int sample_x = 0; sample_x < base->_total_samples; ++sample_x) {
                    x0 = base->_sample_coord[sample_x];
                    u0 = base->_u[0][sample_x];
                    x1 = x0 + u0;
                    w  = y1 - x1;
                    x2 = x1 - u1;
                    _dC_du  [1][sample_y] += - 2 * x2 * base->_x_prob_dx[sample_x] * base->get_w_prob_value(w);
                    _d2C_du2[1][sample_y] += 2 * base->_x_prob_dx[sample_x] * base->get_w_prob_value(w);
                }
            }
            break;
        }
        default:
            break;
    }
}