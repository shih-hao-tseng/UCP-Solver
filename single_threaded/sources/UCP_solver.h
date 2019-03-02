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
#ifndef UCP_SOLVER_H
#define UCP_SOLVER_H
#include <string>

class UCPSolver;

class LocalUpdate {
public:
    ~LocalUpdate () {
        destroy_variables ();
    }

    void set_base (UCPSolver* base);

    virtual void initialize_variables (void);
    virtual void destroy_variables (void);

    virtual void update (int stage);
    virtual void denoised (int stage);

protected:
    UCPSolver* _base;
};

class UCPSolver {
public:
    ~UCPSolver () {
        destroy_variables ();
    }

    virtual double get_J_value (void);
    double* get_u (int stage);

    void compute_result (bool output_file = false, bool silent = true);

    void set_error_bound (double);
    void set_total_iterations (int);
    void set_total_stages (int);
    void set_total_samples (int);

    void set_sample_range (double);
    void set_denoise_radius (double);

    void set_local_update (LocalUpdate* local_update);

    void set_output_file_names (std::string* output_file_names);
    void initialize_u_from_input_file (std::string* input_file_names, bool forced_initialize = false);
    void set_log_file_name (std::string log_file_name);

//protected:
    virtual void initialize_variables (void);
    virtual void destroy_variables (void);

    // output some additional information, designed for the derived classes
    virtual void additional_report (std::ostream& os);
    virtual void additional_log (std::ostream& os);

    virtual void u_initializer (void);
    // get u value using linear interpolation
    double get_u_value (int stage, double coord);
    // compute du/dy
    double get_du_value (int stage, double coord);
    // compute d^2u/dy^2
    double get_d2u_value (int stage, double coord);

    // return true if u_m is locally denoised
    bool u_denoise (int stage);
    // for u_denoise
    virtual double compute_C_value (int stage, double u_m, double y_m);

    double _error_bound      {0.0000000001};
    int    _total_iterations {20};
    int    _total_stages     {2};
    int    _total_samples    {20000};

    double _sample_range         {25.0};
    double _denoise_radius       {0.01};
    int    _denoise_index_radius {0};

    LocalUpdate* _local_update {nullptr};

    double*  _sample_coord     {nullptr};  // sampled coordinates
    double   _sample_coord_step_size {1.0};

    double** _u         {nullptr};
    double** _u_denoise {nullptr};

    std::string* _output_file_names {nullptr};
    std::string* _input_file_names  {nullptr};
    bool _u_initialized_from_input  {false};
    std::string  _log_file_name;
};

#endif // UCP_SOLVER_H