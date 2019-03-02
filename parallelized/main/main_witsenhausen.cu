#include "witsenhausen.cuh"
#include <iostream>
#include <sstream>
using namespace std;

#define TOTAL_STAGES  2
#define TOTAL_SAMPLES 14000
#define SAMPLE_RANGE  25.0

int main () {
	cout.precision(17);

    Witsenhausen prob;
    prob.set_error_bound      (1e-10);
    prob.set_total_iterations (20);
    prob.set_total_samples    (TOTAL_SAMPLES);

    prob.set_sample_range     (SAMPLE_RANGE);
    prob.set_denoise_radius   (5.1*2*SAMPLE_RANGE/(TOTAL_SAMPLES-1));

    Witsenhausen_ModifiedNewton algo;
    algo.set_tau              (0.1);

    prob.set_local_update (&algo);

    system("mkdir -p results/witsenhausen");
    string file_names[TOTAL_STAGES];
    for(int stage = 0; stage < TOTAL_STAGES; ++stage) {
        stringstream file_name;
        file_name << "results/witsenhausen/u" << stage << ".dat";
        file_names[stage] = file_name.str();
    }
    prob.set_output_file_names (file_names);
/*
    string input_file_names[TOTAL_STAGES];
    for(int stage = 0; stage < TOTAL_STAGES; ++stage) {
        stringstream file_name;
        file_name << "results/witsenhausen/u" << stage << ".dat";
        input_file_names[stage] = file_name.str();
    }
    prob.initialize_u_from_input_file (input_file_names);
*/
    prob.set_log_file_name ("results/witsenhausen/log.dat");

    prob.compute_result(true,false);

    cout << "J = " << prob.get_J_value() << endl;
    cout << "x normalization = " << prob.test_normalization_x() << endl;
    cout << "w normalization = " << prob.test_normalization_w() << endl;

    return 0;
}
