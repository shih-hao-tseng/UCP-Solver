#include "channel_coding.h"
#include <iostream>
#include <sstream>
using namespace std;

#define TOTAL_STAGES  2
#define TOTAL_SAMPLES 10000
#define SAMPLE_RANGE  5.0

int main () {
	cout.precision(17);

    ChannelCoding prob;
    prob.set_error_bound      (1e-6);
    prob.set_total_iterations (20);
    prob.set_total_stages     (2);
    prob.set_total_samples    (TOTAL_SAMPLES);

    prob.set_sample_range     (SAMPLE_RANGE);
    prob.set_denoise_radius   (2.1*SAMPLE_RANGE*2/(TOTAL_SAMPLES-1));

    //prob.set_lambda  (1.535);  // -5.7dB
    //prob.set_lambda  (0.88);  // -1.69dB
    prob.set_lambda  (0.1345);  // 5.69dB
    prob.set_sigma_x (1.0);

    //ChannelCoding_GradientMomentum algo;
    ChannelCoding_ModifiedNewton algo;

    prob.set_local_update (&algo);

    system("mkdir -p results/coding");
    string file_names[TOTAL_STAGES];
    for(int stage = 0; stage < TOTAL_STAGES; ++stage) {
        stringstream file_name;
        file_name << "results/coding/u" << stage << ".dat";
        file_names[stage] = file_name.str();
    }
    prob.set_output_file_names (file_names);

    //prob.initialize_u_from_input_file (file_names);

    prob.set_log_file_name ("results/coding/log.dat");

    prob.compute_result(true,false);

    cout << "J = " << prob.get_J_value() << endl;

    return 0;
}
