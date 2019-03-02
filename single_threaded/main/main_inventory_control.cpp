#include "inventory_control.h"
#include <iostream>
#include <sstream>
using namespace std;

#define TOTAL_STAGES  2
#define TOTAL_SAMPLES 1000
#define SAMPLE_RANGE  5.0

int main () {
	cout.precision(17);

    InventoryControl prob;
    prob.set_error_bound      (1e-10);
    prob.set_total_iterations (20);
    prob.set_total_stages     (2);
    prob.set_total_samples    (TOTAL_SAMPLES);

    prob.set_sample_range     (SAMPLE_RANGE);
    prob.set_denoise_radius   (2.1*2*SAMPLE_RANGE/(TOTAL_SAMPLES-1));

    //InventoryControl_GradientMomentum algo;
    InventoryControl_ModifiedNewton algo;

    prob.set_local_update (&algo);

    system("mkdir -p results/inventory");
    string file_names[TOTAL_STAGES];
    for(int stage = 0; stage < TOTAL_STAGES; ++stage) {
        stringstream file_name;
        file_name << "results/inventory/u" << stage << ".dat";
        file_names[stage] = file_name.str();
    }
    prob.set_output_file_names (file_names);

    prob.set_log_file_name ("results/inventory/log.dat");

    //prob.initialize_u_from_input_file (file_names);

    prob.compute_result(true,false);

    cout << "J = " << prob.get_J_value() << endl;

    return 0;
}
