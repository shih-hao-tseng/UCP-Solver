#include <math.h> 
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <time.h>
#include <list>
#include <cmath>

#define PI            3.141592653589793238462643383279502884 // 197169399
#define INV_SQRT_2PI  0.398942280401433
#define TWO_SQRT_TWO  2.828427124746190
#define ERROR_BOUND   0.0000000001
//#define TOTAL_SAMPLES 20000
#define SAMPLE_RANGE  5 // +- SAMPLE_RANGE sigma
#define LOCAL_INTERVAL_RADIUS 2
using namespace std;

double normpdf(const double x, const double mu, const double sigma){
	double power = (x-mu)/sigma;
	return exp(-power*power/2)*INV_SQRT_2PI/sigma;
}


double phi(double x)
{
    // constants
    double a1 =  0.254829592;
    double a2 = -0.284496736;
    double a3 =  1.421413741;
    double a4 = -1.453152027;
    double a5 =  1.061405429;
    double p  =  0.3275911;

    // Save the sign of x
    int sign = 1;
    if (x < 0)
        sign = -1;
    x = fabs(x)/sqrt(2.0);

    // A&S formula 7.1.26
    double t = 1.0/(1.0 + p*x);
    double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);

    return 0.5*(1.0 + sign*y);
}
