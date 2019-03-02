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
#include "helper_functions.cuh"

/*
double normpdf(const double x, const double mu, const double sigma){
	double power = (x-mu)/sigma;
	return exp(-power*power/2)*INV_SQRT_2PI/sigma;
}
*/

__global__
void
gen_normpdf(int n, const double base, const double step, double *vec_x)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) {
    double current_pos = base + i * step;
    vec_x[i] = normpdf(current_pos,1.0);
  }
}

__host__ __device__
double
normpdf(const double x, const double sigma){
    // mu = 0
	double power = x/sigma;
	return exp(-power*power/2)*INV_SQRT_2PI/sigma;
}

__global__
void
gen_normpdf(int n, const float base, const float step, float *vec_x)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) {
    float current_pos = base + i * step;
    vec_x[i] = normpdf(current_pos,1.0);
  }
}

__host__ __device__
float
normpdf(const float x, const float sigma){
    // mu = 0
	float power = x/sigma;
	return exp(-power*power/2)*INV_SQRT_2PI/sigma;
}