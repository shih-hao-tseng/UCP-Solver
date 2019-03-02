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
#ifndef HELPER_FUNCTIONS_CUH
#define HELPER_FUNCTIONS_CUH

#include <math.h>

#define PI            3.141592653589793238462643383279502884 // 197169399
#define INV_SQRT_2PI  0.398942280401433
#define TWO_SQRT_TWO  2.828427124746190

__global__
void gen_normpdf(int n, const double base, const double step, double *vec_x);

__host__ __device__
double normpdf(const double x, const double sigma);

__global__
void gen_normpdf(int n, const float base, const float step, float *vec_x);

__host__ __device__
float normpdf(const float x, const float sigma);

#endif // HELPER_FUNCTIONS_CUH