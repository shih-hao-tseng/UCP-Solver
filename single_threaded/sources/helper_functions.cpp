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
#include "helper_functions.h"

/*
double normpdf(const double x, const double mu, const double sigma){
	double power = (x-mu)/sigma;
	return exp(-power*power/2)*INV_SQRT_2PI/sigma;
}
*/
double
normpdf(const double x, const double sigma){
    // mu = 0
	double power = x/sigma;
	return exp(-power*power/2)*INV_SQRT_2PI/sigma;
}