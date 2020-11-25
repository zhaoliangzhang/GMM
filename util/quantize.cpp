#include <cmath>
#include <algorithm>
#include <iostream>

#include "util/quantize.h"

namespace gmm{

template <typename Dtype>
Dtype fix_data(const Dtype x, const Dtype step,
    const Dtype lb, const Dtype ub) {
  return std::fmin(std::fmax(std::round(x/step)*step, lb), ub);
}

template float fix_data<float>(const float x, const float step,
    const float lb, const float ub);
template double fix_data<double>(const double x, const double step,
    const double lb, const double ub);

template <typename Dtype>
void cpu_fix(const int n, const Dtype* x, Dtype* y,
    const int bit_width, const int p) {
  Dtype step = std::pow(Dtype(2), -p);
  //std::cout<<step<<std::endl;
  Dtype lower_bound = -std::pow(Dtype(2), bit_width-1)*step;
  Dtype upper_bound = std::pow(Dtype(2), bit_width-1)*step - step;
  //std::cout<<lower_bound<<' '<<upper_bound<<std::endl;
  for (auto i = 0; i < n; ++i) {
    y[i] = fix_data(x[i], step, lower_bound, upper_bound);
  }
}

template void cpu_fix<float>(const int n, const float* x, float* y,
    const int bit_width, const int p);
template void cpu_fix<double>(const int n, const double* x, double* y,
    const int bit_width, const int p);

template <typename Dtype>
Dtype cpu_fix_pos_overflow(const int n, const Dtype *x,
                                 const int bit_width) {
  // Use half of step as a guard
  Dtype fix_lb = -std::pow(2, bit_width - 1) - 0.5;
  Dtype fix_ub = std::pow(2, bit_width - 1) - 0.5;

  Dtype x_min, x_max;
  x_min = (*std::min_element(x, x + n));
  x_max = (*std::max_element(x, x + n));

  Dtype step = std::max(x_min / fix_lb, x_max / fix_ub);
  if (step == 0)
    return 0;
  return std::log2(1 / step);
}

template float cpu_fix_pos_overflow<float>(const int n, const float *x,
                                                 const int bit_width);
template double cpu_fix_pos_overflow<double>(const int n, const double *x,
                                                   const int bit_width);

}