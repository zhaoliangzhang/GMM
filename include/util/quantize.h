
namespace gmm{

template <typename Dtype>
Dtype fix_data(const Dtype x, const Dtype step,
    const Dtype lb, const Dtype ub);

// p is position of the dot
template <typename Dtype>
void cpu_fix(const int n, const Dtype* x, Dtype* y,
    const int bit_width, const int p);

template <typename Dtype>
Dtype cpu_fix_pos_overflow(const int n, const Dtype* x, const int bit_width);

}