#pragma once

#include "tensor.hpp"

namespace Convolution {

Linalg::Tensor<4> conv_naive(const Linalg::Tensor<4>& input,
                             const Linalg::Tensor<4>& kernel);

Linalg::Tensor<4> conv_im2col(const Linalg::Tensor<4>& input,
                              const Linalg::Tensor<4>& kernel);
}  // namespace Operation