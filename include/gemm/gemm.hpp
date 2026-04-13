#pragma once
#include "tensor.hpp"
namespace Gemm {
Linalg::Tensor<2> gemm_naive(const Linalg::Tensor<2>& tensor1,
                             const Linalg::Tensor<2>& tensor2);
}  // namespace Gemm