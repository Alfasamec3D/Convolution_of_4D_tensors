#include "gemm.hpp"

Linalg::Tensor<2> Gemm::gemm_naive(const Linalg::Tensor<2>& tensor1,
                             const Linalg::Tensor<2>& tensor2) {
  if (tensor1.dimension(1) != tensor2.dimension(0))
    throw std::invalid_argument("Inpossible to multiplicate matrixes");

  // Dimensions of result matrix
  std::array<size_t, 2> res_dim = {tensor1.dimension(0), tensor2.dimension(1)};

  std::vector<float> result(Dim::length(res_dim));

  for (size_t i = 0; i < tensor1.dimension(0); ++i) {
    for (size_t j = 0; j < tensor2.dimension(1); ++j) {
      float sum = 0;
      for (size_t k = 0; k < tensor1.dimension(1); ++k)
        sum += tensor1.element({i, k}) * tensor2.element({k, j});

      result[Dim::index(res_dim, {i, j})] = sum;
    }
  }
  return {res_dim, result};
}

