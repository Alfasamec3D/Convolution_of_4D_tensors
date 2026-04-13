#include "convolution.hpp"

#include <array>
#include <stdexcept>

#include "dim.hpp"
#include "gemm.hpp"

using namespace Dim;

Linalg::Tensor<4> Convolution::conv_naive(const Linalg::Tensor<4>& input,
                                          const Linalg::Tensor<4>& kernel) {
  if (kernel.dimension(0) != kernel.dimension(3))
    throw std::logic_error("Kernel's height must be equal to kernel's width");
  if (kernel.dimension(2) > input.dimension(2))
    throw std::logic_error("Kernel's heigt is bigger than input's");
  if (kernel.dimension(3) > input.dimension(3))
    throw std::logic_error("Kernel's width is bigger than input's");
  if (kernel.dimension(0) != kernel.dimension(1))
    throw std::logic_error(
        "Kernel's quantity of input channels must be equal to output channels");
  if (input.dimension(1) != kernel.dimension(1))
    throw std::logic_error(
        "Input's quantity of input channels must be equal to kernel's "
        "channels");

  const size_t N = input.dimension(0);
  const size_t C = input.dimension(1);
  const size_t H = input.dimension(2);
  const size_t W = input.dimension(3);
  const size_t K = kernel.dimension(2);

  const size_t H_out = H - K + 1;
  const size_t W_out = W - K + 1;

  std::array<size_t, 4> out_dim = {N, C, H_out, W_out};
  const size_t output_length = length(out_dim);
  std::vector<float> output(output_length);

  for (size_t n = 0; n < N; ++n)
    for (size_t m = 0; m < C; ++m)
      for (size_t y = 0; y < H_out; ++y)
        for (size_t x = 0; x < W_out; ++x) {
          float sum = 0;
          for (size_t c = 0; c < C; ++c)
            for (size_t ky = 0; ky < K; ++ky)
              for (size_t kx = 0; kx < K; ++kx)
                sum += input.element({n, c, y + ky, x + kx}) *
                       kernel.element({m, c, ky, kx});

          output[index(out_dim, {n, m, y, x})] = sum;
        }
  return {out_dim, output};
}

Linalg::Tensor<4> Convolution::conv_im2col(const Linalg::Tensor<4>& input,
                                           const Linalg::Tensor<4>& kernel) {
  if (kernel.dimension(0) != kernel.dimension(3))
    throw std::logic_error("Kernel's height must be equal to kernel's width");
  if (kernel.dimension(2) > input.dimension(2))
    throw std::logic_error("Kernel's heigt is bigger than input's");
  if (kernel.dimension(3) > input.dimension(3))
    throw std::logic_error("Kernel's width is bigger than input's");
  if (kernel.dimension(0) != kernel.dimension(1))
    throw std::logic_error(
        "Kernel's quantity of input channels must be equal to output channels");
  if (input.dimension(1) != kernel.dimension(1))
    throw std::logic_error(
        "Input's quantity of input channels must be equal to kernel's "
        "channels");

  const size_t N = input.dimension(0);
  const size_t C = input.dimension(1);
  const size_t H = input.dimension(2);
  const size_t W = input.dimension(3);
  const size_t K = kernel.dimension(2);

  const size_t H_out = H - K + 1;
  const size_t W_out = W - K + 1;

  std::array<size_t, 4> out_dim = {N, C, H_out, W_out};
  const size_t output_length = length(out_dim);
  std::vector<float> output(output_length);

  size_t patch_size = C * K * K;
  size_t num_patches = H_out * W_out;
  std::array<size_t, 2> kernel_matrix_dim{C, patch_size};
  std::array<size_t, 2> im_matrix_dim{patch_size, num_patches};

  for (size_t n = 0; n < N; ++n) {
    std::vector<float> im_matrix_vector(length(im_matrix_dim));

    for (size_t c = 0; c < C; ++c)
      for (size_t ky = 0; ky < K; ++ky)
        for (size_t kx = 0; kx < K; ++kx) {
          size_t row = (c * K + ky) * K + kx;
          for (size_t y = 0; y < H_out; ++y)
            for (size_t x = 0; x < W_out; ++x) {
              size_t col = y * W_out + x;
              im_matrix_vector[index(im_matrix_dim, {row, col})] =
                  input.element({n, c, y + ky, x + kx});
            }
        }

    Linalg::Tensor kernel_matrix{kernel_matrix_dim, kernel.data()};
    Linalg::Tensor im_matrix{im_matrix_dim, im_matrix_vector};

    Linalg::Tensor result_mat = Gemm::gemm_naive(kernel_matrix, im_matrix);

    for (size_t c = 0; c < C; ++c)
      for (size_t col = 0; col < num_patches; ++col) {
        size_t y = col / W_out;
        size_t x = col % W_out;
        output[index(out_dim, {n, c, y, x})] = result_mat.element({c, col});
      }
  }
  return {out_dim, output};
}
