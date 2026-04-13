#include "convolution.hpp"

#include <gtest/gtest.h>
using namespace Convolution;

TEST(ConvNaiveTest, BasicCase) {
  size_t N = 2;
  size_t C = 2;
  size_t H = 3;
  size_t W = 3;
  size_t K = 3;
  size_t H_out = 1;
  size_t W_out = 1;
  std::array input_dim = {N, C, H, W};
  std::array kernel_dim = {C, C, K, K};
  std::array expected_dim = {N, C, H_out, W_out};

  std::vector<float> input_data(Dim::length(input_dim), 0);
  std::vector<float> kernel_data(Dim::length(kernel_dim), 0);
  std::vector<float> expected_data(Dim::length(expected_dim), 0);

  Linalg::Tensor input{input_dim, input_data};
  Linalg::Tensor kernel{kernel_dim, kernel_data};
  Linalg::Tensor expected_object{expected_dim, expected_data};

  Linalg::Tensor real_object = conv_naive(input, kernel);

  EXPECT_TRUE(approxEql(real_object, expected_object, 0.1));
}

TEST(ConvIm2ColTest, BasicCase) {
  size_t N = 2;
  size_t C = 2;
  size_t H = 3;
  size_t W = 3;
  size_t K = 3;
  size_t H_out = 1;
  size_t W_out = 1;
  std::array input_dim = {N, C, H, W};
  std::array kernel_dim = {C, C, K, K};
  std::array expected_dim = {N, C, H_out, W_out};

  std::vector<float> input_data(Dim::length(input_dim), 0);
  std::vector<float> kernel_data(Dim::length(kernel_dim), 0);
  std::vector<float> expected_data(Dim::length(expected_dim), 0);

  Linalg::Tensor input{input_dim, input_data};
  Linalg::Tensor kernel{kernel_dim, kernel_data};
  Linalg::Tensor expected_object{expected_dim, expected_data};

  Linalg::Tensor real_object = conv_im2col(input, kernel);

  EXPECT_TRUE(approxEql(real_object, expected_object));
}