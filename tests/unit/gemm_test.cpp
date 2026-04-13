#include "gemm.hpp"

#include <gtest/gtest.h>
using namespace Gemm;

TEST(GemmNaiveTest, BasicCase) {
  size_t rows = 3;
  size_t roco = 5;
  size_t cols = 3;
  std::array dimensions1 = {rows, roco};
  std::array dimensions2 = {roco, cols};
  std::array dimensions = {rows, cols};
  std::vector<float> data1{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> data2{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> data{0, 0, 0, 0, 0, 0, 0, 0, 0};
  Linalg::Tensor tensor1{dimensions1, data1};
  Linalg::Tensor tensor2{dimensions2, data2};
  Linalg::Tensor expected_object{dimensions, data};

  Linalg::Tensor real_object = gemm_naive(tensor1, tensor2);

  EXPECT_TRUE(approxEql(real_object, expected_object, 0.1));
}