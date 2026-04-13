#include "tensor.hpp"

#include <gtest/gtest.h>
using namespace Linalg;

TEST(ElementTest, BasicCase) {
  std::array<size_t, 4> dimensions = {1, 1, 1, 1};
  std::vector<float> data{2.3};
  Tensor tensor{dimensions, data};

  std::array<size_t, 4> indexes = {0, 0, 0, 0};

  float real_element = tensor.element(indexes);
  float expected_element = 2.3;
  EXPECT_TRUE(Dim::approxEql(real_element, expected_element));
}

TEST(DataTest, BasicCase) {
  std::array<size_t, 4> dimensions = {1, 2, 1, 2};
  std::vector<float> data = {5.0002, 5.00003, 8.004, 0};
  Tensor tensor{dimensions, data};

  std::vector<float> real_object = tensor.data();
  EXPECT_TRUE(Dim::approxEql(real_object, data));
}

TEST(DimensionTest, BasicCase) {
  std::array<size_t, 4> dimensions = {1, 2, 1, 2};
  std::vector<float> data = {5.0002, 5.00003, 8.004, 0};
  Tensor tensor{dimensions, data};

  const size_t real_object = tensor.dimension(2);
  const size_t expected_object = 1;
  EXPECT_TRUE(Dim::approxEql(real_object, expected_object, 0.000001));
}

TEST(ApproxEqlTensorTest, BasicCase) {
  std::array<size_t, 4> dimensions = {1, 2, 1, 2};
  std::vector<float> data = {5.0002, 5.00003, 8.004, 0};
  Tensor tensor1{dimensions, data};
  Tensor tensor2{dimensions, data};

  EXPECT_TRUE(approxEql(tensor1, tensor2, 0.000001));
}