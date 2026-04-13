#include "dim.hpp"

#include <gtest/gtest.h>
using namespace Dim;

TEST(IndexTest, BasicCase) {
  std::array<size_t, 4> dimensions = {2, 3, 4, 6};
  std::array<size_t, 4> indexes1 = {1, 1, 3, 4};
  size_t real_index1 = index(dimensions, indexes1);
  size_t expected_index1 = 118;
  EXPECT_EQ(real_index1, expected_index1);

  std::array<size_t, 4> indexes2 = {0, 0, 0, 0};
  size_t real_index2 = index(dimensions, indexes2);
  size_t expected_index2 = 0;
  EXPECT_EQ(real_index2, expected_index2);
}

TEST(LengthTest, BasicCase) {
  std::array<size_t, 4> dimensions1 = {2, 3, 4, 6};
  size_t real_length1 = length(dimensions1);
  size_t expected_length1 = 144;
  EXPECT_EQ(real_length1, expected_length1);

  std::array<size_t, 4> dimensions2 = {1, 1, 1, 1};
  size_t real_length2 = length(dimensions2);
  size_t expected_length2 = 1;
  EXPECT_EQ(real_length2, expected_length2);
}

TEST(ApproxEqlNumberTest, BasicCase) {
  float a = 2.000001;
  float b = 2.000001;
  bool result = approxEql(a, b, 0.0000001);
  EXPECT_TRUE(result);
}

TEST(ApproxEqlVectorTest, BasicCase) {
  std::vector<float> vpoint1{5.0002, 5.00003, 8.004};
  std::vector<float> vpoint2{5.0002, 5.00003, 8.004};
  bool result = approxEql(vpoint1, vpoint2, 0.000001);
  EXPECT_TRUE(result);
}