#pragma once

#include <cassert>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "dim.hpp"

namespace Linalg {

template <size_t T>
class Tensor {
  std::array<size_t, T> dimensions_;
  std::vector<float> data_;

 public:
  Tensor(const std::array<size_t, T>& dimensions,
         const std::vector<float>& data)
      : dimensions_(dimensions), data_(data) {
    if (data.size() != Dim::length(dimensions))
      throw std::invalid_argument(
          "Not enough elements in container for tensor");
  }

  float element(const std::array<size_t, T>& indexes) const {
    return data_[Dim::index(dimensions_, indexes)];
  }

  std::vector<float> data() const { return data_; }

  size_t dimension(const size_t& index) const {
    if (index >= T) {
      std::stringstream ss;
      ss << "number of dimension: " << index << std::endl
         << " is bigger than total number of dimensions " << T;
      throw std::logic_error(ss.str());
    }
    return dimensions_[index];
  }
};

template <size_t T, size_t U>
bool approxEql(const Tensor<T>& object1, const Tensor<U>& object2,
               const float& tolerance = Dim::global_tolerance) {
  if (T != U) return false;
  for (size_t i = 0; i < T; ++i)
    if (object1.dimension(i) != object2.dimension(i)) return false;

  return Dim::approxEql(object1.data(), object2.data(), tolerance);
}
}  // namespace Linalg