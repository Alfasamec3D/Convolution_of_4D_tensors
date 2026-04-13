#pragma once
#include <array>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace Dim {

const float global_tolerance = 0.1;
bool approxEql(const float& object1, const float& object2,
               const float& tolerance = global_tolerance);

bool approxEql(const std::vector<float>& object1,
               const std::vector<float>& object2,
               const float& tolerance = global_tolerance);

template <size_t T>
size_t index(const std::array<size_t, T>& dimensions,
             const std::array<size_t, T>& indexes) {
  for (size_t i = 0; i < T; ++i) {
    if (indexes[i] >= dimensions[i]) {
      std::stringstream ss;
      ss << "index number " << i << ": " << indexes[i] << std::endl
         << " is bigger than dimension number " << i << ": " << dimensions[i];
      throw std::logic_error(ss.str());
    };
  }
  size_t final_index = 0;
  for (size_t i = 0; i < T; ++i)
    final_index = final_index * dimensions[i] + indexes[i];
  return final_index;
}

template <size_t T>
size_t length(const std::array<size_t, T>& dimensions) {
  size_t elements_number = 1;

  for (size_t dimension : dimensions) elements_number *= dimension;
  return elements_number;
}

}  // namespace Dim