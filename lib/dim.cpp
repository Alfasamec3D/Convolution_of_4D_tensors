#include "dim.hpp"

bool Dim::approxEql(const float& object1, const float& object2,
                    const float& tolerance) {
  return std::abs(object1 - object2) < tolerance;
}

bool Dim::approxEql(const std::vector<float>& object1,
                    const std::vector<float>& object2, const float& tolerance) {
  if (object1.size() != object2.size()) return false;

  for (size_t i = 0; i < object1.size(); ++i)
    if (!Dim::approxEql(object1.data()[i], object2.data()[i], tolerance))
      return false;
  return true;
}
