#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <ostream>
#include <stdexcept>
#include <vector>

#define M_PI 3.14159265358979323846 /* pi */

class Vector {
public:
  using value_type = double;

  explicit Vector(std::vector<double> components)
      : _components(std::move(components)) {}

  std::size_t dim() const noexcept { return _components.size(); }

  Vector operator+(const Vector &other) const {
    check_same_dim(other);
    std::vector<value_type> result;

    result.reserve(dim());

    for (std::size_t i = 0; i < dim(); ++i) {
      result.push_back(_components[i] + other._components[i]);
    }

    return Vector(std::move(result));
  }

  Vector operator-(const Vector &other) const {
    check_same_dim(other);
    std::vector<value_type> result;
    result.reserve(dim());

    for (std::size_t i = 0; i < dim(); ++i) {
      result.push_back(_components[i] - other._components[i]);
    }

    return Vector(std::move(result));
  }

  value_type operator[](std::size_t i) const { return _components[i]; }

  value_type &operator[](std::size_t i) { return _components[i]; }

  value_type dot(const Vector &other) const {
    check_same_dim(other);
    return std::inner_product(_components.begin(), _components.end(),
                              other._components.begin(), value_type{0});
  }

  value_type magnitude() const {
    value_type sum_sq = 0;
    for (auto x : _components) {
      sum_sq += x * x;
    }

    return std::sqrt(sum_sq);
  }

  value_type angle_degrees(const Vector &b) {
    const double na = magnitude();
    const double nb = b.magnitude();

    const double denom = na * nb;
    double c = dot(b) / denom;
    c = std::max(-1.0, std::min(1.0, c));
    double radians = std::acos(c);
    return radians * (180 / M_PI);
  }

  Vector normalize() const {
    auto mag = magnitude();
    if (mag == value_type{0}) {
      throw std::runtime_error("cannot normalise zero vector");
    }

    std::vector<value_type> result;
    result.reserve(dim());
    for (auto x : _components) {
      result.push_back(x / mag);
    }
    return Vector(std::move(result));
  }

  const std::vector<value_type> &components() const noexcept {
    return _components;
  }

  value_type cosine_similarity(const Vector &other) const {
    auto denom = magnitude() * other.magnitude();

    if (denom == value_type{0}) {
      throw std::runtime_error("cosine similarity undefined for zero vector");
    }

    return dot(other) / denom;
  }

  friend std::ostream &operator<<(std::ostream &os, const Vector &v) {
    os << "Vector([";
    for (std::size_t i = 0; i < v.dim(); ++i) {
      os << v._components[i];
      if (i + 1 < v.dim())
        os << ", ";
    }
    os << "])";
    return os;
  }

private:
  std::vector<value_type> _components;

  void check_same_dim(const Vector &other) const {
    if (dim() != other.dim()) {
      throw std::runtime_error("dimension mismatch");
    }
  }
};
