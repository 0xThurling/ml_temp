#pragma once

#include "vector.hpp"
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <vector>

class Matrix {
public:
  using value_type = double;

  explicit Matrix(std::vector<std::vector<value_type>> rows)
      : _rows(std::move(rows)) {
    if (_rows.empty()) {
      throw std::runtime_error("matrix cannot be empty");
    }

    const std::size_t ncols = _rows[0].size();
    if (ncols == 0) {
      throw std::runtime_error("matrix cannot have empty rows");
    }

    for (const auto &row : _rows) {
      if (row.size() != ncols) {
        throw std::runtime_error("ragged matrix");
      }
    }
  }

  std::size_t rows() const noexcept { return _rows.size(); }

  std::size_t cols() const noexcept { return _rows[0].size(); }

  Vector operator*(const Vector &v) const {
    if (cols() != v.dim()) {
      throw std::runtime_error("matrix/vector dimension mismatch");
    }

    std::vector<value_type> result(rows(), value_type{0});

    for (std::size_t i = 0; i < rows(); ++i) {
      for (std::size_t j = 0; j < cols(); ++j) {
        result[i] += _rows[i][j] * v[j];
      }
    }

    return Vector(std::move(result));
  }

  Matrix operator*(const Matrix &other) const {
    if (cols() != other.rows()) {
      throw std::runtime_error("matrix/matrix dimension mismatch");
    }

    std::vector<std::vector<value_type>> result(
        rows(), std::vector<value_type>(other.cols(), value_type{0}));

    for (std::size_t i = 0; i < rows(); ++i) {
      for (std::size_t j = 0; j < other.cols(); ++j) {
        for (std::size_t k = 0; k < cols(); ++k) {
          result[i][j] += _rows[i][k] * other._rows[k][j];
        }
      }
    }

    return Matrix(std::move(result));
  }

  Matrix transpose() const {
    std::vector<std::vector<value_type>> result(
        cols(), std::vector<value_type>(rows(), value_type{0}));

    for (std::size_t i = 0; i < rows(); ++i) {
      for (std::size_t j = 0; j < rows(); ++j) {
        result[j][i] = _rows[i][j];
      }
    }

    return Matrix(std::move(result));
  }

  friend std::ostream &operator<<(std::ostream &os, const Matrix &m) {
    os << "Matrix([";

    for (std::size_t i = 0; i < m._rows.size(); ++i) {
      os << "[";

      for (std::size_t j = 0; j < m._rows[i].size(); ++j) {
        os << m._rows[i][j];
        if (j + 1 < m._rows[i].size()) {
          os << ", ";
        }
      }

      os << "]";
      if (i + 1 < m._rows.size()) {
        os << ", ";
      }
    }

    os << "])";
    return os;
  }

private:
  std::vector<std::vector<value_type>> _rows;
};
