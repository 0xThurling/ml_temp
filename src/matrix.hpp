#pragma once

#include "vector.hpp"
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
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

  Matrix operator*(const value_type &scalar) const {
    std::vector<std::vector<value_type>> result(
        rows(), std::vector<value_type>(cols(), value_type{0}));

    for (std::size_t i = 0; i < rows(); ++i) {
      for (std::size_t j = 0; j < cols(); ++j) {
        result[i][j] += _rows[i][j] * scalar;
      }
    }

    return Matrix(std::move(result));
  }

  Matrix operator-(const Matrix &other) const {
    if (rows() * cols() != other.rows() * other.cols()) {
      throw std::runtime_error("matrix/matrix dimension mismatch");
    }

    std::vector<std::vector<value_type>> result(
        rows(), std::vector<value_type>(cols(), value_type{0}));

    for (std::size_t i = 0; i < rows(); ++i) {
      for (std::size_t j = 0; j < cols(); ++j) {
        result[i][j] = _rows[i][j] - other._rows[i][j];
      }
    }

    return Matrix(std::move(result));
  }

  Matrix operator+(const Matrix &other) const {
    if (rows() * cols() != other.rows() * other.cols()) {
      throw std::runtime_error("matrix/matrix dimension mismatch");
    }

    std::vector<std::vector<value_type>> result(
        rows(), std::vector<value_type>(cols(), value_type{0}));

    for (std::size_t i = 0; i < rows(); ++i) {
      for (std::size_t j = 0; j < cols(); ++j) {
        result[i][j] = _rows[i][j] + other._rows[i][j];
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

  value_type determinant() const {
    if (rows() != cols()) {
      throw std::runtime_error(
          "determinant is defined only for square matrices");
    }

    if (rows() == 1) {
      return _rows[0][0];
    }

    if (rows() == 2) {
      return _rows[0][0] * _rows[1][1] - _rows[0][1] * _rows[1][0];
    }

    value_type det = 0;
    for (std::size_t j = 0; j < cols(); ++j) {
      std::vector<std::vector<value_type>> minor_rows;

      for (std::size_t i = 1; i < rows(); ++i) {
        std::vector<value_type> minor_row;

        for (std::size_t k = 0; k < cols(); ++k) {
          if (k != j) {
            minor_row.push_back(_rows[i][k]);
          }
        }

        minor_rows.push_back(std::move(minor_row));
      }

      Matrix minor(std::move(minor_rows));
      value_type sign = (j % 2 == 0) ? 1.0 : -1.0;
      det += sign * _rows[0][j] * minor.determinant();
    }

    return det;
  }

  Matrix inverse_2x2() const {
    if (rows() != 2 || cols() != 2) {
      throw std::runtime_error("inverse_2x2 requires a 2x2 matrix");
    }

    value_type det = determinant();

    if (std::abs(det) < 1e-12) {
      throw std::runtime_error("matrix is singular, no inverse exists");
    }

    return Matrix({{_rows[1][1] / det, -_rows[0][1] / det},
                   {_rows[1][0] / det, -_rows[0][0] / det}});
  }

  Matrix eigenvalues_2x2() const {
    if (rows() != 2) {
      throw std::runtime_error("eigenvalues_2x2 requires a 2x2 matrix");
    }

    value_type a = _rows[0][0];
    value_type b = _rows[0][1];
    value_type c = _rows[1][0];
    value_type d = _rows[1][1];

    auto trace = a + b;
    auto det = a * d - b * c;
    auto discriminant = (trace * trace) - 4 * det;

    if (discriminant < 0) {
      auto real = trace / 2;
      auto imag = std::sqrt(-discriminant) / 2;
      return Matrix({{real, imag}, {real, -imag}});
    } else {
      const double s = std::sqrt(discriminant);
      return Matrix({{(trace + s) / 2.0, 0.0}, {(trace - s) / 2.0, 0.0}});
    }
  }

  Vector eigenvector_2x2(std::complex<value_type> lambda) const {
    if (rows() != 2) {
      throw std::runtime_error("eigenvalues_2x2 requires a 2x2 matrix");
    }

    value_type a = _rows[0][0];
    value_type b = _rows[0][1];
    value_type c = _rows[1][0];
    value_type d = _rows[1][1];

    double v0, v1;

    if (std::abs(b) > 1e-10) {
      v0 = b;
      v1 = (lambda - a).real();
    } else if (std::abs(c) > 1e-10) {
      v0 = (lambda - d).real();
      v1 = c;
    } else {
      if (std::abs(a - lambda.real()) < 1e-10) {
        v0 = 1, v1 = 0;
      } else {
        v0 = 0, v1 = 1;
      }
    }

    const value_type mag = std::sqrt(v0 * v0 + v1 * v1);
    return Vector({v0 / mag, v1 / mag});
  }

  static Matrix rotation_2d(const value_type &theta) {
    value_type c = std::cos(theta);
    value_type s = std::sin(theta);

    return Matrix({{c, -s}, {s, -c}});
  }

  static Matrix scaling_2d(const value_type &sx, const value_type &sy) {
    return Matrix({{sx, 0}, {0, sy}});
  }

  static Matrix shearing_2d(const value_type &kx, const value_type &ky) {
    return Matrix({{1, kx}, {ky, 1}});
  }

  static Matrix reflection_x() { return Matrix({{1, 0}, {0, -1}}); }

  static Matrix reflection_y() { return Matrix({{-1, 0}, {0, 1}}); }

  static Matrix identity(std::size_t n) {
    std::vector<std::vector<value_type>> rows(
        n, std::vector<value_type>(n, value_type{0}));

    for (std::size_t i = 0; i < n; ++i) {
      rows[i][i] = value_type{1};
    }

    return Matrix(std::move(rows));
  }

  const std::vector<std::vector<value_type>> &data() const noexcept {
    return _rows;
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
