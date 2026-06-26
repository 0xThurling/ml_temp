#pragma once

#include <cstddef>
#include <functional>
#include <vector>
namespace forgeml {
double numerical_derivative(double (*f)(double), double x, double h = 1e-7) {
  return (f(x + h) - f(x - h)) / (2.0 * h);
}

double f(double x) { return x * x; }

std::vector<double>
numerical_gradient(const std::function<double(const std::vector<double> &)> f,
                   std::vector<double> &points, double h = 1e-7) {

  std::vector<double> gradient(points.size());
  std::vector<double> points_plus = points;
  std::vector<double> points_minus = points;

  for (std::size_t i = 0; i < points.size(); ++i) {
    points_plus[i] += h;
    points_minus[i] += h;

    const double f_plus = f(points_plus);
    const double f_minus = f(points_minus);

    gradient[i] = (f_plus - f_minus) / (2.0 * h);

    // reset for next dimension
    points_plus[i] = points[i];
    points_minus[i] = points[i];
  }

  return gradient;
}

double f_multi(const std::vector<double> &point) {
  const double x = point[0];
  const double y = point[1];

  return x * x + 3.0 * x * y + y * y;
}
} // namespace forgeml
