#include "math/calc.hpp"
#include <iomanip>
#include <iostream>

int main() {
  double xs[] = {-2.0, -1.0, 0.0, 1.0, 2.0};

  std::cout << std::fixed << std::setprecision(6);
  for (double x : xs) {
    double numerical = forgeml::numerical_derivative(forgeml::f, x);
    double analytical = 2.0 * x;

    std::cout << "x=" << std::setw(2) << static_cast<int>(x)
              << "  f'(x) numerical=" << numerical
              << "  analytical=" << std::setprecision(1) << analytical
              << std::setprecision(6) << "\n";
  }

  return 0;
}
