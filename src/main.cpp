#include "matrix.hpp"
#include "vector.hpp"
#include <iostream>
#include <random>
#include <vector>

int main() {
  try {
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 0.1);

    Matrix weights(
        {{dist(rng), dist(rng), dist(rng)}, {dist(rng), dist(rng), dist(rng)}});

    Vector input_vector({1.0, 0.5, -0.3});

    Vector output = weights * input_vector;

    std::cout << "Input (3D): " << input_vector << '\n';
    std::cout << "Output (2D): " << output << '\n';
    std::cout << "This is what a neural network layer does -- matrix "
                 "multiplication.\n";
  } catch (const std::exception &e) {
    std::cerr << "exception: " << e.what() << '\n';
  } catch (...) {
    std::cerr << "unknown exception\n";
  }
}
