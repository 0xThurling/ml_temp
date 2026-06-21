#include "vector.hpp"
#include "matrix.hpp"
#include <iostream>

int main() {
    Matrix rotation_90({{0, -1}, {1, 0}});
    Vector point({3, 1});

    Vector rotated = rotation_90 * point;

    std::cout << "Original: " << point << '\n';
    std::cout << "Rotated 90°: " << rotated << '\n';
}
