#include "vector.hpp"
#include <iostream>

int main() {
    Vector a({1, 2, 3});
    Vector b({4, 5, 6});

    std::cout << "a + b = " << (a + b) << '\n';
    std::cout << "a · b = " << a.dot(b) << '\n';
    std::cout << "|a| = " << a.magnitude() << '\n';
    std::cout << "cosine similarity = " << a.cosine_similarity(b) << '\n';
}
