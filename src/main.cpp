#include "core/vector.hpp"
#include "math/linalg.hpp"
#include "utils/assertions.hpp"
#include <vector>

int main() {
    Vector v1({1, 0});
    Vector v2({0, 1});

    auto test = forgeml::project(v1, v2);

    ML_ASSERT(test.cosine_similarity(v2) == 1, "Not similar vectors");
}
