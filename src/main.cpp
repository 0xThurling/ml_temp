#include <vector>
#include <cmath>
#include <stdexcept>
#include "vector.hpp"
#include "matrix.hpp"

bool is_linearly_independent(const std::vector<Vector>& vectors) {
    if (vectors.empty()) {
        return true;
    }

    const std::size_t n = vectors.size();
    const std::size_t dim = vectors[0].dim();

    for (const auto& v : vectors) {
        if (v.dim() != dim) {
            throw std::runtime_error("dimension mismatch in vector list");
        }
    }

    std::vector<std::vector<double>> mat_rows;
    mat_rows.reserve(n);
    for (const auto& v : vectors) {
        mat_rows.push_back(v.components());
    }

    Matrix mat(std::move(mat_rows));
    std::vector<std::vector<double>> rows = mat.data();

    std::size_t rank = 0;

    for (std::size_t col = 0; col < dim; ++col) {
        std::size_t pivot = rows.size();
        for (std::size_t row = rank; row < rows.size(); ++row) {
            if (std::abs(rows[row][col]) > 1e-10) {
                pivot = row;
                break;
            }
        }

        if (pivot == rows.size()) {
            continue;
        }

        std::swap(rows[rank], rows[pivot]);

        double scale = rows[rank][col];
        for (double& x : rows[rank]) {
            x /= scale;
        }

        for (std::size_t row = 0; row < rows.size(); ++row) {
            if (row != rank && std::abs(rows[row][col]) > 1e-10) {
                double factor = rows[row][col];
                for (std::size_t j = 0; j < dim; ++j) {
                    rows[row][j] -= factor * rows[rank][j];
                }
            }
        }

        ++rank;
    }

    return rank == n;
}

Vector project(const Vector& a, const Vector& b) {
    double denom = b.dot(b);
    if (std::abs(denom) < 1e-10) {
        throw std::runtime_error("cannot project onto near-zero vector");
    }

    double scalar = a.dot(b) / denom;
    std::vector<double> result(b.dim());
    for (std::size_t i = 0; i < b.dim(); ++i) {
        result[i] = scalar * b[i];
    }
    return Vector(std::move(result));
}

std::vector<Vector> gram_schmidt(const std::vector<Vector>& vectors) {
    std::vector<Vector> orthonormal;

    for (const auto& v : vectors) {
        Vector w = v;

        for (const auto& u : orthonormal) {
            Vector proj = project(w, u);
            w = w - proj;
        }

        if (w.magnitude() < 1e-10) {
            continue;
        }

        orthonormal.push_back(w.normalize());
    }

    return orthonormal;
}

int main() {
    Vector v1({1, 0, 0});
    Vector v2({0, 1, 0});

    std::cout << "angle = " << v1.angle_degrees(v2) << " degrees\n";
    return 0;
}
