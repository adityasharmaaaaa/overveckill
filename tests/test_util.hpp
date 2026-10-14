#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ovk_test {

// Deterministic pseudo-random vector with components in [-1, 1).
// `state` is advanced, so successive calls give different vectors.
inline std::vector<float> random_vec(std::size_t dim, std::uint32_t& state) {
    std::vector<float> v(dim);
    for (auto& x : v) {
        state = state * 1664525u + 1013904223u;
        x = static_cast<float>(state >> 8) / static_cast<float>(1u << 24) * 2.0f - 1.0f;
    }
    return v;
}

}  // namespace ovk_test