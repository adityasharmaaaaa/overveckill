#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "overveckill/distance.hpp"
#include "test_framework.hpp"

using namespace overveckill;

namespace {

// Tiny deterministic generator so test data is reproducible.
std::vector<float> random_vec(std::size_t dim, std::uint32_t& state) {
    std::vector<float> v(dim);
    for (auto& x : v) {
        state = state * 1664525u + 1013904223u;
        x = static_cast<float>(state >> 8) / static_cast<float>(1u << 24) * 2.0f - 1.0f;
    }
    return v;
}

}  // namespace

OVK_TEST(l2_known_values) {
    const std::vector<float> a{0.0f, 0.0f};
    const std::vector<float> b{3.0f, 4.0f};
    OVK_CHECK_NEAR(distance(Metric::L2, a, b), 25.0f, 1e-6);  // squared!
    OVK_CHECK_NEAR(distance(Metric::L2, a, a), 0.0f, 1e-6);
}

OVK_TEST(l2_is_symmetric) {
    const std::vector<float> a{1.0f, -2.0f, 3.5f};
    const std::vector<float> b{-4.0f, 0.5f, 2.0f};
    OVK_CHECK_NEAR(distance(Metric::L2, a, b), distance(Metric::L2, b, a), 1e-6);
}

OVK_TEST(inner_product_known_values) {
    const std::vector<float> a{1.0f, 2.0f, 3.0f};
    const std::vector<float> b{4.0f, 5.0f, 6.0f};
    // dot = 4 + 10 + 18 = 32, negated
    OVK_CHECK_NEAR(distance(Metric::InnerProduct, a, b), -32.0f, 1e-5);
}

OVK_TEST(inner_product_bigger_dot_means_closer) {
    const std::vector<float> q{1.0f, 1.0f};
    const std::vector<float> near_v{5.0f, 5.0f};
    const std::vector<float> far_v{1.0f, 0.0f};
    OVK_CHECK(distance(Metric::InnerProduct, q, near_v) <
              distance(Metric::InnerProduct, q, far_v));
}

OVK_TEST(cosine_known_values) {
    const std::vector<float> a{1.0f, 0.0f};
    const std::vector<float> same_dir{5.0f, 0.0f};
    const std::vector<float> ortho{0.0f, 2.0f};
    const std::vector<float> opposite{-3.0f, 0.0f};

    OVK_CHECK_NEAR(distance(Metric::Cosine, a, same_dir), 0.0f, 1e-6);
    OVK_CHECK_NEAR(distance(Metric::Cosine, a, ortho), 1.0f, 1e-6);
    OVK_CHECK_NEAR(distance(Metric::Cosine, a, opposite), 2.0f, 1e-6);
}

OVK_TEST(cosine_ignores_magnitude) {
    const std::vector<float> a{1.0f, 2.0f, 3.0f};
    const std::vector<float> b{2.0f, 4.0f, 6.0f};
    OVK_CHECK_NEAR(distance(Metric::Cosine, a, b), 0.0f, 1e-6);
}

OVK_TEST(cosine_zero_vector_is_not_nan) {
    const std::vector<float> zero{0.0f, 0.0f, 0.0f};
    const std::vector<float> v{1.0f, 2.0f, 3.0f};
    const float d = distance(Metric::Cosine, zero, v);
    OVK_CHECK(d == d);  // NaN != NaN
    OVK_CHECK_NEAR(d, 1.0f, 1e-6);
}

OVK_TEST(dimension_mismatch_throws) {
    const std::vector<float> a{1.0f, 2.0f};
    const std::vector<float> b{1.0f, 2.0f, 3.0f};
    bool threw = false;
    try {
        (void)distance(Metric::L2, a, b);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    OVK_CHECK(threw);
}

OVK_TEST(dispatch_matches_direct_kernels) {
    const std::vector<float> a{1.0f, 2.0f, 3.0f};
    const std::vector<float> b{3.0f, 1.0f, 2.0f};
    OVK_CHECK_EQ(get_distance_fn(Metric::L2)(a.data(), b.data(), 3),
                 scalar::l2_squared(a.data(), b.data(), 3));
    OVK_CHECK_EQ(get_distance_fn(Metric::InnerProduct)(a.data(), b.data(), 3),
                 scalar::neg_inner_product(a.data(), b.data(), 3));
    OVK_CHECK_EQ(get_distance_fn(Metric::Cosine)(a.data(), b.data(), 3),
                 scalar::cosine_distance(a.data(), b.data(), 3));
}


OVK_TEST(kernels_match_double_reference) {
    std::uint32_t state = 12345;
    const std::vector<std::size_t> dims{1, 3, 7, 8, 15, 16, 17, 100, 131};

    for (std::size_t dim : dims) {
        const auto a = random_vec(dim, state);
        const auto b = random_vec(dim, state);

        double l2 = 0.0, ab = 0.0, aa = 0.0, bb = 0.0;
        for (std::size_t i = 0; i < dim; ++i) {
            const double x = a[i];
            const double y = b[i];
            l2 += (x - y) * (x - y);
            ab += x * y;
            aa += x * x;
            bb += y * y;
        }
        const double cosine = 1.0 - ab / (std::sqrt(aa) * std::sqrt(bb));

        OVK_CHECK_NEAR(distance(Metric::L2, a, b), l2, 1e-3);
        OVK_CHECK_NEAR(distance(Metric::InnerProduct, a, b), -ab, 1e-3);
        OVK_CHECK_NEAR(distance(Metric::Cosine, a, b), cosine, 1e-3);
    }
}

OVK_TEST(metric_string_roundtrip) {
    for (Metric m : {Metric::L2, Metric::InnerProduct, Metric::Cosine}) {
        const auto parsed = parse_metric(to_string(m));
        OVK_REQUIRE(parsed.has_value());
        OVK_CHECK(*parsed == m);
    }
    OVK_CHECK(!parse_metric("manhattan").has_value());
}