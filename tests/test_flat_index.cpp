#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "overveckill/flat_index.hpp"
#include "test_framework.hpp"
#include "test_util.hpp"

using namespace overveckill;

OVK_TEST(flat_empty_index_returns_nothing) {
    FlatIndex idx(3);
    const std::vector<float> q{1.0f, 2.0f, 3.0f};
    OVK_CHECK_EQ(idx.size(), std::size_t{0});
    OVK_CHECK(idx.search(q, 5).empty());
}

OVK_TEST(flat_ids_are_sequential_and_get_roundtrips) {
    FlatIndex idx(2);
    const std::vector<float> a{1.0f, 2.0f};
    const std::vector<float> b{3.0f, 4.0f};
    OVK_CHECK_EQ(idx.add(a), 0u);
    OVK_CHECK_EQ(idx.add(b), 1u);
    OVK_CHECK_EQ(idx.size(), std::size_t{2});

    const auto got = idx.get(1);
    OVK_REQUIRE(got.size() == 2);
    OVK_CHECK_NEAR(got[0], 3.0f, 1e-9);
    OVK_CHECK_NEAR(got[1], 4.0f, 1e-9);
}

OVK_TEST(flat_get_out_of_range_throws) {
    FlatIndex idx(2);
    OVK_CHECK_THROWS(idx.get(0), std::out_of_range);
}

OVK_TEST(flat_l2_nearest_neighbors) {
    FlatIndex idx(2, Metric::L2);
    const std::vector<float> p0{0.0f, 0.0f};
    const std::vector<float> p1{10.0f, 10.0f};
    const std::vector<float> p2{1.0f, 1.0f};
    const std::vector<float> p3{5.0f, 5.0f};
    idx.add(p0);
    idx.add(p1);
    idx.add(p2);
    idx.add(p3);

    const std::vector<float> q{0.9f, 0.9f};
    const auto res = idx.search(q, 2);
    OVK_REQUIRE(res.size() == 2);
    OVK_CHECK_EQ(res[0].id, 2u);
    OVK_CHECK_NEAR(res[0].distance, 0.02f, 1e-5);
    OVK_CHECK_EQ(res[1].id, 0u);
    OVK_CHECK_NEAR(res[1].distance, 1.62f, 1e-5);
}

OVK_TEST(flat_k_larger_than_size_returns_all) {
    FlatIndex idx(1);
    const std::vector<float> a{3.0f};
    const std::vector<float> b{1.0f};
    const std::vector<float> c{2.0f};
    idx.add(a);
    idx.add(b);
    idx.add(c);

    const std::vector<float> q{0.0f};
    const auto res = idx.search(q, 10);
    OVK_REQUIRE(res.size() == 3);
    OVK_CHECK_EQ(res[0].id, 1u);
    OVK_CHECK_EQ(res[1].id, 2u);
    OVK_CHECK_EQ(res[2].id, 0u);

    // A gigantic k must not blow up memory.
    OVK_CHECK_EQ(idx.search(q, std::numeric_limits<std::size_t>::max()).size(),
                 std::size_t{3});
}

OVK_TEST(flat_k_zero_returns_nothing) {
    FlatIndex idx(1);
    const std::vector<float> a{1.0f};
    idx.add(a);
    OVK_CHECK(idx.search(a, 0).empty());
}

OVK_TEST(flat_inner_product_ranking) {
    FlatIndex idx(2, Metric::InnerProduct);
    const std::vector<float> p0{1.0f, 0.0f};
    const std::vector<float> p1{0.0f, 1.0f};
    const std::vector<float> p2{2.0f, 2.0f};
    idx.add(p0);
    idx.add(p1);
    idx.add(p2);

    const std::vector<float> q{1.0f, 1.0f};
    const auto res = idx.search(q, 3);
    OVK_REQUIRE(res.size() == 3);
    OVK_CHECK_EQ(res[0].id, 2u);  // dot = 4
    OVK_CHECK_EQ(res[1].id, 0u);  // dot = 1 (tie, smaller id first)
    OVK_CHECK_EQ(res[2].id, 1u);  // dot = 1
}

OVK_TEST(flat_cosine_ranking_ignores_magnitude) {
    FlatIndex idx(2, Metric::Cosine);
    const std::vector<float> p0{100.0f, 0.0f};
    const std::vector<float> p1{1.0f, 1.0f};
    const std::vector<float> p2{0.0f, -1.0f};
    idx.add(p0);
    idx.add(p1);
    idx.add(p2);

    const std::vector<float> q{2.0f, 0.0f};
    const auto res = idx.search(q, 3);
    OVK_REQUIRE(res.size() == 3);
    OVK_CHECK_EQ(res[0].id, 0u);
    OVK_CHECK_NEAR(res[0].distance, 0.0f, 1e-6);
    OVK_CHECK_EQ(res[1].id, 1u);
    OVK_CHECK_NEAR(res[1].distance, 0.29289322f, 1e-5);
    OVK_CHECK_EQ(res[2].id, 2u);
    OVK_CHECK_NEAR(res[2].distance, 1.0f, 1e-6);
}

OVK_TEST(flat_dimension_mismatch_throws) {
    FlatIndex idx(3);
    const std::vector<float> two{1.0f, 2.0f};
    OVK_CHECK_THROWS(idx.add(two), std::invalid_argument);
    OVK_CHECK_THROWS(idx.search(two, 1), std::invalid_argument);
}

OVK_TEST(flat_rejects_non_finite_values) {
    FlatIndex idx(2);
    const std::vector<float> bad_nan{1.0f, std::numeric_limits<float>::quiet_NaN()};
    const std::vector<float> bad_inf{std::numeric_limits<float>::infinity(), 0.0f};
    OVK_CHECK_THROWS(idx.add(bad_nan), std::invalid_argument);
    OVK_CHECK_THROWS(idx.add(bad_inf), std::invalid_argument);
    OVK_CHECK_EQ(idx.size(), std::size_t{0});  // failed adds change nothing

    const std::vector<float> ok{1.0f, 2.0f};
    idx.add(ok);
    OVK_CHECK_THROWS(idx.search(bad_nan, 1), std::invalid_argument);
}

OVK_TEST(flat_zero_dim_throws) {
    OVK_CHECK_THROWS(FlatIndex(0), std::invalid_argument);
}

OVK_TEST(flat_matches_reference_for_all_metrics) {
    constexpr std::size_t dim = 17;  // deliberately not a power of two
    constexpr std::size_t n = 200;
    constexpr std::size_t k = 10;

    for (Metric m : {Metric::L2, Metric::InnerProduct, Metric::Cosine}) {
        std::uint32_t state = 777;
        FlatIndex idx(dim, m);
        for (std::size_t i = 0; i < n; ++i) idx.add(ovk_test::random_vec(dim, state));

        for (int qn = 0; qn < 10; ++qn) {
            const auto q = ovk_test::random_vec(dim, state);

            // Reference: score everything with the safe wrapper, full sort.
            std::vector<Neighbor> ref;
            for (std::size_t i = 0; i < n; ++i) {
                const Id id = static_cast<Id>(i);
                ref.push_back({id, distance(m, q, idx.get(id))});
            }
            std::sort(ref.begin(), ref.end());
            ref.resize(k);

            OVK_CHECK(idx.search(q, k) == ref);
        }
    }
}

OVK_TEST(flat_self_query_finds_itself) {
    constexpr std::size_t count = 50;
    std::uint32_t state = 99;
    FlatIndex idx(8);
    std::vector<std::vector<float>> vs;
    for (std::size_t i = 0; i < count; ++i) {
        vs.push_back(ovk_test::random_vec(8, state));
        idx.add(vs.back());
    }
    for (std::size_t i = 0; i < count; ++i) {
        const auto res = idx.search(vs[i], 1);
        OVK_REQUIRE(res.size() == 1);
        OVK_CHECK_EQ(res[0].id, static_cast<Id>(i));
        OVK_CHECK_NEAR(res[0].distance, 0.0f, 1e-9);
    }
}