#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "overveckill/topk.hpp"
#include "test_framework.hpp"

using namespace overveckill;

OVK_TEST(topk_keeps_k_smallest) {
    TopK top(3);
    Id id = 0;
    for (float d : {5.0f, 1.0f, 4.0f, 2.0f, 3.0f, 0.5f}) {
        top.push({id++, d});
    }
    const auto out = top.take_sorted();
    OVK_REQUIRE(out.size() == 3);
    OVK_CHECK_EQ(out[0].id, 5u);  // 0.5
    OVK_CHECK_EQ(out[1].id, 1u);  // 1.0
    OVK_CHECK_EQ(out[2].id, 3u);  // 2.0
}

OVK_TEST(topk_fewer_than_k_items) {
    TopK top(5);
    top.push({0, 2.0f});
    top.push({1, 1.0f});
    OVK_CHECK(!top.full());
    OVK_CHECK(top.threshold() == std::numeric_limits<float>::infinity());
    const auto out = top.take_sorted();
    OVK_REQUIRE(out.size() == 2);
    OVK_CHECK_EQ(out[0].id, 1u);
    OVK_CHECK_EQ(out[1].id, 0u);
}

OVK_TEST(topk_zero_k_keeps_nothing) {
    TopK top(0);
    OVK_CHECK(!top.push({0, 1.0f}));
    OVK_CHECK(top.take_sorted().empty());
}

OVK_TEST(topk_ties_break_by_smaller_id) {
    TopK top(2);
    top.push({7, 1.0f});
    top.push({3, 1.0f});
    top.push({5, 1.0f});
    top.push({1, 1.0f});
    const auto out = top.take_sorted();
    OVK_REQUIRE(out.size() == 2);
    OVK_CHECK_EQ(out[0].id, 1u);
    OVK_CHECK_EQ(out[1].id, 3u);
}

OVK_TEST(topk_threshold_tightens) {
    TopK top(2);
    top.push({0, 4.0f});
    top.push({1, 2.0f});
    OVK_CHECK_NEAR(top.threshold(), 4.0f, 1e-9);
    OVK_CHECK(top.push({2, 1.0f}));
    OVK_CHECK_NEAR(top.threshold(), 2.0f, 1e-9);
    OVK_CHECK(!top.push({3, 9.0f}));
}

OVK_TEST(topk_matches_full_sort) {
    std::uint32_t state = 42;
    std::vector<Neighbor> all;
    for (Id i = 0; i < 1000; ++i) {
        state = state * 1664525u + 1013904223u;
        // Only 65536 distinct values, so ties actually occur.
        all.push_back({i, static_cast<float>(state >> 16) / 65536.0f});
    }

    for (std::size_t k : {1u, 5u, 17u, 1000u, 2000u}) {
        TopK top(k);
        for (const auto& n : all) top.push(n);

        auto expected = all;
        std::sort(expected.begin(), expected.end());
        expected.resize(std::min(k, expected.size()));

        OVK_CHECK(top.take_sorted() == expected);
    }
}