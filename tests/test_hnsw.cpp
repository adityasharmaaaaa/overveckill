#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "overveckill/flat_index.hpp"
#include "overveckill/hnsw_index.hpp"
#include "overveckill/index.hpp"
#include "overveckill/store.hpp"
#include "test_framework.hpp"
#include "test_util.hpp"

using namespace overveckill;

static_assert(VectorIndex<FlatIndex>);
static_assert(VectorIndex<HnswIndex>);

namespace {

// Fraction of the exact top-k that the approximate search also returned.
double recall_at_k(const FlatIndex& flat, const HnswIndex& hnsw,
                   const std::vector<std::vector<float>>& queries, std::size_t k,
                   std::size_t ef) {
    std::size_t hits = 0;
    std::size_t total = 0;
    for (const auto& q : queries) {
        const auto truth = flat.search(q, k);
        const auto got = hnsw.search(q, k, ef);
        for (const auto& t : truth) {
            ++total;
            for (const auto& g : got) {
                if (g.id == t.id) {
                    ++hits;
                    break;
                }
            }
        }
    }
    return total == 0 ? 1.0 : static_cast<double>(hits) / static_cast<double>(total);
}

}  // namespace

OVK_TEST(hnsw_empty_index_returns_nothing) {
    HnswIndex idx(3);
    const std::vector<float> q{1.0f, 2.0f, 3.0f};
    OVK_CHECK_EQ(idx.size(), std::size_t{0});
    OVK_CHECK_EQ(idx.max_level(), -1);
    OVK_CHECK(idx.search(q, 5).empty());
    OVK_CHECK(idx.validate());
}

OVK_TEST(hnsw_single_vector) {
    HnswIndex idx(2);
    const std::vector<float> v{1.0f, 2.0f};
    OVK_CHECK_EQ(idx.add(v), 0u);
    const auto res = idx.search(v, 3);
    OVK_REQUIRE(res.size() == 1);
    OVK_CHECK_EQ(res[0].id, 0u);
    OVK_CHECK_NEAR(res[0].distance, 0.0f, 1e-9);
    OVK_CHECK(idx.validate());
}

OVK_TEST(hnsw_ids_sequential_and_get_roundtrips) {
    HnswIndex idx(2);
    const std::vector<float> a{1.0f, 2.0f};
    const std::vector<float> b{3.0f, 4.0f};
    OVK_CHECK_EQ(idx.add(a), 0u);
    OVK_CHECK_EQ(idx.add(b), 1u);
    const auto got = idx.get(1);
    OVK_REQUIRE(got.size() == 2);
    OVK_CHECK_NEAR(got[0], 3.0f, 1e-9);
    OVK_CHECK_NEAR(got[1], 4.0f, 1e-9);
    OVK_CHECK_THROWS(idx.get(5), std::out_of_range);
}

OVK_TEST(hnsw_input_validation) {
    HnswIndex idx(2);
    const std::vector<float> one{1.0f};
    const std::vector<float> bad_nan{1.0f, std::numeric_limits<float>::quiet_NaN()};
    OVK_CHECK_THROWS(idx.add(one), std::invalid_argument);
    OVK_CHECK_THROWS(idx.add(bad_nan), std::invalid_argument);
    OVK_CHECK_EQ(idx.size(), std::size_t{0});
    OVK_CHECK_EQ(idx.slot_count(), std::size_t{0});

    const std::vector<float> ok{1.0f, 2.0f};
    idx.add(ok);
    OVK_CHECK_THROWS(idx.search(one, 1), std::invalid_argument);
    OVK_CHECK_THROWS(idx.search(bad_nan, 1), std::invalid_argument);
}

OVK_TEST(hnsw_rejects_bad_params) {
    OVK_CHECK_THROWS(HnswIndex(0), std::invalid_argument);
    OVK_CHECK_THROWS(HnswIndex(4, Metric::L2, HnswParams{.M = 1}), std::invalid_argument);
    OVK_CHECK_THROWS(HnswIndex(4, Metric::L2, HnswParams{.ef_search = 0}),
                     std::invalid_argument);
}

OVK_TEST(hnsw_k_zero_and_huge_k) {
    HnswIndex idx(1);
    for (float x : {3.0f, 1.0f, 2.0f}) {
        const std::vector<float> v{x};
        idx.add(v);
    }
    const std::vector<float> q{0.0f};
    OVK_CHECK(idx.search(q, 0).empty());

    const auto res = idx.search(q, std::numeric_limits<std::size_t>::max());
    OVK_REQUIRE(res.size() == 3);
    OVK_CHECK_EQ(res[0].id, 1u);
    OVK_CHECK_EQ(res[1].id, 2u);
    OVK_CHECK_EQ(res[2].id, 0u);
}

OVK_TEST(hnsw_small_set_is_exact) {
    // With ef >= n the search can reach every node, so it must equal brute force.
    constexpr std::size_t dim = 6;
    constexpr std::size_t n = 100;
    std::uint32_t state = 5;
    FlatIndex flat(dim);
    HnswIndex hnsw(dim);
    for (std::size_t i = 0; i < n; ++i) {
        const auto v = ovk_test::random_vec(dim, state);
        flat.add(v);
        hnsw.add(v);
    }
    OVK_CHECK(hnsw.validate());
    for (int qn = 0; qn < 20; ++qn) {
        const auto q = ovk_test::random_vec(dim, state);
        OVK_CHECK(hnsw.search(q, 10, n) == flat.search(q, 10));
    }
}

OVK_TEST(hnsw_self_query_finds_itself) {
    constexpr std::size_t count = 300;
    std::uint32_t state = 77;
    HnswIndex idx(12);
    std::vector<std::vector<float>> vs;
    for (std::size_t i = 0; i < count; ++i) {
        vs.push_back(ovk_test::random_vec(12, state));
        idx.add(vs.back());
    }
    std::size_t found = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const auto res = idx.search(vs[i], 1);
        if (!res.empty() && res[0].id == static_cast<Id>(i)) ++found;
    }
    OVK_CHECK(found >= count * 99 / 100);
}

OVK_TEST(hnsw_graph_invariants_hold_for_all_metrics) {
    for (Metric m : {Metric::L2, Metric::InnerProduct, Metric::Cosine}) {
        std::uint32_t state = 11;
        HnswIndex idx(8, m, HnswParams{.M = 6, .ef_construction = 40});
        for (int i = 0; i < 500; ++i) idx.add(ovk_test::random_vec(8, state));
        OVK_CHECK(idx.validate());
        OVK_CHECK(idx.max_level() >= 1);  // 500 nodes with M=6 will have upper layers
    }
}

OVK_TEST(hnsw_recall_is_high_and_improves_with_ef) {
    constexpr std::size_t dim = 16;
    constexpr std::size_t n = 3000;
    constexpr std::size_t k = 10;
    std::uint32_t state = 2025;
    FlatIndex flat(dim);
    HnswIndex hnsw(dim);
    for (std::size_t i = 0; i < n; ++i) {
        const auto v = ovk_test::random_vec(dim, state);
        flat.add(v);
        hnsw.add(v);
    }
    std::vector<std::vector<float>> queries;
    for (int i = 0; i < 100; ++i) queries.push_back(ovk_test::random_vec(dim, state));

    const double low = recall_at_k(flat, hnsw, queries, k, 10);
    const double mid = recall_at_k(flat, hnsw, queries, k, 64);
    const double high = recall_at_k(flat, hnsw, queries, k, 256);
    std::cout << "    recall@10: ef=10 -> " << low << ", ef=64 -> " << mid
              << ", ef=256 -> " << high << "\n";

    OVK_CHECK(low <= mid);
    OVK_CHECK(mid <= high);
    OVK_CHECK(mid >= 0.85);
    OVK_CHECK(high >= 0.97);
}

OVK_TEST(hnsw_same_seed_builds_same_graph) {
    constexpr std::size_t dim = 8;
    std::uint32_t s1 = 3, s2 = 3;
    HnswIndex a(dim, Metric::L2, HnswParams{.seed = 9});
    HnswIndex b(dim, Metric::L2, HnswParams{.seed = 9});
    for (int i = 0; i < 400; ++i) {
        a.add(ovk_test::random_vec(dim, s1));
        b.add(ovk_test::random_vec(dim, s2));
    }
    OVK_CHECK_EQ(a.max_level(), b.max_level());
    std::uint32_t qs = 8;
    for (int i = 0; i < 20; ++i) {
        const auto q = ovk_test::random_vec(dim, qs);
        OVK_CHECK(a.search(q, 10) == b.search(q, 10));
    }
}

OVK_TEST(hnsw_removed_vectors_never_returned) {
    constexpr std::size_t dim = 8;
    constexpr std::size_t n = 600;
    std::uint32_t state = 404;
    HnswIndex idx(dim);
    for (std::size_t i = 0; i < n; ++i) idx.add(ovk_test::random_vec(dim, state));

    for (Id id = 0; id < n; id += 2) OVK_CHECK(idx.remove(id));
    OVK_CHECK(!idx.remove(0));      // already removed
    OVK_CHECK(!idx.remove(99999));  // unknown
    OVK_CHECK_EQ(idx.size(), n / 2);
    OVK_CHECK_EQ(idx.slot_count(), n);  // tombstones keep their slots
    OVK_CHECK(!idx.contains(2));
    OVK_CHECK_THROWS(idx.get(2), std::out_of_range);
    OVK_CHECK(idx.validate());

    for (int qn = 0; qn < 30; ++qn) {
        const auto q = ovk_test::random_vec(dim, state);
        const auto res = idx.search(q, 10, 100);
        OVK_CHECK_EQ(res.size(), std::size_t{10});
        for (const auto& r : res) OVK_CHECK(r.id % 2 == 1);
    }
}

OVK_TEST(hnsw_remove_everything_then_search) {
    HnswIndex idx(2);
    const std::vector<float> a{1.0f, 1.0f};
    const std::vector<float> b{2.0f, 2.0f};
    idx.add(a);
    idx.add(b);
    idx.remove(0);
    idx.remove(1);
    OVK_CHECK_EQ(idx.size(), std::size_t{0});
    OVK_CHECK(idx.search(a, 5).empty());

    // Still usable afterwards.
    const std::vector<float> c{3.0f, 3.0f};
    const Id id = idx.add(c);
    const auto res = idx.search(c, 5);
    OVK_REQUIRE(res.size() == 1);
    OVK_CHECK_EQ(res[0].id, id);
}

OVK_TEST(hnsw_works_inside_store) {
    using HnswStore = BasicStore<HnswIndex>;
    HnswStore store(HnswIndex(4, Metric::L2, HnswParams{.M = 8, .seed = 1}));

    std::uint32_t state = 1;
    for (int i = 0; i < 200; ++i) {
        Metadata m;
        m["i"] = std::int64_t{i};
        store.upsert("doc-" + std::to_string(i), ovk_test::random_vec(4, state), m);
    }
    OVK_CHECK_EQ(store.size(), std::size_t{200});

    // Replace one, erase one.
    const std::vector<float> marker{50.0f, 50.0f, 50.0f, 50.0f};
    OVK_CHECK(!store.upsert("doc-7", marker));
    OVK_CHECK(store.erase("doc-8"));
    OVK_CHECK_EQ(store.size(), std::size_t{199});

    const auto hits = store.search(marker, 3);
    OVK_REQUIRE(!hits.empty());
    OVK_CHECK_EQ(hits[0].key, std::string("doc-7"));
    for (const auto& h : hits) OVK_CHECK(h.key != "doc-8");
    OVK_CHECK(!store.get("doc-8").has_value());

    // Default-params convenience constructor also works.
    HnswStore plain(4);
    plain.upsert("x", marker);
    OVK_CHECK_EQ(plain.search(marker, 1).size(), std::size_t{1});
}