#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "overveckill/store.hpp"
#include "test_framework.hpp"
#include "test_util.hpp"

using namespace overveckill;

OVK_TEST(store_upsert_and_get) {
    Store store(2);
    const std::vector<float> v{1.0f, 2.0f};
    Metadata meta;
    meta["lang"] = std::string("en");
    meta["year"] = std::int64_t{2024};
    meta["score"] = 0.5;
    meta["draft"] = false;

    store.upsert("doc-1", v, meta);

    const auto rec = store.get("doc-1");
    OVK_REQUIRE(rec.has_value());
    OVK_CHECK_EQ(rec->key, std::string("doc-1"));
    OVK_CHECK(rec->vector == v);
    OVK_CHECK(rec->metadata == meta);
    OVK_CHECK(std::holds_alternative<std::int64_t>(rec->metadata.at("year")));
    OVK_CHECK(std::holds_alternative<bool>(rec->metadata.at("draft")));
}

OVK_TEST(store_upsert_reports_insert_vs_update) {
    Store store(2);
    const std::vector<float> v{1.0f, 2.0f};
    OVK_CHECK(store.upsert("a", v));
    OVK_CHECK(!store.upsert("a", v));
    OVK_CHECK(store.upsert("b", v));
    OVK_CHECK_EQ(store.size(), std::size_t{2});
}

OVK_TEST(store_upsert_replaces_vector_and_metadata) {
    Store store(2);
    const std::vector<float> v1{1.0f, 1.0f};
    const std::vector<float> v2{9.0f, 9.0f};
    Metadata m1;
    m1["rev"] = std::int64_t{1};
    Metadata m2;
    m2["rev"] = std::int64_t{2};
    m2["note"] = std::string("edited");

    store.upsert("doc", v1, m1);
    store.upsert("doc", v2, m2);

    OVK_CHECK_EQ(store.size(), std::size_t{1});
    const auto rec = store.get("doc");
    OVK_REQUIRE(rec.has_value());
    OVK_CHECK(rec->vector == v2);
    OVK_CHECK(rec->metadata == m2);
}

OVK_TEST(store_get_missing_returns_nullopt) {
    Store store(2);
    OVK_CHECK(!store.get("nope").has_value());
    OVK_CHECK(!store.contains("nope"));
}

OVK_TEST(store_erase) {
    Store store(2);
    const std::vector<float> v{1.0f, 2.0f};
    store.upsert("a", v);

    OVK_CHECK(!store.erase("unknown"));
    OVK_CHECK(store.erase("a"));
    OVK_CHECK(!store.erase("a"));
    OVK_CHECK(!store.contains("a"));
    OVK_CHECK(!store.get("a").has_value());
    OVK_CHECK_EQ(store.size(), std::size_t{0});
}

OVK_TEST(store_search_returns_keys_and_metadata) {
    Store store(2, Metric::L2);
    Metadata ma;
    ma["tag"] = std::string("alpha");
    Metadata mb;
    mb["tag"] = std::string("beta");
    const std::vector<float> near_v{1.0f, 1.0f};
    const std::vector<float> far_v{10.0f, 10.0f};
    store.upsert("far", far_v, mb);
    store.upsert("near", near_v, ma);

    const std::vector<float> q{0.0f, 0.0f};
    const auto hits = store.search(q, 2);
    OVK_REQUIRE(hits.size() == 2);
    OVK_CHECK_EQ(hits[0].key, std::string("near"));
    OVK_CHECK_NEAR(hits[0].distance, 2.0f, 1e-6);
    OVK_CHECK(hits[0].metadata == ma);
    OVK_CHECK_EQ(hits[1].key, std::string("far"));
    OVK_CHECK_NEAR(hits[1].distance, 200.0f, 1e-4);
    OVK_CHECK(hits[1].metadata == mb);
}

OVK_TEST(store_search_ignores_replaced_and_erased) {
    Store store(2);
    const std::vector<float> origin{0.0f, 0.0f};
    const std::vector<float> mid{5.0f, 5.0f};
    const std::vector<float> faraway{100.0f, 100.0f};
    store.upsert("a", origin);
    store.upsert("b", mid);

    // Move "a" far away: its old position must no longer match.
    store.upsert("a", faraway);
    auto hits = store.search(origin, 5);
    OVK_REQUIRE(hits.size() == 2);
    OVK_CHECK_EQ(hits[0].key, std::string("b"));
    OVK_CHECK_EQ(hits[1].key, std::string("a"));

    store.erase("b");
    hits = store.search(origin, 5);
    OVK_REQUIRE(hits.size() == 1);
    OVK_CHECK_EQ(hits[0].key, std::string("a"));
}

OVK_TEST(store_failed_upsert_changes_nothing) {
    Store store(2);
    const std::vector<float> good{1.0f, 2.0f};
    const std::vector<float> bad_dim{1.0f};
    const std::vector<float> bad_nan{1.0f, std::numeric_limits<float>::quiet_NaN()};
    Metadata m1;
    m1["v"] = std::int64_t{1};
    store.upsert("a", good, m1);

    OVK_CHECK_THROWS(store.upsert("a", bad_dim), std::invalid_argument);
    OVK_CHECK_THROWS(store.upsert("a", bad_nan), std::invalid_argument);
    OVK_CHECK_THROWS(store.upsert("new", bad_dim), std::invalid_argument);

    OVK_CHECK_EQ(store.size(), std::size_t{1});
    OVK_CHECK(!store.contains("new"));
    const auto rec = store.get("a");
    OVK_REQUIRE(rec.has_value());
    OVK_CHECK(rec->vector == good);
    OVK_CHECK(rec->metadata == m1);
}

OVK_TEST(store_rejects_empty_key) {
    Store store(2);
    const std::vector<float> v{1.0f, 2.0f};
    OVK_CHECK_THROWS(store.upsert("", v), std::invalid_argument);
    OVK_CHECK_EQ(store.size(), std::size_t{0});
}

OVK_TEST(store_reuses_slots) {
    Store store(2);
    const std::vector<float> v{1.0f, 2.0f};

    for (int i = 0; i < 100; ++i) store.upsert("same", v);
    OVK_CHECK_EQ(store.size(), std::size_t{1});
    OVK_CHECK(store.slot_count() <= std::size_t{2});

    for (int i = 0; i < 100; ++i) {
        const std::string key = "k" + std::to_string(i);
        store.upsert(key, v);
        store.erase(key);
    }
    OVK_CHECK_EQ(store.size(), std::size_t{1});
    OVK_CHECK(store.slot_count() <= std::size_t{3});
}

OVK_TEST(store_matches_reference_after_churn) {
    constexpr std::size_t dim = 8;
    std::uint32_t state = 2024;
    Store store(dim, Metric::L2);
    std::map<std::string, std::vector<float>> ref;

    // 500 random upserts and erases over only 40 distinct keys.
    for (int step = 0; step < 500; ++step) {
        state = state * 1664525u + 1013904223u;
        const std::string key = "k" + std::to_string((state >> 16) % 40);
        state = state * 1664525u + 1013904223u;
        if ((state >> 16) % 4 == 0) {
            store.erase(key);
            ref.erase(key);
        } else {
            const auto v = ovk_test::random_vec(dim, state);
            store.upsert(key, v);
            ref[key] = v;
        }
    }

    OVK_CHECK_EQ(store.size(), ref.size());
    for (const auto& [key, v] : ref) {
        const auto rec = store.get(key);
        OVK_REQUIRE(rec.has_value());
        OVK_CHECK(rec->vector == v);
    }

    for (int qn = 0; qn < 10; ++qn) {
        const auto q = ovk_test::random_vec(dim, state);

        std::vector<std::pair<float, std::string>> expected;
        for (const auto& [key, v] : ref) {
            expected.emplace_back(distance(Metric::L2, q, v), key);
        }
        std::sort(expected.begin(), expected.end());
        if (expected.size() > 5) expected.resize(5);

        const auto hits = store.search(q, 5);
        OVK_REQUIRE(hits.size() == expected.size());
        for (std::size_t i = 0; i < hits.size(); ++i) {
            OVK_CHECK_EQ(hits[i].key, expected[i].second);
        }
    }
}