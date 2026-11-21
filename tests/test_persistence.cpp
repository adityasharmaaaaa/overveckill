#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "overveckill/flat_index.hpp"
#include "overveckill/hnsw_index.hpp"
#include "overveckill/serialize.hpp"
#include "overveckill/store.hpp"
#include "test_framework.hpp"
#include "test_util.hpp"

using namespace overveckill;

namespace {

template <class T>
std::string to_bytes(const T& obj) {
    std::ostringstream out(std::ios::binary);
    obj.save(out);
    return out.str();
}

template <class T>
T from_bytes(const std::string& bytes) {
    std::istringstream in(bytes, std::ios::binary);
    return T::load(in);
}

std::vector<std::vector<float>> random_vecs(std::size_t n, std::size_t dim, std::uint32_t seed) {
    std::vector<std::vector<float>> out;
    for (std::size_t i = 0; i < n; ++i) out.push_back(ovk_test::random_vec(dim, seed));
    return out;
}

std::filesystem::path temp_path(const char* name) {
    return std::filesystem::temp_directory_path() / name;
}

}  // namespace

// ---- low-level encoding ---------------------------------------------------

OVK_TEST(io_integers_are_little_endian) {
    std::ostringstream out(std::ios::binary);
    io::Writer w(out);
    w.u(std::uint32_t{0x01020304});
    const std::string b = out.str();
    OVK_REQUIRE(b.size() == 4);
    OVK_CHECK_EQ(static_cast<int>(b[0]), 0x04);
    OVK_CHECK_EQ(static_cast<int>(b[3]), 0x01);
}

OVK_TEST(io_roundtrips_scalars_and_metadata) {
    Metadata meta;
    meta["i"] = std::int64_t{-1234567890123};
    meta["d"] = -0.125;
    meta["b"] = true;
    meta["s"] = std::string("hello\0world", 11);

    std::ostringstream out(std::ios::binary);
    io::Writer w(out);
    w.u(std::int32_t{-7});
    w.f32(3.5f);
    w.metadata(meta);

    std::istringstream in(out.str(), std::ios::binary);
    io::Reader r(in);
    OVK_CHECK_EQ(r.u<std::int32_t>(), -7);
    OVK_CHECK_EQ(r.f32(), 3.5f);
    OVK_CHECK(r.metadata() == meta);
}

// ---- FlatIndex ------------------------------------------------------------

OVK_TEST(flat_save_load_roundtrip) {
    FlatIndex idx(8, Metric::Cosine);
    for (const auto& v : random_vecs(100, 8, 1)) idx.add(v);
    idx.remove(3);
    idx.remove(50);

    const FlatIndex back = from_bytes<FlatIndex>(to_bytes(idx));
    OVK_CHECK_EQ(back.dim(), std::size_t{8});
    OVK_CHECK(back.metric() == Metric::Cosine);
    OVK_CHECK_EQ(back.size(), idx.size());
    OVK_CHECK_EQ(back.slot_count(), idx.slot_count());
    OVK_CHECK(!back.contains(3));
    OVK_CHECK(!back.contains(50));

    for (const auto& q : random_vecs(20, 8, 99)) {
        OVK_CHECK(back.search(q, 10) == idx.search(q, 10));
    }
    // Saving again gives identical bytes.
    OVK_CHECK(to_bytes(back) == to_bytes(idx));
}

OVK_TEST(flat_load_preserves_slot_reuse_order) {
    FlatIndex idx(2);
    for (const auto& v : random_vecs(10, 2, 5)) idx.add(v);
    idx.remove(2);
    idx.remove(7);

    FlatIndex back = from_bytes<FlatIndex>(to_bytes(idx));
    const std::vector<float> v{0.5f, 0.5f};
    OVK_CHECK_EQ(back.add(v), idx.add(v));
    OVK_CHECK_EQ(back.add(v), idx.add(v));
}

OVK_TEST(flat_empty_roundtrip) {
    const FlatIndex idx(4);
    const FlatIndex back = from_bytes<FlatIndex>(to_bytes(idx));
    OVK_CHECK_EQ(back.size(), std::size_t{0});
    OVK_CHECK_EQ(back.dim(), std::size_t{4});
}

// ---- HnswIndex ------------------------------------------------------------

OVK_TEST(hnsw_save_load_roundtrip) {
    HnswParams p;
    p.M = 8;
    p.ef_construction = 64;
    p.ef_search = 32;
    HnswIndex idx(16, Metric::L2, p);
    for (const auto& v : random_vecs(800, 16, 7)) idx.add(v);
    idx.remove(10);
    idx.remove(400);

    const HnswIndex back = from_bytes<HnswIndex>(to_bytes(idx));
    OVK_CHECK(back.validate());
    OVK_CHECK_EQ(back.size(), idx.size());
    OVK_CHECK_EQ(back.slot_count(), idx.slot_count());
    OVK_CHECK_EQ(back.max_level(), idx.max_level());
    OVK_CHECK_EQ(back.params().M, std::size_t{8});
    OVK_CHECK_EQ(back.params().ef_search, std::size_t{32});
    OVK_CHECK(!back.contains(10));

    // Same graph, so search results are identical, not just similar.
    for (const auto& q : random_vecs(30, 16, 123)) {
        OVK_CHECK(back.search(q, 10) == idx.search(q, 10));
        OVK_CHECK(back.search(q, 5, 100) == idx.search(q, 5, 100));
    }
    OVK_CHECK(to_bytes(back) == to_bytes(idx));
}

OVK_TEST(hnsw_inserts_after_load_match_original) {
    HnswIndex a(8);
    for (const auto& v : random_vecs(200, 8, 3)) a.add(v);
    HnswIndex b = from_bytes<HnswIndex>(to_bytes(a));

    // The RNG state is restored, so both indexes keep building the same graph.
    for (const auto& v : random_vecs(200, 8, 4)) {
        a.add(v);
        b.add(v);
    }
    OVK_CHECK(b.validate());
    OVK_CHECK(to_bytes(a) == to_bytes(b));
}

OVK_TEST(hnsw_empty_roundtrip) {
    const HnswIndex idx(3, Metric::InnerProduct);
    const HnswIndex back = from_bytes<HnswIndex>(to_bytes(idx));
    OVK_CHECK_EQ(back.size(), std::size_t{0});
    OVK_CHECK_EQ(back.max_level(), -1);
    OVK_CHECK(back.metric() == Metric::InnerProduct);
    OVK_CHECK(back.search(std::vector<float>{1, 2, 3}, 5).empty());
}

// ---- Store ----------------------------------------------------------------

OVK_TEST(store_save_load_roundtrip) {
    Store store(4);
    const auto vecs = random_vecs(50, 4, 11);
    for (std::size_t i = 0; i < vecs.size(); ++i) {
        Metadata m;
        m["i"] = static_cast<std::int64_t>(i);
        m["even"] = (i % 2 == 0);
        m["name"] = "doc-" + std::to_string(i);
        store.upsert("k" + std::to_string(i), vecs[i], m);
    }
    store.erase("k3");
    store.upsert("k4", vecs[0]);  // replace: moves to a new slot

    const Store back = from_bytes<Store>(to_bytes(store));
    OVK_CHECK_EQ(back.size(), store.size());
    OVK_CHECK(!back.contains("k3"));
    for (std::size_t i = 0; i < vecs.size(); ++i) {
        const std::string key = "k" + std::to_string(i);
        const auto a = store.get(key);
        const auto b = back.get(key);
        OVK_REQUIRE(a.has_value() == b.has_value());
        if (!a) continue;
        OVK_CHECK(a->vector == b->vector);
        OVK_CHECK(a->metadata == b->metadata);
    }
    for (const auto& q : random_vecs(10, 4, 77)) {
        const auto ha = store.search(q, 5);
        const auto hb = back.search(q, 5);
        OVK_REQUIRE(ha.size() == hb.size());
        for (std::size_t i = 0; i < ha.size(); ++i) {
            OVK_CHECK_EQ(ha[i].key, hb[i].key);
            OVK_CHECK_EQ(ha[i].distance, hb[i].distance);
            OVK_CHECK(ha[i].metadata == hb[i].metadata);
        }
    }
}

OVK_TEST(store_hnsw_file_roundtrip) {
    using HStore = BasicStore<HnswIndex>;
    HStore store(HnswIndex(6, Metric::L2));
    const auto vecs = random_vecs(300, 6, 21);
    for (std::size_t i = 0; i < vecs.size(); ++i) {
        store.upsert("v" + std::to_string(i), vecs[i]);
    }
    store.erase("v0");

    const auto path = temp_path("ovk_test_store_hnsw.ovk");
    store.save(path);
    OVK_CHECK(!std::filesystem::exists(path.string() + ".tmp"));
    HStore back = HStore::load(path);
    std::filesystem::remove(path);

    OVK_CHECK_EQ(back.size(), store.size());
    for (const auto& q : random_vecs(10, 6, 5)) {
        const auto ha = store.search(q, 5);
        const auto hb = back.search(q, 5);
        OVK_REQUIRE(ha.size() == hb.size());
        for (std::size_t i = 0; i < ha.size(); ++i) OVK_CHECK_EQ(ha[i].key, hb[i].key);
    }
    // Still writable after loading.
    OVK_CHECK(back.upsert("new", vecs[1]));
    OVK_CHECK(back.contains("new"));
}

// ---- corrupt input --------------------------------------------------------

OVK_TEST(load_rejects_wrong_magic) {
    FlatIndex idx(2);
    idx.add(std::vector<float>{1, 2});
    const std::string bytes = to_bytes(idx);
    OVK_CHECK_THROWS(from_bytes<HnswIndex>(bytes), SerializationError);
    OVK_CHECK_THROWS(from_bytes<FlatIndex>(std::string("nope")), SerializationError);
    OVK_CHECK_THROWS(from_bytes<FlatIndex>(std::string()), SerializationError);
}

OVK_TEST(load_rejects_future_version) {
    FlatIndex idx(2);
    std::string bytes = to_bytes(idx);
    bytes[4] = 2;  // format version field
    OVK_CHECK_THROWS(from_bytes<FlatIndex>(bytes), SerializationError);
}

OVK_TEST(load_rejects_every_truncation) {
    HnswIndex idx(4);
    for (const auto& v : random_vecs(30, 4, 8)) idx.add(v);
    const std::string bytes = to_bytes(idx);
    for (std::size_t n = 0; n < bytes.size(); ++n) {
        OVK_CHECK_THROWS(from_bytes<HnswIndex>(bytes.substr(0, n)), SerializationError);
    }
}

OVK_TEST(load_rejects_huge_length_without_allocating) {
    // Valid header, then a vector length of 2^60 floats and no data.
    std::ostringstream out(std::ios::binary);
    io::Writer w(out);
    w.header("OVKF");
    w.len(4);
    w.metric(Metric::L2);
    w.u(std::uint64_t{1} << 60);
    OVK_CHECK_THROWS(from_bytes<FlatIndex>(out.str()), SerializationError);
}

OVK_TEST(load_rejects_corrupt_hnsw_link) {
    HnswIndex idx(2);
    for (const auto& v : random_vecs(20, 2, 9)) idx.add(v);
    std::string bytes = to_bytes(idx);
    // Entry point (u32) and max level (i32) are the last 8 bytes.
    // Point the entry at a slot that doesn't exist.
    bytes[bytes.size() - 8] = static_cast<char>(0xFF);
    bytes[bytes.size() - 7] = static_cast<char>(0xFF);
    OVK_CHECK_THROWS(from_bytes<HnswIndex>(bytes), SerializationError);
}

OVK_TEST(load_file_rejects_trailing_data_and_missing_file) {
    const auto path = temp_path("ovk_test_trailing.ovk");
    FlatIndex idx(2);
    idx.save(path);
    {
        std::ofstream out(path, std::ios::binary | std::ios::app);
        out << "junk";
    }
    OVK_CHECK_THROWS(FlatIndex::load(path), SerializationError);
    std::filesystem::remove(path);
    OVK_CHECK_THROWS(FlatIndex::load(path), SerializationError);
}
