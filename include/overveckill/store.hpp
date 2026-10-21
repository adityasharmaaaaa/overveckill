#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "overveckill/flat_index.hpp"
#include "overveckill/index.hpp"
#include "overveckill/metadata.hpp"
#include "overveckill/metric.hpp"
#include "overveckill/types.hpp"

namespace overveckill {

// A full stored record, returned by Store::get.
struct Record {
    std::string key;
    std::vector<float> vector;
    Metadata metadata;
};

// One search result, returned by Store::search.
struct Hit {
    std::string key;
    float distance;
    Metadata metadata;
};

// Maps user-facing string keys to index slots and keeps metadata.
// Not thread-safe (yet).
template <VectorIndex Index>
class BasicStore {
public:
    //wrap an index you configured yourself (e.g. an HnswIndex with custom params).
    explicit BasicStore(Index index) : index_(std::move(index)) {}
    //convenience: build the index from just a dimension and metric.
    explicit BasicStore(std::size_t dim, Metric metric = Metric::L2) : index_(dim, metric) {}

    std::size_t dim() const noexcept { return index_.dim(); }
    Metric metric() const noexcept { return index_.metric(); }

    // Number of stored records.
    std::size_t size() const noexcept { return key_to_slot_.size(); }

    // Slots allocated in the underlying index (diagnostics: shows slot reuse).
    std::size_t slot_count() const noexcept { return index_.slot_count(); }

    bool contains(const std::string& key) const { return key_to_slot_.contains(key); }

    // Inserts or replaces a record. Returns true if the key is new, false if
    // an existing record was replaced. If this throws, the store is unchanged.
    bool upsert(const std::string& key, std::span<const float> vec, Metadata meta = {}) {
        if (key.empty()) throw std::invalid_argument("Store::upsert: key must not be empty");

        // Add first: validation happens inside, so a bad vector throws before
        // we touch anything.
        const Id fresh = index_.add(vec);
        if (fresh >= slots_.size()) slots_.resize(index_.slot_count());
        slots_[fresh] = SlotInfo{key, std::move(meta)};

        const auto it = key_to_slot_.find(key);
        if (it == key_to_slot_.end()) {
            key_to_slot_.emplace(key, fresh);
            return true;
        }

        const Id old = it->second;
        it->second = fresh;
        index_.remove(old);
        slots_[old] = SlotInfo{};
        return false;
    }

    std::optional<Record> get(const std::string& key) const {
        const auto it = key_to_slot_.find(key);
        if (it == key_to_slot_.end()) return std::nullopt;
        const Id slot = it->second;
        const auto v = index_.get(slot);
        return Record{key, std::vector<float>(v.begin(), v.end()), slots_[slot].metadata};
    }

    // Returns true if the key existed.
    bool erase(const std::string& key) {
        const auto it = key_to_slot_.find(key);
        if (it == key_to_slot_.end()) return false;
        const Id slot = it->second;
        key_to_slot_.erase(it);
        index_.remove(slot);
        slots_[slot] = SlotInfo{};
        return true;
    }

    // k nearest records, closest first.
    std::vector<Hit> search(std::span<const float> query, std::size_t k) const {
        const auto neighbors = index_.search(query, k);
        std::vector<Hit> out;
        out.reserve(neighbors.size());
        for (const auto& n : neighbors) {
            const SlotInfo& s = slots_[n.id];
            out.push_back(Hit{s.key, n.distance, s.metadata});
        }
        return out;
    }

private:
    struct SlotInfo {
        std::string key;
        Metadata metadata;
    };

    Index index_;
    std::unordered_map<std::string, Id> key_to_slot_;
    std::vector<SlotInfo> slots_;  // indexed by slot id; empty entries are free slots
};

using Store = BasicStore<FlatIndex>;

}  