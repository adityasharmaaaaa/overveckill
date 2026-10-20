#pragma once

#include <concepts>
#include <cstddef>
#include <span>
#include <vector>

#include "overveckill/metric.hpp"
#include "overveckill/types.hpp"

namespace overveckill {

//what the Store (and later the Recall Auditor) requires from an index.
//flatIndex and HnswIndex both satisfy it, so they are interchangeable.
template <class T>
concept VectorIndex = requires(T& idx, const T& cidx, std::span<const float> v, Id id,
                               std::size_t k) {
    { cidx.dim() } -> std::convertible_to<std::size_t>;
    { cidx.metric() } -> std::same_as<Metric>;
    { cidx.size() } -> std::convertible_to<std::size_t>;
    { cidx.slot_count() } -> std::convertible_to<std::size_t>;
    { cidx.contains(id) } -> std::convertible_to<bool>;
    { idx.add(v) } -> std::same_as<Id>;
    { idx.remove(id) } -> std::same_as<bool>;
    { cidx.get(id) } -> std::same_as<std::span<const float>>;
    { cidx.search(v, k) } -> std::same_as<std::vector<Neighbor>>;
};

