#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "overveckill/distance.hpp"
#include "overveckill/metric.hpp"
#include "overveckill/topk.hpp"
#include "overveckill/types.hpp"

namespace overveckill {

struct HnswParams {
    // Max links per node on layers >= 1. Layer 0 allows 2*M.
    std::size_t M = 16;
    // Candidate list size while building. Bigger = better graph, slower build.
    std::size_t ef_construction = 200;
    // Default candidate list size while searching. Bigger = better recall, slower.
    std::size_t ef_search = 64;
    // Seeds the level generator, so the same inputs build the same graph.
    std::uint64_t seed = 42;
};

// Hierarchical Navigable Small World graph (Malkov & Yashunin, 2016).
//
// Every vector is a node. Each node lives on layers 0..level, where the level
// is drawn at random (most nodes only on layer 0, exponentially fewer higher
// up). Search enters at the top, greedily hops toward the query on each upper
// layer, then runs a wider best-first search on layer 0.
//
// Deletes are TOMBSTONES for now: a removed node stays in the graph (so paths
// through it keep working) but is never returned. Its slot is not reused.
// Phase 6 replaces this with real graph repair.
class HnswIndex {
public:
    explicit HnswIndex(std::size_t dim, Metric metric = Metric::L2, HnswParams params = {})
        : dim_(dim),
          metric_(metric),
          dist_(get_distance_fn(metric)),
          params_(params),
          rng_(params.seed) {
        if (dim == 0) throw std::invalid_argument("HnswIndex: dim must be > 0");
        if (params_.M < 2) throw std::invalid_argument("HnswIndex: M must be >= 2");
        if (params_.ef_search == 0) {
            throw std::invalid_argument("HnswIndex: ef_search must be > 0");
        }
        params_.ef_construction = std::max(params_.ef_construction, params_.M);
        level_mult_ = 1.0 / std::log(static_cast<double>(params_.M));
    }

    std::size_t dim() const noexcept { return dim_; }
    Metric metric() const noexcept { return metric_; }
    const HnswParams& params() const noexcept { return params_; }

    // Every id ever handed out (live or tombstoned). Valid ids are < slot_count().
    std::size_t slot_count() const noexcept { return levels_.size(); }

    // Live (searchable) vectors.
    std::size_t size() const noexcept { return live_count_; }

    bool contains(Id id) const noexcept { return id < slot_count() && deleted_[id] == 0; }

    // Highest layer in the graph, or -1 if the index is empty.
    int max_level() const noexcept { return max_level_; }

    Id add(std::span<const float> v) {
        check_vector(v, "add");
        if (slot_count() >= std::numeric_limits<Id>::max()) {
            throw std::length_error("HnswIndex: id space exhausted");
        }

        const Id id = static_cast<Id>(slot_count());
        const int level = random_level();

        data_.insert(data_.end(), v.begin(), v.end());
        levels_.push_back(static_cast<std::uint8_t>(level));
        deleted_.push_back(0);
        links_.emplace_back(static_cast<std::size_t>(level) + 1);
        ++live_count_;

        if (max_level_ < 0) {  // first node: nothing to connect to
            entry_ = id;
            max_level_ = level;
            return id;
        }

        connect(id, level);
        if (level > max_level_) {
            entry_ = id;
            max_level_ = level;
        }
        return id;
    }

    // Tombstone delete. Returns false if the id is unknown or already removed.
    bool remove(Id id) {
        if (!contains(id)) return false;
        deleted_[id] = 1;
        --live_count_;
        return true;
    }

    std::span<const float> get(Id id) const {
        if (!contains(id)) throw std::out_of_range("HnswIndex::get: bad or removed id");
        return {vec(id), dim_};
    }

    // Approximate k nearest live vectors, closest first, using params().ef_search.
    std::vector<Neighbor> search(std::span<const float> query, std::size_t k) const {
        return search(query, k, params_.ef_search);
    }

    // Same, with an explicit ef (candidate list size). ef is raised to at least k.
    std::vector<Neighbor> search(std::span<const float> query, std::size_t k,
                                 std::size_t ef) const {
        check_vector(query, "search");
        if (k == 0 || live_count_ == 0) return {};

        ef = std::min(std::max(ef, k), slot_count());  // capped: huge values can't over-allocate
        const Id start = greedy_closest(query.data(), entry_, max_level_, 0);
        std::vector<Neighbor> found =
            search_layer(query.data(), std::span<const Id>(&start, 1), ef, 0, true);
        if (found.size() > k) found.resize(k);
        return found;
    }

    // Debug aid: verifies the structural invariants of the graph.
    // Returns true if everything is consistent.
    bool validate() const {
        const std::size_t n = slot_count();
        if (n == 0) return max_level_ == -1;
        if (entry_ >= n || levels_[entry_] != max_level_) return false;

        for (std::size_t i = 0; i < n; ++i) {
            const Id id = static_cast<Id>(i);
            if (links_[i].size() != static_cast<std::size_t>(levels_[i]) + 1) return false;
            if (levels_[i] > max_level_) return false;
            for (int layer = 0; layer <= levels_[i]; ++layer) {
                const std::vector<Id>& list = links(id, layer);
                if (list.size() > max_links(layer)) return false;
                std::vector<Id> sorted = list;
                std::sort(sorted.begin(), sorted.end());
                if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
                    return false;  // duplicate link
                }
                for (Id nb : list) {
                    if (nb >= n || nb == id) return false;
                    if (levels_[nb] < layer) return false;  // link to a node not on this layer
                }
            }
        }
        return true;
    }

private:
    // ---- helpers ---------------------------------------------------------

    const float* vec(Id id) const noexcept {
        return data_.data() + static_cast<std::size_t>(id) * dim_;
    }

    std::vector<Id>& links(Id id, int layer) {
        return links_[id][static_cast<std::size_t>(layer)];
    }
    const std::vector<Id>& links(Id id, int layer) const {
        return links_[id][static_cast<std::size_t>(layer)];
    }

    std::size_t max_links(int layer) const noexcept {
        return layer == 0 ? params_.M * 2 : params_.M;
    }

    // Level ~ floor(-ln(U) / ln(M)): P(level >= l) = M^-l.
    int random_level() {
        // 53 random bits -> u in [0, 1). We use 1-u in (0, 1] so log() never sees 0.
        const double u = static_cast<double>(rng_() >> 11) * (1.0 / 9007199254740992.0);
        const double r = -std::log(1.0 - u) * level_mult_;
        return std::min(static_cast<int>(r), 30);
    }

    // ---- search ----------------------------------------------------------

    // Walks down from `from_layer` to `to_layer + 1`, always hopping to the
    // closest neighbor until no neighbor improves. Returns where it ended up.
    Id greedy_closest(const float* q, Id entry, int from_layer, int to_layer) const {
        Id cur = entry;
        float cur_d = dist_(q, vec(cur), dim_);
        for (int layer = from_layer; layer > to_layer; --layer) {
            bool improved = true;
            while (improved) {
                improved = false;
                Id best = cur;
                float best_d = cur_d;
                for (Id nb : links(cur, layer)) {
                    const float d = dist_(q, vec(nb), dim_);
                    if (d < best_d) {
                        best = nb;
                        best_d = d;
                    }
                }
                if (best != cur) {
                    cur = best;
                    cur_d = best_d;
                    improved = true;
                }
            }
        }
        return cur;
    }

    // Best-first search inside one layer. Keeps the `ef` closest nodes found.
    // Returns them sorted closest-first.
    //
    // skip_deleted: tombstoned nodes are still walked through (they keep the
    // graph connected) but are not allowed into the result set.
    std::vector<Neighbor> search_layer(const float* q, std::span<const Id> entries,
                                       std::size_t ef, int layer, bool skip_deleted) const {
        std::vector<std::uint8_t> visited(slot_count(), 0);

        // Candidates to expand: min-heap (closest on top).
        auto farther = [](const Neighbor& a, const Neighbor& b) { return b < a; };
        std::priority_queue<Neighbor, std::vector<Neighbor>, decltype(farther)> candidates(
            farther);
        // Best results so far: bounded max-heap (worst on top).
        TopK results(ef);

        for (Id e : entries) {
            if (visited[e]) continue;
            visited[e] = 1;
            const Neighbor n{e, dist_(q, vec(e), dim_)};
            candidates.push(n);
            if (!(skip_deleted && deleted_[e])) results.push(n);
        }

        while (!candidates.empty()) {
            const Neighbor c = candidates.top();
            // The closest unexpanded candidate is already worse than our worst
            // kept result: nothing reachable from here can improve the answer.
            if (results.full() && c.distance > results.threshold()) break;
            candidates.pop();

            for (Id nb : links(c.id, layer)) {
                if (visited[nb]) continue;
                visited[nb] = 1;
                const Neighbor n{nb, dist_(q, vec(nb), dim_)};
                if (!results.full() || n.distance < results.threshold()) {
                    candidates.push(n);
                    if (!(skip_deleted && deleted_[nb])) results.push(n);
                }
            }
        }
        return results.take_sorted();
    }


    // Heuristic neighbor selection (the paper's Algorithm 4).
    // `cands` must be sorted closest-first relative to the base node.
    // A candidate is kept only if it is closer to the base node than to any
    // neighbor already kept. This spreads links in different directions
    // instead of wasting them all on one tight cluster.
    std::vector<Neighbor> select_neighbors(const std::vector<Neighbor>& cands,
                                           std::size_t m) const {
        std::vector<Neighbor> chosen;
        chosen.reserve(m);
        for (const Neighbor& c : cands) {
            if (chosen.size() >= m) break;
            bool keep = true;
            for (const Neighbor& s : chosen) {
                if (dist_(vec(c.id), vec(s.id), dim_) < c.distance) {
                    keep = false;
                    break;
                }
            }
            if (keep) chosen.push_back(c);
        }
        return chosen;
    }

    // Links `from` -> `to` on `layer`; if `from` is now over capacity, re-selects
    // its best links with the same heuristic.
    void add_backlink(Id from, Id to, int layer) {
        std::vector<Id>& list = links(from, layer);
        list.push_back(to);
        const std::size_t cap = max_links(layer);
        if (list.size() <= cap) return;

        std::vector<Neighbor> cands;
        cands.reserve(list.size());
        for (Id x : list) cands.push_back({x, dist_(vec(from), vec(x), dim_)});
        std::sort(cands.begin(), cands.end());

        list.clear();
        for (const Neighbor& n : select_neighbors(cands, cap)) list.push_back(n.id);
    }

    // Wires a freshly stored node into the graph, layer by layer.
    void connect(Id id, int level) {
        const float* q = vec(id);

        // Layers above the new node's level: just find a good entry point.
        const Id start = greedy_closest(q, entry_, max_level_, level);

        std::vector<Id> entries{start};
        for (int layer = std::min(level, max_level_); layer >= 0; --layer) {
            const std::vector<Neighbor> found =
                search_layer(q, entries, params_.ef_construction, layer, false);
            const std::vector<Neighbor> chosen = select_neighbors(found, params_.M);

            std::vector<Id>& mine = links(id, layer);
            for (const Neighbor& n : chosen) mine.push_back(n.id);
            for (const Neighbor& n : chosen) add_backlink(n.id, id, layer);

            // The whole candidate list seeds the search on the next layer down.
            entries.clear();
            for (const Neighbor& n : found) entries.push_back(n.id);
        }
    }

    void check_vector(std::span<const float> v, const char* where) const {
        if (v.size() != dim_) {
            throw std::invalid_argument(std::string("HnswIndex::") + where +
                                        ": dimension mismatch");
        }
        for (float x : v) {
            if (!std::isfinite(x)) {
                throw std::invalid_argument(std::string("HnswIndex::") + where +
                                            ": non-finite value");
            }
        }
    }


    std::size_t dim_;
    Metric metric_;
    DistanceFn dist_;
    HnswParams params_;
    double level_mult_ = 0.0;
    std::mt19937_64 rng_;

    std::vector<float> data_;                          // slot_count() * dim_ floats
    std::vector<std::uint8_t> levels_;                 // top layer of each node
    std::vector<std::uint8_t> deleted_;                // 1 = tombstone
    std::vector<std::vector<std::vector<Id>>> links_;  // [node][layer] -> neighbors
    std::size_t live_count_ = 0;
    Id entry_ = 0;
    int max_level_ = -1;  // -1 = empty
};

}  