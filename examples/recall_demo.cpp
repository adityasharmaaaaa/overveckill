#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "overveckill/flat_index.hpp"
#include "overveckill/hnsw_index.hpp"

using namespace overveckill;
using Clock = std::chrono::steady_clock;

static double micros_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

// Usage: recall_demo [n] [dim] [queries]
int main(int argc, char** argv) {
    const std::size_t n = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 20000;
    const std::size_t dim = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 32;
    const std::size_t nq = argc > 3 ? std::strtoul(argv[3], nullptr, 10) : 200;
    constexpr std::size_t k = 10;

    std::mt19937 rng(1234);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    auto make = [&] {
        std::vector<float> v(dim);
        for (auto& x : v) x = gauss(rng);
        return v;
    };

    FlatIndex flat(dim);
    HnswIndex hnsw(dim);

    std::cout << "n=" << n << " dim=" << dim << " queries=" << nq << " k=" << k
              << " M=" << hnsw.params().M << " ef_construction="
              << hnsw.params().ef_construction << "\n";

    const auto build_start = Clock::now();
    for (std::size_t i = 0; i < n; ++i) {
        const auto v = make();
        flat.add(v);
        hnsw.add(v);
    }
    std::cout << "built in " << micros_since(build_start) / 1e6 << " s (flat + hnsw), "
              << "graph valid: " << (hnsw.validate() ? "yes" : "NO") << "\n\n";

    std::vector<std::vector<float>> queries;
    std::vector<std::vector<Neighbor>> truth;
    const auto flat_start = Clock::now();
    for (std::size_t i = 0; i < nq; ++i) {
        queries.push_back(make());
        truth.push_back(flat.search(queries.back(), k));
    }
    const double flat_us = micros_since(flat_start) / static_cast<double>(nq);

    std::cout << std::left << std::setw(8) << "ef" << std::setw(12) << "recall@10"
              << std::setw(14) << "us/query" << "speedup vs flat\n";
    for (std::size_t ef : {10u, 20u, 40u, 80u, 160u, 320u}) {
        std::size_t hits = 0;
        const auto t0 = Clock::now();
        std::vector<std::vector<Neighbor>> results;
        for (const auto& q : queries) results.push_back(hnsw.search(q, k, ef));
        const double us = micros_since(t0) / static_cast<double>(nq);

        for (std::size_t i = 0; i < nq; ++i) {
            for (const auto& t : truth[i]) {
                for (const auto& r : results[i]) {
                    if (r.id == t.id) { ++hits; break; }
                }
            }
        }
        const double recall = static_cast<double>(hits) / static_cast<double>(nq * k);
        std::cout << std::left << std::setw(8) << ef << std::setw(12) << std::fixed
                  << std::setprecision(4) << recall << std::setw(14)
                  << std::setprecision(1) << us << std::setprecision(1) << flat_us / us
                  << "x\n";
    }
    std::cout << "\nflat (exact): " << std::fixed << std::setprecision(1) << flat_us
              << " us/query\n";
    return 0;
}