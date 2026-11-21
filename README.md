# overveckill

a vector database in c++, written from scratch. yes, it was probably unnecessary.

i wanted to understand what actually happens inside a vector search instead of just calling one, so i built the pieces: distance functions, top-k, an exact (flat) index, a vector store, an HNSW index, and save/load to disk. header-only, C++20, no dependencies.

## results

`examples/recall_demo.cpp` builds the flat and HNSW indexes on the same data and checks HNSW's answers against the exact ones. `ef` is how much of the graph HNSW explores per query.

```
n=20000 dim=32 queries=200 k=10 M=16 ef_construction=200
built in 19.8133 s (flat + hnsw), graph valid: yes

ef      recall@10   us/query      speedup vs flat
10      0.5250      315.0         7.3x
20      0.7020      360.0         6.4x
40      0.8610      484.0         4.7x
80      0.9575      612.6         3.7x
160     0.9910      933.6         2.5x
320     0.9975      1520.3        1.5x

flat (exact): 2290.9 us/query
```

at `ef=80` it finds about 96% of the true neighbours at 3.7x the speed of brute force. at `ef=160` it's 99% at 2.5x. raising `ef` trades speed for accuracy and converges on exact search. the demo also validates the graph structure after building it.

this is a small test (20k random vectors, 32 dims, MacBook Air) where brute force is still fast, so the speedups are modest. bigger runs are on the list.

## build

```
cmake -S . -B build
cmake --build build
./build/hello
./build/recall_demo
./build/ovk_tests
```

to build with ASan + UBSan:

```
cmake -S . -B build -DOVK_SANITIZE=ON
```

warnings are on by default: `-Wall -Wextra -Wpedantic -Wconversion -Wshadow`.

## using it

it's an `INTERFACE` library, so from another cmake project:

```cmake
add_subdirectory(overveckill)
target_link_libraries(your_thing PRIVATE overveckill)
```

requires C++20.

## saving and loading

the flat index, the HNSW index and the store can all be written to disk and read back:

```cpp
#include "overveckill/hnsw_index.hpp"
#include "overveckill/store.hpp"

using namespace overveckill;

BasicStore<HnswIndex> store(HnswIndex(384, Metric::Cosine));
store.upsert("doc-1", embedding, {{"lang", std::string("en")}});
store.save("docs.ovk");

auto again = BasicStore<HnswIndex>::load("docs.ovk");
```

`FlatIndex` and `HnswIndex` have the same `save(path)` / `load(path)`, and there are overloads that take a `std::ostream` / `std::istream`.

the format is a small custom binary one:

- little-endian on every platform, so files move between machines
- each object starts with a magic (`OVKF` flat, `OVKH` hnsw, `OVKS` store) and a format version
- the HNSW graph is saved as-is, including its random generator state. a loaded index gives exactly the same search results as the original, and adding to it builds the same graph the original would have
- saves go to `<path>.tmp` and are then renamed, so a crash mid-save doesn't leave a half-written file
- loading checks everything and throws `SerializationError` on a bad magic, an unknown version, truncated data, or a graph that fails validation. a corrupt length field can't make it allocate huge amounts of memory

## limitations

- index build is slow, about 20s for 20k vectors
- HNSW deletes are tombstones: removed nodes stay in the graph and their memory is never reclaimed
- save/load is a full snapshot. there's no write-ahead log or incremental save, so anything after the last save is lost on a crash
- no filtering on metadata during search
- single-threaded, scalar distance code (no SIMD)
- it's a library, not a server

## todo

- compaction / graph repair for HNSW deletes
- metadata-filtered search
- memory-mapped loading for big indexes
- SIMD distance kernels
- multithreaded build and search
- benchmarks on larger sets and real embeddings
- comparison against hnswlib and faiss

## status

work in progress, v0.3.0. not benchmarked against other libraries yet.