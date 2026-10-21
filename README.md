# overveckill

a vector database in c++, written from scratch. yes, it was probably unnecessary.

i wanted to understand what actually happens inside a vector search instead of just calling one, so i built the pieces: distance functions, top-k, an exact (flat) index, a vector store, and an HNSW index. header-only, C++20, no dependencies.

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

## limitations

- index build is slow, about 20s for 20k vectors
- no persistence, the index lives in memory only
- no delete or update
- single-threaded, scalar distance code (no SIMD)
- it's a library, not a server

## todo

- save / load an index
- delete and update
- SIMD distance kernels
- multithreaded build and search
- benchmarks on larger sets and real embeddings
- comparison against hnswlib and faiss

## status

work in progress, v0.2.0. not benchmarked against other libraries yet.