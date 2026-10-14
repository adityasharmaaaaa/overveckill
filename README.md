# overveckill

> vector store written in c++ from scratch 
(yes, it was probably unnecessary)

## Status

Work in progress.

- [x] Distance metrics (L2, inner product, cosine)
- [x] Top-k heap
- [x] Exact flat index (ground truth)
- [ ] Store layer (external ids, metadata, delete)
- [ ] HNSW
- [ ] Recall auditor, explain mode

## Build

    cmake -S . -B build
    cmake --build build
    ./build/hello

## Test

    ./build/ovk_tests