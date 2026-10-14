#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>
#include <string>

#include "overveckill/distance.hpp"
#include "overveckill/metric.hpp"
#include "overveckill/topk.hpp"
#include "overveckill/types.hpp"

namespace overveckill{

//exact nearest neighbor index

//compares the query against every stored vector.
//slow but always correct. used as ground truth for approximate indexes
//hnsw and recall auditor are mesasured against this

//vectors are stored contiguously in one buffer for cache friendly scans

class FlatIndex{
public:
    explicit FlatIndex(std::size_t dim, Metric metric = Metric::L2)
        : dim_(dim), metric_(metric), dist_(get_distance_fn(metric)){
            if(dim==0) throw std::invalid_argument("Flat index: dim must be > 0");
    }

    std::size_t dim() const noexcept { return dim_; }
    Metric metric() const noexcept { return metric_; }
    std::size_t size() const noexcept { return data_.size()/dim_;}

    //appends a vector and return its id (ids are 0,1,2,3,.....)
    Id add(std::span<const float> v){
        check_vector(v,"add");
        if(size()>=std::numeric_limits<Id>::max()){
            throw std::length_error("Flat index: id space exhausted");
        }
        const Id id = static_cast<Id>(size());
        data_.insert(data_.end(), v.begin(), v.end());
        return id;
    }

    std::span<const float> get(Id id) const{
        if(id>=size()) throw std::out_of_range("Flat index:: get: bad id");
        return {data_.data()+static_cast<std::size_t>(id)*dim_, dim_};
    }

    //returns k nearest vectors, closest first
    //return fewer than k if index holds then fewer than k vectors
    std::vector<Neighbor> search(std::span<const float>query, std::size_t k)const{
        check_vector(query,"search");
        const std::size_t n = size();
        TopK top(std::min(k,n));
        const float* base = data_.data();
        for(std::size_t i=0; i<n; i++){
            top.push({static_cast<Id>(i),dist_(query.data(),base+i*dim_,dim_)});
        }
        return top.take_sorted();
    }

private:
     void check_vector(std::span<const float> v, const char* where) const {
        if (v.size() != dim_) {
            throw std::invalid_argument(std::string("FlatIndex::") + where +
                                        ": dimension mismatch");
        }
        for (float x : v) {
            if (!std::isfinite(x)) {
                throw std::invalid_argument(std::string("FlatIndex::") + where +
                                            ": non-finite value");
            }
        }
    }

    std::size_t dim_;
    Metric metric_;
    DistanceFn dist_;
    std::vector<float> data_;
};

}