#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

    std::size_t slot_count() const noexcept { return data_.size()/dim_; }
    std::size_t size() const noexcept { return slot_count()-free_.size(); }

    bool contains(Id id) const noexcept { return id<slot_count() && live_[id]!=0; }

    Id add(std::span<const float> v){
        check_vector(v,"add");
        
        if(!free_.empty()){
            const Id id = free_.back();
            free_.pop_back();
            std::copy(v.begin(),v.end(),
                        data_.data() + static_cast<std::size_t>(id)*dim_);
            live_[id]=1;
            return id;
        }

        if(slot_count()>=std::numeric_limits<Id>::max()){
            throw std::length_error("Flatindex: id space exhausted");
        }

        const Id id = static_cast<Id>(slot_count());
        data_.insert(data_.end(),v.begin(),v.end());
        live_.push_back(1);
        return id;
    }

    bool remove(Id id){
        if(!contains(id)) return false;
        live_[id]=0;
        free_.push_back(id);
        return true;
    }

    std::span<const float> get(Id id) const{
        if(!contains(id)) throw std::out_of_range("Flat index:: get: bad or removed id");
        return {data_.data()+static_cast<std::size_t>(id)*dim_, dim_};
    }

    //returns k nearest vectors, closest first
    //return fewer than k if index holds then fewer than k vectors
    std::vector<Neighbor> search(std::span<const float> query, std::size_t k) const {
        check_vector(query, "search");
        const std::size_t slots = slot_count();
        TopK top(std::min(k, size()));  // capped so a huge k can't over-allocate
        const float* base = data_.data();
        for (std::size_t i = 0; i < slots; ++i) {
            if (!live_[i]) continue;
            top.push({static_cast<Id>(i), dist_(query.data(), base + i * dim_, dim_)});
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
    std::vector<std::uint8_t>live_;
    std::vector<Id>free_;
};

}