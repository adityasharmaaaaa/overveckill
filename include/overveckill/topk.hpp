#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "overveckill/types.hpp"

namespace overveckill{

class TopK{
public:
    explicit TopK(std::size_t k) : k_(k) {heap_.reserve(k);}

    std::size_t capacity() const noexcept { return k_; }
    std::size_t size() const noexcept { return heap_.size(); }
    bool full() const noexcept { return heap_.size()>=k_; }

    float threshold() const noexcept {
        if(k_ == 0) return -std::numeric_limits<float>::infinity();
        return full() ? heap_.front().distance : std::numeric_limits<float>::infinity();
    }

    bool push(Neighbor n){
        if(k_ == 0) return false;
        if(heap_.size()<k_) {
            heap_.push_back(n);
            std::push_heap(heap_.begin(),heap_.end());
            return true;
        }
        if(!(n<heap_.front())) return false;
        std::pop_heap(heap_.begin(), heap_.end());
        heap_.back()=n;
        std::push_heap(heap_.begin(), heap_.end());
        return true;
    }

    std::vector<Neighbor> take_sorted(){
        std::sort_heap(heap_.begin(),heap_.end());
        std::vector<Neighbor>out=std::move(heap_);
        heap_.clear();
        return out;
    }

private:
    std::size_t k_;
    std::vector<Neighbor> heap_;
};

}