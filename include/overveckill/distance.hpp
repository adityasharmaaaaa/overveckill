#pragma once 

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>

#include "overveckill/metric.hpp"

namespace overveckill{

using DistanceFn = float (*)(const float* a, const float* b, std::size_t dim);

namespace scalar{

inline float l2_squared(const float* a, const float* b, std::size_t dim) noexcept{
    float sum=0.0f;
    for(std::size_t i=0; i<dim; i++){
        const float d = a[i]-b[i];
        sum+=d*d;
    }
    return sum;
}

inline float dot(const float* a, const float* b, std::size_t dim) noexcept{
    float sum=0.0f;
    for(std::size_t i=0; i<dim; i++){
        sum+=a[i]*b[i];
    }
    return sum;
}

inline float neg_inner_product(const float* a, const float* b, std::size_t dim) noexcept{
    return -dot(a,b,dim);
}

inline float cosine_distance(const float* a, const float* b, std::size_t dim) noexcept{
    float ab=0.0f;
    float aa=0.0f;
    float bb=0.0f;
    for(std::size_t i=0; i<dim; i++){
        ab+=a[i]*b[i];
        aa+=a[i]*a[i];
        bb+=b[i]*b[i];
    }

    if(aa==0.0f || bb==0.0f) return 1.0f;
    const float similarity=ab/(std::sqrt(aa)*std::sqrt(bb));

    return std::clamp(1.0f-similarity,0.0f,2.0f);
}
}

inline DistanceFn get_distance_fn(Metric m) noexcept{
    switch (m){
        case Metric::L2: return &scalar::l2_squared;
        case Metric::InnerProduct: return &scalar::neg_inner_product;
        case Metric::Cosine: return &scalar::cosine_distance;
    }
    return nullptr;
}

inline float distance(Metric m, std::span<const float> a, std::span<const float> b){
    if(a.size()!=b.size()){
        throw std::invalid_argument("distance: dimension mismatch");
    }
    return get_distance_fn(m)(a.data(),b.data(),a.size());
}

}