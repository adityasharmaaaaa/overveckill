#pragma once

#include <optional>
#include <string_view>

namespace overkill{

enum class Metric{
    L2,
    InnerProduct,
    Cosine,
};

inline constexpr const char* to_string(Metric m){
    switch(m){
        case Metric::L2: return "l2";
        case Metric::InnerProduct: return "ip";
        case Metric::Cosine: return "cosine";
    }
    return "unknown";
}

inline std::optional<Metric> parse_metric(std::string_view s){
    if(s=="l2") return Metric::L2;
    if(s=="ip") return Metric::InnerProduct;
    if(s=="cosine") return Metric::Cosine;
    return std::nullopt;
}

}