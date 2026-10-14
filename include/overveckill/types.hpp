#pragma once

#include <cstdint>

namespace overveckill {

using Id = std::uint32_t;

struct Neighbor{
    Id id;
    float distance;

    friend bool operator<(const Neighbor& a, const Neighbor& b) noexcept{
        return a.distance < b.distance || (a.distance == b.distance && a.id < b.id);
    }

    friend bool operator==(const Neighbor&, const Neighbor&) = default;
};

}