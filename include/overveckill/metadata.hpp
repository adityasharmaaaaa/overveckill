#pragma once 

#include <cstdint>
#include <map>
#include <string>
#include <variant>

namespace overveckill{

//a typed metadata value. Types matter: filtered search needs to
//compare "year >= 2026" as numbers, not as strings.
using MetaValue = std::variant<std::int64_t, double, bool, std::string>;

// std::map keeps keys sorted, so iteration order and operator== are deterministic.
using Metadata = std::map<std::string,MetaValue>;

}