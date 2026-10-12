#include <iostream>

#include "overveckill/version.hpp"

int main() {
    std::cout << overveckill::name << " v"
              << overveckill::version_major << "."
              << overveckill::version_minor << "."
              << overveckill::version_patch << "\n"
              << overveckill::tagline << "\n";
    return 0;
}