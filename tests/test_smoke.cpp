#include <string>

#include "overveckill/version.hpp"
#include "test_framework.hpp"

OVK_TEST(version_is_0_1_0) {
    OVK_CHECK_EQ(overveckill::version_major, 0);
    OVK_CHECK_EQ(overveckill::version_minor, 1);
    OVK_CHECK_EQ(overveckill::version_patch, 0);
}

OVK_TEST(name_and_tagline) {
    OVK_CHECK_EQ(std::string(overveckill::name), std::string("overveckill"));
    OVK_CHECK(std::string(overveckill::tagline).find("c++") != std::string::npos);
}

OVK_TEST(float_comparison_works) {
    OVK_CHECK_NEAR(0.1 + 0.2, 0.3, 1e-9);
}