#pragma once

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace ovk_test {

struct TestCase {
    const char* name;
    void (*fn)();
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

struct Registrar {
    Registrar(const char* name, void (*fn)()) { registry().push_back({name, fn}); }
};

struct RequireFailure {};

inline int& current_failures() {
    static int n = 0;
    return n;
}

inline void report_failure(const char* file, int line, const std::string& msg) {
    std::cerr << "    " << file << ":" << line << ": " << msg << "\n";
    ++current_failures();
}

template <typename T>
std::string to_str(const T& v) {
    std::ostringstream os;
    os << v;
    return os.str();
}

}  

#define OVK_TEST(name)                                                  \
    static void name();                                                 \
    static ::ovk_test::Registrar ovk_registrar_##name(#name, &name);    \
    static void name()

// Non-fatal: records the failure and keeps running the test.
#define OVK_CHECK(expr)                                                 \
    do {                                                                \
        if (!(expr))                                                    \
            ::ovk_test::report_failure(__FILE__, __LINE__,              \
                                       "CHECK failed: " #expr);         \
    } while (0)

// Fatal: records the failure and stops this test immediately.
#define OVK_REQUIRE(expr)                                               \
    do {                                                                \
        if (!(expr)) {                                                  \
            ::ovk_test::report_failure(__FILE__, __LINE__,              \
                                       "REQUIRE failed: " #expr);       \
            throw ::ovk_test::RequireFailure{};                         \
        }                                                               \
    } while (0)

// Equality with both values printed on failure.
#define OVK_CHECK_EQ(a, b)                                              \
    do {                                                                \
        const auto& ovk_a = (a);                                        \
        const auto& ovk_b = (b);                                        \
        if (!(ovk_a == ovk_b))                                          \
            ::ovk_test::report_failure(                                 \
                __FILE__, __LINE__,                                     \
                "CHECK_EQ failed: " #a " == " #b " (" +                 \
                    ::ovk_test::to_str(ovk_a) + " vs " +                \
                    ::ovk_test::to_str(ovk_b) + ")");                   \
    } while (0)

// Floating point comparison with an absolute tolerance.
#define OVK_CHECK_NEAR(a, b, eps)                                       \
    do {                                                                \
        const double ovk_a = static_cast<double>(a);                    \
        const double ovk_b = static_cast<double>(b);                    \
        if (!(std::fabs(ovk_a - ovk_b) <= (eps)))                       \
            ::ovk_test::report_failure(                                 \
                __FILE__, __LINE__,                                     \
                "CHECK_NEAR failed: " #a " ~ " #b " (" +                \
                    ::ovk_test::to_str(ovk_a) + " vs " +                \
                    ::ovk_test::to_str(ovk_b) + ")");                   \
    } while (0)