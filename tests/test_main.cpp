#include <exception>
#include <iostream>
#include <string>

#include "test_framework.hpp"

// Usage: ovk_tests [substring]   (runs only tests whose name contains it)
int main(int argc, char** argv) {
    const std::string filter = argc > 1 ? argv[1] : "";
    int ran = 0;
    int failed = 0;

    for (const auto& t : ovk_test::registry()) {
        if (!filter.empty() && std::string(t.name).find(filter) == std::string::npos) {
            continue;
        }
        ++ran;
        ovk_test::current_failures() = 0;
        std::cout << "[ RUN  ] " << t.name << "\n";

        try {
            t.fn();
        } catch (const ovk_test::RequireFailure&) {
            // already reported
        } catch (const std::exception& e) {
            std::cerr << "    uncaught exception: " << e.what() << "\n";
            ++ovk_test::current_failures();
        } catch (...) {
            std::cerr << "    uncaught unknown exception\n";
            ++ovk_test::current_failures();
        }

        if (ovk_test::current_failures() == 0) {
            std::cout << "[  OK  ] " << t.name << "\n";
        } else {
            std::cout << "[ FAIL ] " << t.name << "\n";
            ++failed;
        }
    }

    std::cout << "\n" << (ran - failed) << "/" << ran << " tests passed\n";
    return failed == 0 ? 0 : 1;
}