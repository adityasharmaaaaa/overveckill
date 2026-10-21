cmake_minimum_required(VERSION 3.16)
project(overveckill VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(OVK_BUILD_TESTS "Build tests" ON)
option(OVK_BUILD_EXAMPLES "Build examples" ON)
option(OVK_SANITIZE "Enable ASan + UBSan" OFF)

add_library(overveckill INTERFACE)
target_include_directories(overveckill INTERFACE include)
target_compile_features(overveckill INTERFACE cxx_std_20)

function(ovk_configure target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4)
  else()
    target_compile_options(${target} PRIVATE
      -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    if(OVK_SANITIZE)
      target_compile_options(${target} PRIVATE
        -fsanitize=address,undefined -fno-omit-frame-pointer)
      target_link_options(${target} PRIVATE -fsanitize=address,undefined)
    endif()
  endif()
endfunction()

if(OVK_BUILD_EXAMPLES)
  add_executable(hello examples/hello.cpp)
  target_link_libraries(hello PRIVATE overveckill)
  ovk_configure(hello)
  add_executable(recall_demo examples/recall_demo.cpp)
  target_link_libraries(recall_demo PRIVATE overveckill)
  ovk_configure(recall_demo)
endif()


if(OVK_BUILD_TESTS)
  enable_testing()
  add_executable(ovk_tests
    tests/test_main.cpp
    tests/test_smoke.cpp
    tests/test_distance.cpp
    tests/test_topk.cpp
    tests/test_flat_index.cpp
    tests/test_store.cpp
    tests/test_hnsw.cpp)
  target_link_libraries(ovk_tests PRIVATE overveckill)
  target_include_directories(ovk_tests PRIVATE tests)
  ovk_configure(ovk_tests)
  add_test(NAME ovk_tests COMMAND ovk_tests)
endif()