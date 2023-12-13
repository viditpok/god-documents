cmake_minimum_required(VERSION 3.13)

project("project4" C)

set(CC /usr/bin/gcc)
set (CMAKE_C_FLAGS "-Wall -Wextra -Wsign-conversion -Wpointer-arith -Wcast-qual -Wwrite-strings -Wshadow -Wmissing-prototypes -Wpedantic -Wwrite-strings -g -std=gnu99 -lm")
file(GLOB sources src/*)
include_directories(src)

add_executable(cs2200_project4 ${sources})
