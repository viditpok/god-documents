#pragma once

#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/**
 * Simulates a kernel panic by printing a message and then immediately killing the simulation.
 */
static inline void panic(const char *message) {
    if (message != NULL)
        printf("panic: %s\n", message);
    printf("-> Hint: To debug a panic, use GDB.\n");
    printf("-> Set a breakpoint on the panic function and examine the backtrace to determine where it was called.\n");
    exit(EXIT_FAILURE);
}

/**
 * Calculates a random integer in a cross-platform predictable way.
 */
uint32_t prng_rand(void);
