#pragma once

#include "stats.h"
#include "util.h"

/* The time taken to read/write a byte to/from memory */
#define MEMORY_ACCESS_TIME 200
/* The time taken to read a page from the disk */
#define DISK_PAGE_READ_TIME 150000
/* The time taken to write a page to the disk */
#define DISK_PAGE_WRITE_TIME 250000

typedef struct stats_t {
    /* Number of memory accesses */
    uint64_t accesses;
    /* Accesses that result in a page fault */
    uint64_t page_faults;
    /* Writebacks to disk */
    uint64_t writebacks;
    /* Average Access Time */
    double amat;
} stats_t;

/* staticallly initiatlized to keep information about the memory operations */
extern stats_t stats;

void compute_stats(void);
