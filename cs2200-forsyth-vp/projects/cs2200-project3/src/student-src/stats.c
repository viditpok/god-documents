#include "stats.h"

/* The stats. See the definition in stats.h. */
stats_t stats;

/**
 * --------------------------------- PROBLEM 10 --------------------------------------
 * Checkout PDF section 10 for this problem
 * 
 * Calulate the total average time it takes for an access
 * 
 * HINTS:
 * 		- You may find the #defines in the stats.h file useful.
 * 		- You will need to include code to increment many of these stats in
 * 		the functions you have written for other parts of the project.
 * -----------------------------------------------------------------------------------
 */
void compute_stats() {
    stats.amat = (double) ((stats.page_faults * DISK_PAGE_READ_TIME + stats.writebacks * DISK_PAGE_WRITE_TIME + stats.accesses * MEMORY_ACCESS_TIME) / ((double) stats.accesses));
}
